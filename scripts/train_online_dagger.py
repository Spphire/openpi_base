"""
Online DAgger training script for OpenPi.

This script implements online DAgger training with adaptive sampling between
offline and online data, fetching new episodes from a data cloud service.

Author: Wendi Chen
"""
import os
import sys
sys.path.append(os.getcwd())

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.training.online_data_fetcher import OnlineDataFetcher


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    online_config: _config.OnlineDaggerTrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        combined_config = {
            **dataclasses.asdict(config),
            "online_dagger": dataclasses.asdict(online_config),
        }
        wandb.init(
            name=f"{config.exp_name}_dagger",
            config=combined_config,
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def train_step_with_per_sample_loss(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array], at.Array]:
    """Train step that also returns per-sample losses for adaptive sampling."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_sample_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info, per_sample_loss


def main(args: _config.OnlineDaggerTrainConfig):
    init_logging()
    logging.info(f"Running Online DAgger on: {platform.node()}")

    # Get base training config
    config = _config.get_config(args.config_name)
    if args.exp_name:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.overwrite:
        config = dataclasses.replace(config, overwrite=True)
    if args.resume:
        config = dataclasses.replace(config, resume=True)
    if args.init_checkpoint:
        logging.info(f"Using custom init checkpoint: {args.init_checkpoint}")
        config = dataclasses.replace(
            config, weight_loader=_weight_loaders.CheckpointWeightLoader(params_path=args.init_checkpoint)
        )

    # Use online dagger config directly from args
    online_config = args

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, online_config, resuming=resuming, enabled=config.wandb_enabled)

    # Create hybrid data loader
    data_loader = _data_loader.create_hybrid_data_loader(
        config,
        online_repo_id=online_config.online_repo_id,
        sharding=data_sharding,
        window_size=online_config.window_size,
        boost_factor=online_config.boost_factor,
        min_online_ratio=online_config.min_online_ratio,
        max_online_ratio=online_config.max_online_ratio,
        initial_online_weight=online_config.initial_online_weight,
        robot_type=online_config.robot_type,
        fps=online_config.fps,
        features=online_config.features,
    )
    data_iter = iter(data_loader)
    batch_with_flags = next(data_iter)

    # Unpack batch (observation, actions, is_online)
    if len(batch_with_flags) == 3:
        observation, actions, is_online = batch_with_flags
    else:
        observation, actions = batch_with_flags
        is_online = None

    batch = (observation, actions)
    logging.info(f"Initialized hybrid data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # Initialize data fetcher
    data_fetcher = OnlineDataFetcher(
        datacloud_endpoint=online_config.datacloud_endpoint,
        identifier=online_config.identifier,
        query_filter=online_config.query_filter,
        robot_type=online_config.robot_type,
        fps=online_config.fps,
        task_description=online_config.task_description,
        features=online_config.features,
        use_absolute_action=online_config.use_absolute_action,
        action_type=online_config.action_type,
        temporal_downsample_ratio=online_config.temporal_downsample_ratio,
        use_dino=online_config.use_dino,
        episode_clip_head_seconds=online_config.episode_clip_head_seconds,
        episode_clip_tail_seconds=online_config.episode_clip_tail_seconds,
        gripper_width_bias=online_config.gripper_width_bias,
        gripper_width_scale=online_config.gripper_width_scale,
    )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    hybrid_dataset = data_loader.dataset

    for step in pbar:
        # Fetch new episodes periodically
        if step > 0 and step % online_config.fetch_interval == 0:
            new_episodes = data_fetcher.fetch_new_episodes()
            if new_episodes:
                hybrid_dataset.append_episodes(new_episodes, task=online_config.task_description)
                logging.info(f"Added {len(new_episodes)} new episodes at step {step}")

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        # Update adaptive sampler weights based on loss
        if is_online is not None:
            loss_value = float(jax.device_get(info["loss"]))
            is_online_np = np.array(jax.device_get(is_online))

            # Update sampler with average losses
            n_online = np.sum(is_online_np)
            n_offline = len(is_online_np) - n_online

            if n_online > 0:
                hybrid_dataset.adaptive_sampler.add_loss(loss_value, is_online=True)
            if n_offline > 0:
                hybrid_dataset.adaptive_sampler.add_loss(loss_value, is_online=False)

            hybrid_dataset.adaptive_sampler.update_weights()

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Add online dagger specific metrics
            sampling_stats = hybrid_dataset.get_sampling_stats()
            reduced_info.update(
                {
                    "online_weight": sampling_stats["online_weight"],
                    "offline_weight": sampling_stats["offline_weight"],
                    "online_loss_mean": sampling_stats["online_loss_mean"],
                    "offline_loss_mean": sampling_stats["offline_loss_mean"],
                    "online_episodes": hybrid_dataset.online_episodes_count,
                    "offline_episodes": hybrid_dataset.offline_episodes_count,
                    "online_samples": hybrid_dataset.online_len,
                    "offline_samples": hybrid_dataset.offline_len,
                    "fetched_episodes": data_fetcher.fetched_count,
                }
            )

            info_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        # Get next batch
        batch_with_flags = next(data_iter)
        if len(batch_with_flags) == 3:
            observation, actions, is_online = batch_with_flags
        else:
            observation, actions = batch_with_flags
            is_online = None
        batch = (observation, actions)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    args = tyro.cli(_config.OnlineDaggerTrainConfig)
    main(args)
