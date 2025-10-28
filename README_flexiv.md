# TODO
- 修正state的定义：因为机器拿到的state和训练集state定义不一样。根据diffusion policy中把state转到了relative pose，因此在该框架中需要把state转为上一次的delta 动作；
- ~~增加模型推理可视化脚本和ground truth的对比~~ ✔
- 推理pipeline接入SuperInference 
- ~~训练过程中加入推理可视化记录~~ ✔
- ~~确认抓夹的归一化方式，以及抓夹是否使用delta: 抓夹用的绝对动作，应该在数据预处理阶段避免对它的delta计算~~ ✔ 
- ~~加入zarr转lerobot数据集~~ ✔

# 说明
# Usage
0. 安装
按照README.md安装
注：由于服务器是阿里云服务器，需要在每条安装命令前加上 UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple 来加速

额外依赖
```
imagecodecs==2023.9.18
zarr==2.16.1
numcodecs==0.11.0
```
1. 数据转换：
```shell
# 注意1：转换的数据集会被存到环境变量HF_LEROBOT_HOME对应的路径下，若不存在，则默认为~/.cache/huggingface/lerobot
# 注意2：config中要写清楚task
python preprocess_data/h5_to_lerobot.py CONFIG_PATH REPO_NAME --fps FPS --robot_type ROBOT_NAME

# Example from h5
python preprocess_data/h5_to_lerobot.py preprocess_data/configs/debug.yaml flexiv/pick_dolls_debug --fps 25 --robot_type bimanual_flexiv

# Example from zarr
python preprocess_data/zarr_to_lerobot.py preprocess_data/configs/debug_zarr.yaml flexiv/pick_1014 --fps 25 --robot_type bimanual_flexiv
```

2. 编写配置文件
- 如果有新的机器人、输出空间变化、观测空间变化，则需要在src/openpi/policies/下新建 xxx_policy.py，并在其中定义生成伪batch、Input空间、Output空间，并将该policy在src/openpi/training/config.py中import

- 在src/openpi/training/config.py中创建两种config：DataConfig和TrainingConfig
```shell
# DataConfig例子，其中若新采集的数据格式无变化，则可以沿用之前的DataConfig
@dataclasses.dataclass(frozen=True)
class LeRobotBimanualFlexivDataConfig(DataConfigFactory):

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *args, **kwargs) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/left_wrist_image": "left_wrist_image",
                        "observation/right_wrist_image": "right_wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[bimanual_flexiv_policy.BimanualFlexivInputs(model_type=model_config.model_type)],
            outputs=[bimanual_flexiv_policy.BimanualFlexivOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

# TrainingConfig例子，绝大多数情况需要重写
_CONFIGS = [
    ...,
    # 新增加的training config
    TrainConfig(
        name="pi05_pick1010all", # change config name here
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False), # change chunk size here
        data=LeRobotBimanualFlexivDataConfig(
            repo_id="flexiv/pick_1010all", # change dataset name here
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=True,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    ...
```

3. 生成统计信息

```shell
python scripts/compute_norm_stats.py --config-name pi05_pick1010all # 跟training config中定义的名称保持一致
```

4. 训练
```shell
python scripts/train.py pi05_pick1010all --exp-name=EXP_NAME --overwrite # pi05_pick1010all可以被替换成其他定义好的training config
```

5. 可视化训练结果
```shell
python model_inference_visualization/visualize_inference.py -cfg pi05_pick1010all -ckpt CKPT_PATH --output-dir OURPUT_DIR # pi05_flexiv_pick可以被替换成其他定义好的training config
```

6. 训练中断resume
```shell
python scripts/train.py pi05_pick1010all --exp-name=exp_pi05_1010_all --resume # config名称和实验名称保持相同
```