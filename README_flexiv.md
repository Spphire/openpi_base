# TODO
- 增加模型推理可视化脚本和ground truth的对比
- 推理pipeline接入SuperInference
- 测试policy server的功能
- 训练过程中加入推理可视化脚本

# 说明

## 数据格式
```python
episode_data['left_wrist_image'] = img_left[:-1,:]
episode_data['right_wrist_image'] = img_right[:-1,:]
state = np.concatenate(
    [episode_data['robot0_eef_pos'], 
    episode_data['robot0_eef_rot_axis_angle'], 
    episode_data['robot0_gripper_width'], 
    episode_data['robot1_eef_pos'], 
    episode_data['robot1_eef_rot_axis_angle'],
    episode_data['robot1_gripper_width']], axis=1).astype(np.float32)
episode_data['state'] = state[:-1,:]
episode_data['actions'] = state[1:,:] - state[:-1,:]
```

**动作空间**：state_{t+1} - state_{t} -> delta动作空间

**观测空间**：
 - state为左右臂末端位姿和抓夹的拼接, i.e., proprio
 - 图像为左右臂，无三方视图

# Usage
0. 安装
按照README.md安装

1. 数据转换：
```shell
python preprocess_data/h5_to_lerobot.py CONFIG_PATH REPO_NAME --fps FPS --robot_type ROBOT_NAME

# Example
python preprocess_data/h5_to_lerobot.py preprocess_data/configs/debug.yaml flexiv/pick_dolls_debug --fps 25 --robot_type bimanual_flexiv
```

2. 编写配置文件
- 如果有新的机器人、输出空间变化、观测空间变化，则需要在src/openpi/policies/下新建 xxx_policy.py，并在其中定义生成伪batch、Input空间、Output空间，并将该policy在src/openpi/training/config.py中import

- 在src/openpi/training/config.py中创建两种config：DataConfig和TrainingConfig
```shell
# DataConfig例子，其中若新采集的数据格式无变化，则可以沿用之前的DataConfig
@dataclasses.dataclass(frozen=True)
class LeRobotBimanualFlexivDataConfig(DataConfigFactory):
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
        name="pi05_flexiv_pick",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotBimanualFlexivDataConfig(
            repo_id="flexiv/pick_dolls_debug",
            base_config=DataConfig(prompt_from_task=False),
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
python scripts/compute_norm_stats.py --config-name pi05_flexiv_pick # 跟training config中定义的名称保持一致
```

4. 训练
```shell
python scripts/train.py pi05_flexiv_pick --exp-name=EXP_NAME --overwrite # pi05_flexiv_pick可以被替换成其他定义好的training config
```