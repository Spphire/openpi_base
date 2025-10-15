# 训练过程轨迹可视化

本文档介绍如何在训练过程中启用轨迹可视化功能，将真实动作（ground truth）和预测动作的对比图保存到本地文件。

## 功能特性

### 1. 轨迹对比图
- **2D轨迹对比**: 显示每个关节维度的真实值vs预测值时间序列
- **3D轨迹可视化**: 显示左右手在3D空间中的运动轨迹
- **分手显示**: 分别显示左手和右手的轨迹

### 2. 轨迹指标
- **MSE (Mean Squared Error)**: 整体轨迹均方误差
- **MAE (Mean Absolute Error)**: 整体轨迹平均绝对误差  
- **分手位置误差**: 左手和右手位置的独立MSE指标
- **轨迹长度和维度**: 用于验证数据完整性

### 3. 本地文件保存
- **自动保存**: 每5000步自动保存轨迹可视化
- **结构化存储**: 保存到checkpoint目录下的`trajectory_visualizations`文件夹
- **最小内存占用**: 直接保存到文件，不占用wandb内存

## 使用方法

### 自动启用
轨迹可视化功能已经集成到训练脚本中，无需额外配置即可使用：

```bash
python scripts/train.py -cfg your_config_name
```

### 文件结构
可视化文件将保存在以下结构中：

```
your_checkpoint_dir/
└── trajectory_visualizations/
    ├── step_00005000/
    │   ├── trajectory_comparison.png
    │   ├── 3d_trajectories.png
    │   └── trajectory_metrics.txt
    ├── step_00010000/
    │   ├── trajectory_comparison.png
    │   ├── 3d_trajectories.png
    │   └── trajectory_metrics.txt
    └── ...
```

### 配置参数
可以通过修改以下参数来调整可视化行为：

```python
# 在 src/openpi/training/trajectory_visualizer.py 中
initialize_trajectory_visualizer(
    action_dim=14,        # 动作维度 (默认14)
    save_interval=5000    # 保存间隔 (默认5000步)
)
```

### Wandb指标记录
轨迹指标会自动记录到wandb：

- `trajectory_mse`: 整体轨迹均方误差
- `trajectory_mae`: 整体轨迹平均绝对误差
- `left_hand_pos_mse`: 左手位置均方误差
- `right_hand_pos_mse`: 右手位置均方误差
- `trajectory_length`: 轨迹长度
- `action_dim`: 动作维度

## 技术实现

### 最小化修改
对train.py的修改极其简洁：

```python
# 只添加了一行导入
from openpi.training.trajectory_visualizer import maybe_save_training_trajectories

# 在训练循环中只添加了几行代码
observation, actions = batch
model = nnx.merge(train_state.model_def, train_state.params)
traj_metrics = maybe_save_training_trajectories(
    step, model, train_rng, observation, actions, str(config.checkpoint_dir)
)
if traj_metrics:  # Only log if we actually computed metrics
    wandb.log(traj_metrics, step=step)
```

### 数据格式
系统假设以下数据格式：
- **动作维度**: 14维 (前7维左手，后7维右手)
- **每只手格式**: `[abs_x, abs_y, abs_z, abs_rx, abs_ry, abs_rz, abs_gripper_width]`
- **状态格式**: 与动作相同的14维格式

### 轨迹计算
- **绝对位置**: 动作中的位置已经是绝对坐标，无需累积计算
- **初始状态**: 从observation.state中提取初始位置
- **轨迹构建**: `trajectory[0] = initial_state`, `trajectory[1:] = action_positions`

### 性能优化
- **条件执行**: 只在需要保存时才进行模型推理
- **内存友好**: 直接保存到文件，不占用内存
- **错误隔离**: 可视化错误不会中断训练
- **最小开销**: 大部分时间只有一个简单的步数检查

## 故障排除

### 常见问题

1. **内存不足**
   ```
   解决方案: 增加save_interval参数，减少可视化频率
   initialize_trajectory_visualizer(save_interval=10000)
   ```

2. **推理失败**
   ```
   现象: 控制台出现 "Failed to save trajectory visualizations"
   解决方案: 检查模型的sample_actions方法是否正确实现
   ```

3. **文件权限问题**
   ```
   现象: 无法创建可视化文件夹
   解决方案: 确保checkpoint目录有写权限
   ```

### 调试模式
可以通过修改save_interval来调试：

```python
# 更频繁的保存用于调试
initialize_trajectory_visualizer(save_interval=100)
```

## 自定义配置

### 修改保存间隔
在trajectory_visualizer.py中修改默认参数：

```python
def initialize_trajectory_visualizer(action_dim: int = 14, save_interval: int = 10000):
    # 改为每10000步保存一次
```

### 自定义可视化
可以继承`TrainingTrajectoryVisualizer`类来添加自定义可视化：

```python
class CustomTrajectoryVisualizer(TrainingTrajectoryVisualizer):
    def create_custom_plot(self, gt_actions, pred_actions):
        # 自定义可视化逻辑
        pass
```

### 不同模型适配
对于不同的机器人模型，可以调整动作维度：

```python
# 对于其他机器人
initialize_trajectory_visualizer(action_dim=your_action_dim)
```

## 性能影响

- **计算开销**: 约增加1-2%的训练时间（仅在保存步骤）
- **内存开销**: 几乎无额外内存占用
- **存储开销**: 每个可视化约2-3MB，根据保存频率累积

这种实现方式在保持训练脚本简洁的同时，提供了强大的轨迹可视化功能。