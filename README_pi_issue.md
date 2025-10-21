# 输入本体感知
pi0的输入本体感知：delta state? 跟预训练不匹配
pi0的输入本体感知：transform和delta是否有冲突？
丢弃本体感知

# 输入文本
替换成
put xxx in xxx
place xxx in xxx

# 旋转表示
旋转表示不能用rotation axis angle，要转成rpy

scipy默认的

# 结论
1. 不用本体感知
2. zarr里用tcp gripper坐标系 (rerun可视化看一下)
3. state的旋转转成rpy （在zarr-to-lerobot里写）
4. action使用减去第一帧state
5. 文本换place\put

# 实现
1. plot_3d_trajectories(left_pos, left_pose[:, :3]) 
   plt.savefig('tmp_traj.png')
   红色为tcp坐标系，范围更大，没问题