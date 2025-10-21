"""
1. action=state, no prev state
2. use tcp gripper coordinate system
3. convert the rotation representation to rpy (euler)
"""

DIFFUSION_POLICY_DIR = "/home/wz/Data-Scaling-Laws/"
import os
import sys
sys.path.append(DIFFUSION_POLICY_DIR)  # add parent dir to
import zarr
import argparse
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pos_rot_to_mat, mat_to_pos_rot, pose_to_pos_rot, pos_rot_to_pose, pose_to_mat, mat_to_pose
import numpy as np
register_codecs()
from pathlib import Path
import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation


class Converter:
    def __init__(
        self,
        config,
        output_path,
        robot_type: str = "bimanual_flexiv",
        fps: int = 25,
        push_to_hub: bool = False,
    ):
        self.visualizer = None
        self.output_path = output_path
        self.robot_type = robot_type
        self.config = config
        self.fps = fps
        self.push_to_hub = push_to_hub
        self.zarr_paths = self.config["data"]
        self.tasks = self.config['task']
        assert len(self.zarr_paths)==len(self.tasks)
        self.features = self.config["features"]
        for k in self.features:
            self.features[k]["shape"] = tuple(self.features[k]["shape"])
        self.tcp_to_vive_transform = np.array([ 
            [  0.0000  , -0.9397 ,  0.3420,  -0.0317],
            [1.0000 ,  0.0000  , 0.0000  , 0.0000],
            [0.0000  , 0.3420 ,  0.9397 , -0.272],
            [0.0000  , 0.0000  , 0.0000  , 1.0000]
        ]).astype(np.float32)
        self.vive_to_tcp_transform = np.linalg.inv(self.tcp_to_vive_transform)

    def read_dataset(self, dataset_path):
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )
        return replay_buffer

    def run(self):
        dataset = LeRobotDataset.create(
            repo_id=self.output_path,
            robot_type=self.robot_type,
            fps=self.fps,
            features=self.features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        ep_idx = 0
        for zarr_path, task in zip(self.zarr_paths, self.tasks):
            replay_buffer = self.read_dataset(zarr_path)
            episode_ends = replay_buffer.episode_ends[:]
            episode_start = [0] + list(episode_ends)
            num_episodes_zarr = len(episode_ends)
            for episode_id in range(num_episodes_zarr):
                try:
                    start_idx = episode_start[episode_id]
                    end_idx = episode_ends[episode_id]
                    episode_data = {}
                    all_keys = ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width','robot1_eef_pos', 'robot1_eef_rot_axis_angle','robot1_gripper_width','camera0_rgb','camera1_rgb']
                    episode_data = {
                        k: getattr(replay_buffer.data, k)[start_idx:end_idx] for k in all_keys
                    }
                    # construct state 
                    # convert pose to mat
                    left_pose = np.concatenate([episode_data["robot0_eef_pos"], episode_data["robot0_eef_rot_axis_angle"]], axis=1)
                    right_pose = np.concatenate([episode_data["robot1_eef_pos"], episode_data["robot1_eef_rot_axis_angle"]], axis=1)
                    left_gripper_width = episode_data["robot0_gripper_width"]
                    right_gripper_width = episode_data["robot1_gripper_width"]
                    left_pose_mat = pose_to_mat(left_pose)
                    right_pose_mat = pose_to_mat(right_pose)
                    # transform to tcp coordinate system
                    if getattr(self, "vive_to_tcp_transform", None) is not None:
                        left_pose_mat =  left_pose_mat @ self.vive_to_tcp_transform
                        right_pose_mat = right_pose_mat @ self.vive_to_tcp_transform
                    # convert to xyzrpy representation
                    left_pos = left_pose_mat[:, :3, 3]
                    right_pos = right_pose_mat[:, :3, 3]
                    left_rpy = Rotation.from_matrix(left_pose_mat[:, :3, :3]).as_euler('xyz', degrees=False)
                    right_rpy = Rotation.from_matrix(right_pose_mat[:, :3, :3]).as_euler('xyz', degrees=False)
                    left_state = np.concatenate([left_pos, left_rpy, left_gripper_width], axis=1)
                    right_state = np.concatenate([right_pos, right_rpy, right_gripper_width], axis=1)
                    state = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
                    # construct actions (use state as the action)
                    episode_data["state"] = state
                    episode_data["actions"] = state
                    episode_data["left_wrist_image"] = episode_data['camera0_rgb']
                    episode_data["right_wrist_image"] = episode_data['camera1_rgb']
                    episode_data["task"] = task
                    num_frames = state.shape[0]
                    for step in range(num_frames):
                        frame_dict = {feat: episode_data[feat][step] for feat in self.features}
                        frame_dict["task"] = task
                        dataset.add_frame(frame_dict)
                    dataset.save_episode()
                    ep_idx += 1
                except Exception as e:
                    print(f"Error processing {episode_id} of {zarr_path}: {e}")

        if self.push_to_hub:
            dataset.push_to_hub(
                tags=["flexiv", self.robot_type],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )

def main(args):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists(): shutil.rmtree(output_path)
    import yaml
    with open(args.config_path) as f:
        config_data = yaml.safe_load(f)
    Converter(config_data, output_path, fps=args.fps, robot_type=args.robot_type, push_to_hub=args.push_to_hub).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to lerobot dataset")
    parser.add_argument(
        "config_path",
        type=Path,
        nargs="?",
        default=Path("preprocess_data/configs/debug.yaml"),
        help="Path to the config YAML file (default: config0925.yaml)",
    )
    parser.add_argument(
        "repo_name",
        type=Path,
        nargs="?",
        default=Path("flexiv/pick"),
        help="the name of the dataset",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="the fps of data",
        default=25,
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        help="the type of robot",
        default="bimanual_flexiv",
    )
    parser.add_argument(
        "--push_to_hub",
        help="the type of robot",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
