"""
1. action=state, no prev state
2. do not use tcp gripper coordinate system, use iphone coordinate system instead
3. convert the rotation representation to rpy (euler)
"""

from sys import settrace
import zarr
import argparse
import numpy as np
from pathlib import Path
import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v)
    v_mag = np.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v

def ortho6d_to_matrix(ortho6d: np.ndarray) -> np.ndarray:
    x_raw = ortho6d[0:3]
    y_raw = ortho6d[3:6]
    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    z = z[:, np.newaxis]
    matrix = np.concatenate((x, y, z), axis=1)
    return matrix

def matrix_to_rotvec(rot):
    return Rotation.from_matrix(rot).as_rotvec()

def resize(
    images,
    target_h=224,
    target_w=224,
    mode="pad",          # "pad" or "crop"
    pad_value=0,
    interpolation=cv2.INTER_LINEAR,
):
    """
    Resize images with aspect ratio preserved, then pad or crop.

    Args:
        images (np.ndarray): (b, h, w, 3)
        target_h (int)
        target_w (int)
        mode (str): "pad" or "crop"
        pad_value (int or tuple): padding value
        interpolation: cv2 interpolation

    Returns:
        np.ndarray: (b, target_h, target_w, 3)
    """
    assert mode in ["pad", "crop"]
    b, h, w, c = images.shape
    assert c == 3

    if h==target_h and w==target_w:
        return images

    resized_images = []

    for img in images:
        scale_h = target_h / h
        scale_w = target_w / w

        if mode == "pad":
            scale = min(scale_h, scale_w)
        else:  # crop
            scale = max(scale_h, scale_w)

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        img_resized = cv2.resize(
            img, (new_w, new_h), interpolation=interpolation
        )

        if mode == "pad":
            pad_h = target_h - new_h
            pad_w = target_w - new_w

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            img_out = cv2.copyMakeBorder(
                img_resized,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_value,
            )

        else:  # crop
            start_y = (new_h - target_h) // 2
            start_x = (new_w - target_w) // 2

            img_out = img_resized[
                start_y:start_y + target_h,
                start_x:start_x + target_w
            ]

        resized_images.append(img_out)

    return np.stack(resized_images, axis=0)

def convert_episode_to_lerobot_format(
    episode_data: dict,
    robot_type: str,
    task: str = "",
) -> dict:
    """
    Convert raw episode data to LeRobot-compatible format.
    
    Args:
        episode_data: Dict containing raw episode data with keys like 
                      'left_robot_tcp_pose', 'left_robot_gripper_width', 
                      'left_wrist_img', etc.
        robot_type: Either "single_iphone_flexiv" or "bimanual_iphone_flexiv"
        task: Task description string
    
    Returns:
        Dict with converted data including 'state', 'actions', and image keys
    """
    result = {}
    
    # Process left arm data
    left_pose_9d = episode_data["left_robot_tcp_pose"]
    left_pose = np.zeros((left_pose_9d.shape[0], 6), dtype=np.float32)
    for i in range(left_pose.shape[0]):
        left_pose[i][:3] = left_pose_9d[i][:3]
        left_pose[i][3:] = matrix_to_rotvec(ortho6d_to_matrix(left_pose_9d[i][3:9]))
    left_gripper_width = episode_data["left_robot_gripper_width"]
    left_state = np.concatenate([left_pose, left_gripper_width], axis=1)
    
    if robot_type == "single_iphone_vr_flexiv":
        state = left_state
    else:
        # Process right arm data for bimanual
        right_pose_9d = episode_data["right_robot_tcp_pose"]
        right_pose = np.zeros((right_pose_9d.shape[0], 6), dtype=np.float32)
        for i in range(right_pose.shape[0]):
            right_pose[i][:3] = right_pose_9d[i][:3]
            right_pose[i][3:] = matrix_to_rotvec(ortho6d_to_matrix(right_pose_9d[i][3:9]))
        right_gripper_width = episode_data["right_robot_gripper_width"]
        right_state = np.concatenate([right_pose, right_gripper_width], axis=1)
        state = np.concatenate([left_state, right_state], axis=1)
    
    result["state"] = state.astype(np.float32)
    result["actions"] = state.astype(np.float32).copy()

    # Copy image data
    if "left_wrist_img" in episode_data:
        result["left_wrist_img"] = resize(episode_data["left_wrist_img"], mode="crop")
    if "right_wrist_img" in episode_data:
        result["right_wrist_img"] = resize(episode_data["right_wrist_img"], mode="crop")
    if "left_eye_img" in episode_data:
        result["left_eye_img"] = resize(episode_data["left_eye_img"], mode="pad")
    if "right_eye_img" in episode_data:
        result["right_eye_img"] = resize(episode_data["right_eye_img"], mode="pad")
    
    result["task"] = task
    
    return result


class ReplayBuffer:
    """
    Replay buffer for iphone zarr dataset
    """
    def __init__(self, meta, data):
        self.meta = meta
        self.data = data
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']


class Converter:
    def __init__(
        self,
        config,
        output_path,
        robot_type: str = "bimanual_iphone_vr_flexiv",
        fps: int = 10,
        push_to_hub: bool = False,
    ):
        self.visualizer = None
        self.output_path = output_path
        assert robot_type in ["single_iphone_vr_flexiv", "bimanual_iphone_vr_flexiv"], f"Unsupported robot type: {robot_type}"
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

    def read_dataset(self, dataset_path):
        src_root = zarr.group(dataset_path)
        meta = dict()
        for key, value in src_root['meta'].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]

        keys = src_root['data'].keys()
        data = dict()
        for key in keys:
            arr = src_root['data'][key]
            data[key] = arr[:]

        replay_buffer = ReplayBuffer(meta, data)
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
            for episode_id in tqdm(range(num_episodes_zarr), desc="Processing episodes"):
                try:
                    start_idx = episode_start[episode_id]
                    end_idx = episode_ends[episode_id]
                    all_keys = ['left_wrist_img', 'left_robot_tcp_pose', 'left_robot_gripper_width']
                    all_keys.extend(['left_eye_img', 'right_eye_img'])
                    if self.robot_type == "bimanual_iphone_vr_flexiv":
                        all_keys.extend(['right_wrist_img', 'right_robot_tcp_pose', 'right_robot_gripper_width'])
                    raw_episode_data = {
                        k: replay_buffer.data[k][start_idx:end_idx] for k in all_keys
                    }
                    # Convert to LeRobot format using shared function
                    episode_data = convert_episode_to_lerobot_format(
                        raw_episode_data, self.robot_type, task
                    )
                    # save all frames
                    num_frames = episode_data["state"].shape[0]
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
        # default=Path("preprocess_data/configs/q3_shop_bagging_0202_100_zarr.yaml"),
        default=Path("preprocess_data/configs/q3_mouse_zarr.yaml"),
        help="Path to the config YAML file (default: config0925.yaml)",
    )
    parser.add_argument(
        "repo_name",
        type=Path,
        nargs="?",
        default=Path("q3_mouse"),
        help="the name of the dataset",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="the fps of data",
        default=30,
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        help="the type of robot",
        default="single_iphone_vr_flexiv",
    )
    parser.add_argument(
        "--push_to_hub",
        help="the type of robot",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
