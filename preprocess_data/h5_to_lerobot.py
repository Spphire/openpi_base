"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""
import sys
import os
import shutil
import argparse
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import scipy.spatial.transform as st
from concurrent.futures import ThreadPoolExecutor

# ---------- helpers ----------------------------------------------------------
def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: (..., 4)  [x, y, z, w]
    return (..., 3) axis-angle (OpenCV convention, norm = angle)
    """
    rot = st.Rotation.from_quat(quat[..., [0, 1, 2, 3]])  # scipy: [x,y,z,w]
    return rot.as_rotvec()


def replace_zero_rows_with_last_nonzero(arr):
    """
    arr: (N, D) np.ndarray
    将每一行全为0的行替换为最近一次非零行（前面没有非零则保持为零）
    """
    arr = arr.copy()
    last_nonzero = None
    for i in range(len(arr)):
        if np.all(arr[i] == 0):
            if last_nonzero is not None:
                arr[i] = last_nonzero
        else:
            last_nonzero = arr[i].copy()
    return arr


def detect_pose_anomalies(pose_data: np.ndarray, position_threshold: float = 0.03,
                         tracker_name: str = "tracker") -> dict:
    """
    Detect anomalies in pose data including zero poses and large jumps.
    
    Args:
        pose_data: (T, 7) array with [x, y, z, qx, qy, qz, qw] format
        position_threshold: threshold in meters for detecting jumps (default: 1cm)
        tracker_name: name for logging purposes
    
    Returns:
        dict with anomaly information: {
            'zero_frames': list of frame indices with all-zero poses,
            'jump_frames': list of (frame_idx, distance) tuples for large jumps,
            'has_anomalies': bool indicating if any anomalies were found
        }
    """
    anomalies = {
        'zero_frames': [],
        'jump_frames': [],
        'has_anomalies': False
    }
    
    if len(pose_data) == 0:
        return anomalies
    
    # Check for all-zero poses (before any processing)
    positions = pose_data[:, :3]  # Extract position data
    
    for i, pos in enumerate(positions):
        if np.allclose(pos, 0.0, atol=1e-6):
            anomalies['zero_frames'].append(i)
    
    # Check for large position jumps between consecutive frames
    if len(positions) > 1:
        position_diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        
        for i, diff in enumerate(position_diffs):
            if diff > position_threshold:
                anomalies['jump_frames'].append((i + 1, diff))  # i+1 because we compare with next frame
    
    # Set has_anomalies flag
    anomalies['has_anomalies'] = len(anomalies['zero_frames']) > 0 or len(anomalies['jump_frames']) > 0
    
    return anomalies


def apply_mask_to_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image, setting masked regions to black (0).
    
    Args:
        img: (H, W, C) uint8 image
        mask: (H, W) uint8 mask where 255 means keep, 0 means mask out
    
    Returns:
        Masked image (H, W, C) uint8
    """
    if mask.ndim == 2:
        # Convert 2D mask to 3D for broadcasting
        mask = mask[:, :, np.newaxis]
    
    # Apply mask: keep pixels where mask > 0, set to 0 where mask == 0
    masked_img = img * (mask > 0).astype(np.uint8)
    return masked_img


def center_crop_square_resize_with_mask(img: np.ndarray, mask: np.ndarray = None, tgt: int = 224) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    mask: (H, W) uint8 mask (optional)
    1) Apply mask if provided
    2) center-crop the 640-edge to 480×480
    3) resize → tgt×tgt
    """
    # Apply mask first if provided
    if mask is not None:
        img = apply_mask_to_image(img, mask)
    
    H, W, C = img.shape
    if W == 1200:
        pass  # already square
    else:
        off = (W - 1200) // 2
        img = img[:, off: off + 1200]
    # 使用OpenCV高质量resize
    img = cv2.resize(img, (tgt, tgt), interpolation=cv2.INTER_AREA)
    return img


def center_crop_square_resize(img: np.ndarray, tgt: int = 224) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    1) center-crop the 640-edge to 480×480
    2) resize → tgt×tgt
    """
    H, W, C = img.shape
    if W == 1200:
        pass  # already square
    else:
        off = (W - 1200) // 2
        img = img[:, off: off + 1200]
    # 使用OpenCV高质量resize
    img = cv2.resize(img, (tgt, tgt), interpolation=cv2.INTER_AREA)
    return img


compression_level = 99

def extract_episode_files(hdf5_info):
    all_valid_files = []
    all_valid_lines = []
    all_valid_feats = []
    all_valid_masks = []
    all_valid_tasks = []
    for key in hdf5_info:
        hdf5_dir = Path(key)
        files = sorted(hdf5_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
        info = hdf5_info[key]
        if info["lines"] is not None:
            with open(os.path.join(os.path.dirname(__file__), info["lines"]), 'r') as f:
                lines = f.readlines()
            lines = [[int(x) for x in line.strip().split(' ')] for line in lines]
        else:
            lines = [None for _ in range(len(files))]
        invalid_id = info["invalid_id"]
        valid_index = [idx for idx, f in enumerate(files) if int(f.stem.split("_")[1]) not in invalid_id]
        valid_files = [files[idx] for idx in valid_index]
        valid_lines = [lines[idx] for idx in valid_index]
        valid_feats = [info["feat"] for _ in range(len(valid_files))]
        valid_tasks = [info.get('task', '') for _ in range(len(valid_files))]
        # Extract mask information if available
        valid_masks = [info.get("masks", {}) for _ in range(len(valid_files))]
        all_valid_files.extend(valid_files)
        all_valid_lines.extend(valid_lines)
        all_valid_feats.extend(valid_feats)
        all_valid_masks.extend(valid_masks)
        all_valid_tasks.extend(valid_tasks)
    return all_valid_files, all_valid_lines, all_valid_feats, all_valid_masks, all_valid_tasks

class Converter:
    def __init__(self, config, output_path, robot_type: str="bimanual_flexiv", fps: int=25, enable_visualization: bool = False, enable_anomaly_detection: bool = True, push_to_hub: bool = False):
        self.enable_visualization = enable_visualization
        self.enable_anomaly_detection = enable_anomaly_detection
        self.visualizer = None
        self.output_path = output_path
        self.robot_type = robot_type
        self.config = config
        self.fps = fps
        self.push_to_hub = push_to_hub
        self.hdf5_info = {}
        for k, v in self.config['data'].items():
            if k not in self.hdf5_info:
                self.hdf5_info[k] = v
        self.features = self.config['features']
        for k in self.features:
            self.features[k]['shape'] = tuple(self.features[k]['shape'])
        self.problematic_episodes = []
        self.anomaly_stats = {
            'total_episodes': 0,
            'episodes_with_anomalies': 0,
            'total_zero_frames': 0,
            'total_jump_frames': 0,
            'max_jump_distance': 0.0
        }
        # Mask storage
        self.masks = {}

    def _load_masks(self, mask_info: dict) -> dict:
        """
        Load mask images from configuration.
        
        Args:
            mask_info: Dictionary containing mask paths
            
        Returns:
            Dictionary with loaded mask arrays
        """
        masks = {}
        
        if not mask_info:
            return masks
            
        for mask_name, mask_path in mask_info.items():
            try:
                # Construct full path relative to script directory
                full_mask_path = os.path.join(os.path.dirname(__file__), mask_path)
                
                if os.path.exists(full_mask_path):
                    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks[mask_name] = mask
                        print(f"Loaded mask: {mask_name} from {full_mask_path} (shape: {mask.shape})")
                    else:
                        print(f"Warning: Could not load mask {full_mask_path}")
                else:
                    print(f"Warning: Mask file not found: {full_mask_path}")
                    
            except Exception as e:
                print(f"Error loading mask {mask_name}: {e}")
                
        return masks

    def run(self):
        dataset = LeRobotDataset.create(
            repo_id=self.output_path,
            robot_type=self.robot_type,
            fps=self.fps,
            features=self.features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        # 1. collect all episode_XXXX.hdf5
        files, lines, feats, masks_info, tasks = extract_episode_files(self.hdf5_info)
        # Load masks if available
        if masks_info and len(masks_info) > 0: self.masks = self._load_masks(masks_info[0])  # Assuming all episodes use same masks
        # 3. process each episode
        T_records = {}
        for ep_idx, (ep_path, line, feat, task) in enumerate(tqdm(zip(files, lines, feats, tasks), total=len(files), desc="Processing episodes")):
            with h5py.File(ep_path, "r") as h5:
                try:
                    episode_data = dict()
                    # Extract unified pose data or fallback to separate devices
                    left_grip_key = feat['grip_left']
                    right_grip_key = feat['grip_right']
                    grip_left = h5[left_grip_key][()] / 1000.0  # (T, 1) float32
                    grip_right = h5[right_grip_key][()] / 1000.0  # (T, 1) float32
                    # Check if we have unified pose data or separate devices
                    if 'pose_unified' in feat:
                        # New unified format: [T, 2, 7] where [:, 0, :] is left and [:, 1, :] is right
                        unified_pose_key = feat['pose_unified']
                        unified_pose = h5[unified_pose_key][()]  # (T, 2, 7) [px, py, pz, qx, qy, qz, qw]
                        if ep_idx == 0:  # Log format info for first episode
                            print(f"Using unified pose format: {unified_pose.shape} from {unified_pose_key}")
                        # Extract left and right poses from unified data
                        # Convention: tracker_roles = ['left_foot', 'right_foot'] maps to [left_arm, right_arm]
                        left_pose = unified_pose[:, 0, :]  # (T, 7) - first tracker (left_foot -> left_arm)
                        right_pose = unified_pose[:, 1, :]  # (T, 7) - second tracker (right_foot -> right_arm)
                    else:
                        # Fallback to separate devices for backward compatibility
                        left_pose_key = feat['pose_left']
                        right_pose_key = feat['pose_right']
                        left_pose = h5[left_pose_key][()]  # (T, 7) [px, py, pz, qx, qy, qz, qw]
                        right_pose = h5[right_pose_key][()]  # (T, 7) [px, py, pz, qx, qy, qz, qw]
                        if ep_idx == 0:  # Log format info for first episode
                            print(f"Using separate pose devices: {left_pose.shape} from {left_pose_key}, {right_pose.shape} from {right_pose_key}")
                    
                    # Detect anomalies in raw pose data before processing
                    episode_anomalies = {}
                    if self.enable_anomaly_detection:
                        # Detect anomalies in left pose (before any processing)
                        left_anomalies = detect_pose_anomalies(left_pose, tracker_name="left_tracker")
                        episode_anomalies['left'] = left_anomalies
                        
                        # Detect anomalies in right pose (before any processing) 
                        right_anomalies = detect_pose_anomalies(right_pose, tracker_name="right_tracker")
                        episode_anomalies['right'] = right_anomalies
                        
                        # Check if this episode has any anomalies
                        has_episode_anomalies = left_anomalies['has_anomalies'] or right_anomalies['has_anomalies']
                        
                        if has_episode_anomalies:
                            episode_info = {
                                'episode_idx': ep_idx,
                                'episode_path': str(ep_path),
                                'anomalies': episode_anomalies
                            }
                            self.problematic_episodes.append(episode_info)
                            
                            # Update statistics
                            self.anomaly_stats['total_zero_frames'] += len(left_anomalies['zero_frames']) + len(right_anomalies['zero_frames'])
                            self.anomaly_stats['total_jump_frames'] += len(left_anomalies['jump_frames']) + len(right_anomalies['jump_frames'])
                            
                            # Track maximum jump distance
                            for _, distance in left_anomalies['jump_frames'] + right_anomalies['jump_frames']:
                                self.anomaly_stats['max_jump_distance'] = max(self.anomaly_stats['max_jump_distance'], distance)

                    # Process left pose
                    left_pose = replace_zero_rows_with_last_nonzero(left_pose)
                    pos_left = left_pose[:, :3]
                    quat_left = left_pose[:, 3:]
                    axis_angle_left = quaternion_to_axis_angle(quat_left)
                    pose_mat_left = pose_to_mat(np.concatenate([pos_left, axis_angle_left], axis=-1))
                    pose_left = mat_to_pose(pose_mat_left)
                    pos_left = pose_left[:, :3]
                    axis_angle_left = pose_left[:, 3:]

                    # Process right pose
                    right_pose = replace_zero_rows_with_last_nonzero(right_pose)
                    pos_right = right_pose[:, :3]
                    quat_right = right_pose[:, 3:]
                    axis_angle_right = quaternion_to_axis_angle(quat_right)
                    pose_mat_right = pose_to_mat(np.concatenate([pos_right, axis_angle_right], axis=-1))
                    pose_right = mat_to_pose(pose_mat_right)
                    pos_right = pose_right[:, :3]
                    axis_angle_right = pose_right[:, 3:]

                    # Convert axis-angle to quaternions for visualization
                    quat_left = st.Rotation.from_rotvec(axis_angle_left).as_quat()
                    quat_right = st.Rotation.from_rotvec(axis_angle_right).as_quat()
                    # Calculate timestamps based on available data sources
                    if 'pose_unified' in feat:
                        all_ts = [left_pose.shape[0], grip_left.shape[0], grip_right.shape[0], h5[feat['img_left']].shape[0], h5[feat['img_right']].shape[0]]
                    else:
                        all_ts = [left_pose.shape[0], right_pose.shape[0], grip_left.shape[0], grip_right.shape[0], h5[feat['img_left']].shape[0], h5[feat['img_right']].shape[0]]
                    T = min(all_ts)
                    maxT = max(all_ts)
                    if np.abs(maxT - T) >= 3:
                        print(f"Warning: max/min length mismatch: {T} vs {maxT} in {ep_path}")
                    if line is not None:
                        start, end = line
                        end = min(end, T)
                    else:
                        start, end = 0, T
                    T_records[str(ep_path)] = [start, end]
                    start_pose_left = np.concatenate([pos_left[0], axis_angle_left[0]]).astype(np.float64)
                    end_pose_left = np.concatenate([pos_left[-1], axis_angle_left[-1]]).astype(np.float64)

                    start_pose_right = np.concatenate([pos_right[0], axis_angle_right[0]]).astype(np.float64)
                    end_pose_right = np.concatenate([pos_right[-1], axis_angle_right[-1]]).astype(np.float64)
                    # append

                    episode_data['robot0_eef_pos'] = pos_left[start:end]
                    episode_data['robot0_eef_rot_axis_angle'] = axis_angle_left[start:end]
                    episode_data['robot0_gripper_width'] = grip_left[start:end]
                    episode_data['robot0_demo_start_pose'] = np.tile(start_pose_left, (end - start, 1))
                    episode_data['robot0_demo_end_pose'] = np.tile(end_pose_left, (end - start, 1))
                    episode_data['robot1_eef_pos'] = pos_right[start:end]
                    episode_data['robot1_eef_rot_axis_angle'] = axis_angle_right[start:end]
                    episode_data['robot1_gripper_width'] = grip_right[start:end]
                    episode_data['robot1_demo_start_pose'] = np.tile(start_pose_right, (end - start, 1))
                    episode_data['robot1_demo_end_pose'] = np.tile(end_pose_right, (end - start, 1))

                    # process images
                    left_img_key = feat['img_left']
                    right_img_key = feat['img_right']
                    img_left = h5[left_img_key][start:end]  # for camera0_rgb
                    img_right = h5[right_img_key][start:end]  # for camera1_rgb
                    
                    # Get masks for left and right cameras
                    left_mask = self.masks.get('mask_left', None)
                    right_mask = self.masks.get('mask_right', None)
                    
                    # Process images in parallel with masks
                    with ThreadPoolExecutor() as executor:
                        # Create partial functions with masks
                        process_left = lambda img: center_crop_square_resize_with_mask(img, left_mask)
                        process_right = lambda img: center_crop_square_resize_with_mask(img, right_mask)
                        img_left = list(executor.map(process_left, img_left))
                        img_right = list(executor.map(process_right, img_right))
                    img_left = np.stack(img_left)
                    img_right = np.stack(img_right)
                    num_frames = img_left.shape[0] - 1
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
                    episode_data['task'] = task
                    for step in range(num_frames):
                        frame_dict = {feat: episode_data[feat][step] for feat in self.features}
                        frame_dict['task'] = task
                        dataset.add_frame(frame_dict)
                    dataset.save_episode()
                except Exception as e:
                    print(f"Error processing {ep_path}: {e}")
                # Update episode counter
                self.anomaly_stats['total_episodes'] += 1

        
        # Output anomaly detection results
        if self.enable_anomaly_detection:
            self._output_anomaly_results()

        if self.push_to_hub:
            dataset.push_to_hub(
                tags=["flexiv", self.robot_type],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )

    
    def _output_anomaly_results(self) -> None:
        """Output comprehensive anomaly detection results."""
        try:
            # Update final statistics
            self.anomaly_stats['episodes_with_anomalies'] = len(self.problematic_episodes)
            
            print("=" * 80)
            print("VIVE TRACKER POSE ANOMALY DETECTION RESULTS")
            print("=" * 80)
            
            # Overall statistics
            print(f"Total episodes processed: {self.anomaly_stats['total_episodes']}")
            print(f"Episodes with anomalies: {self.anomaly_stats['episodes_with_anomalies']}")
            print(f"Anomaly rate: {self.anomaly_stats['episodes_with_anomalies']/max(1, self.anomaly_stats['total_episodes'])*100:.1f}%")
            print(f"Total zero frames detected: {self.anomaly_stats['total_zero_frames']}")
            print(f"Total jump frames detected: {self.anomaly_stats['total_jump_frames']}")
            print(f"Maximum jump distance: {self.anomaly_stats['max_jump_distance']:.4f}m")
            
            if self.problematic_episodes:
                print("\nPROBLEMATIC EPISODES:")
                print("-" * 60)
                
                problematic_indices = []
                for episode_info in self.problematic_episodes:
                    ep_idx = episode_info['episode_idx']
                    ep_path = episode_info['episode_path']
                    anomalies = episode_info['anomalies']
                    
                    problematic_indices.append(ep_idx)
                    
                    print(f"Episode {ep_idx}: {Path(ep_path).name}")
                    
                    # Left tracker anomalies
                    left_anom = anomalies['left']
                    if left_anom['has_anomalies']:
                        print(f"  Left tracker:")
                        if left_anom['zero_frames']:
                            print(f"    - Zero frames: {len(left_anom['zero_frames'])} (indices: {left_anom['zero_frames'][:10]}{'...' if len(left_anom['zero_frames']) > 10 else ''})")
                        if left_anom['jump_frames']:
                            print(f"    - Jump frames: {len(left_anom['jump_frames'])}")
                            for frame_idx, distance in left_anom['jump_frames'][:5]:  # Show first 5 jumps
                                print(f"      Frame {frame_idx}: {distance:.4f}m jump")
                            if len(left_anom['jump_frames']) > 5:
                                print(f"      ... and {len(left_anom['jump_frames']) - 5} more jumps")
                    
                    # Right tracker anomalies
                    right_anom = anomalies['right']
                    if right_anom['has_anomalies']:
                        print(f"  Right tracker:")
                        if right_anom['zero_frames']:
                            print(f"    - Zero frames: {len(right_anom['zero_frames'])} (indices: {right_anom['zero_frames'][:10]}{'...' if len(right_anom['zero_frames']) > 10 else ''})")
                        if right_anom['jump_frames']:
                            print(f"    - Jump frames: {len(right_anom['jump_frames'])}")
                            for frame_idx, distance in right_anom['jump_frames'][:5]:  # Show first 5 jumps
                                print(f"      Frame {frame_idx}: {distance:.4f}m jump")
                            if len(right_anom['jump_frames']) > 5:
                                print(f"      ... and {len(right_anom['jump_frames']) - 5} more jumps")
                    
                    print("")
                
                print("SUMMARY OF PROBLEMATIC EPISODE INDICES:")
                print(f"Episodes with anomalies: {sorted(problematic_indices)}")
                
            else:
                print("\n✅ No pose anomalies detected in any episodes!")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"Error outputting anomaly results: {e}")




def main(args):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    import yaml
    with open(args.config_path, 'r') as f:
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
