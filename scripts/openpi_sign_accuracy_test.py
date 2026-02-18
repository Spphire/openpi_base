import os
import pathlib
import argparse
import csv
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.policies import single_iphone_vr_flexiv_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_openpi_policy(config_name: str, checkpoint_path: str):
    """Load OpenPI policy from checkpoint."""
    config = _config.get_config(config_name)
    print(f"Loading policy from {checkpoint_path}")
    policy = _policy_config.create_trained_policy(config, checkpoint_path)
    return policy, config

def load_openpi_dataset_from_config(config, episode_id: int = 0):
    """Load OpenPI dataset using repo_id and delta_timestamps."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # Get repo_id from config
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    if not repo_id or repo_id == "fake":
        raise ValueError("No valid repo_id found in config. Cannot load dataset.")

    # Load dataset metadata
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    action_horizon = config.model.action_horizon

    # Create delta_timestamps based on fps and action_sequence_keys
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }

    print(f"Dataset FPS: {dataset_meta.fps}, Action horizon: {action_horizon}")
    print(f"Delta timestamps: {delta_timestamps}")

    # Load dataset with specific episode
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        episodes=[episode_id]
    )

    episodes = {}
    for episode_id in range(len(dataset)):
        episode_data = dataset[episode_id]
        episodes[episode_id] = {
            "state": episode_data["state"],
            "actions": episode_data["actions"],
            "gripper_width": episode_data["left_robot_gripper_width"],
            "eye_image": episode_data.get("left_eye_img"),
            "left_wrist_image": episode_data.get("left_wrist_img"),
            "right_wrist_image": episode_data.get("right_wrist_img"),
        }
    return episodes

def find_closing_window(gripper_width: np.ndarray, close_eps: float = 1e-4, min_len: int = 5) -> Optional[Tuple[int, int]]:
    """Find the closing window based on gripper width."""
    w = np.asarray(gripper_width).squeeze()
    dw = np.diff(w)
    dec = dw < -close_eps

    segments = []
    start = None
    for i, flag in enumerate(dec):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(dec) - 1))

    for s, e in segments:
        if e - s + 1 >= min_len:
            return s, e
    return None

def compute_sign_accuracy(pred_actions: np.ndarray, gt_actions: np.ndarray, window: Tuple[int, int]) -> float:
    """Compute sign accuracy within the closing window."""
    start, end = window
    pred_x = pred_actions[start:end + 1, 0]
    gt_x = gt_actions[start:end + 1, 0]

    valid = (np.abs(gt_x) > 1e-6) & (np.abs(pred_x) > 1e-6)
    if valid.sum() == 0:
        return float("nan")

    correct = np.sign(pred_x[valid]) == np.sign(gt_x[valid])
    return correct.mean()

def load_openpi_config_and_policy(config_name: str, checkpoint_path: str):
    """Load OpenPI configuration and policy."""
    config = _config.get_config(config_name)
    print(f"Loading policy from {checkpoint_path}")
    policy = _policy_config.create_trained_policy(config, checkpoint_path)
    return config, policy

def get_repoid_from_config(config):
    """Load dataset path from configuration using repo_id and construct full path."""
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    if not repo_id or repo_id == "fake":
        raise ValueError("No valid repo_id found in config. Cannot load dataset.")
    return data_config, repo_id

def main():
    parser = argparse.ArgumentParser(description="OpenPI Sign Accuracy Evaluation")
    parser.add_argument("--config", default='pi05_iPhoneVRSingle_q3_mouse', help="Configuration name for the policy")
    parser.add_argument("--checkpoint", default='checkpoints/pi05_iPhoneVRSingle_q3_mouse/q3mouse_normal/29999', help="Path to model checkpoint")
    parser.add_argument("--output-csv", default="openpi_sign_accuracy.csv", help="Path to save results")
    args = parser.parse_args()

    # Load configuration and policy
    config, policy = load_openpi_config_and_policy(args.config, args.checkpoint)

    # Load dataset path from configuration
    data_config, repo_id = get_repoid_from_config(config)
            
    # Load dataset with proper delta_timestamps for action chunks
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    action_horizon = config.model.action_horizon
    
    # Create delta_timestamps based on fps and action_sequence_keys
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] 
        for key in data_config.action_sequence_keys
    }

    results = []
    for ep_idx in range(295):
        dataset = LeRobotDataset(
            repo_id, 
            delta_timestamps=delta_timestamps,
            episodes=[ep_idx]
        )
        gripper_width = [raw_data['state'][-1] for raw_data in dataset]
        window = find_closing_window(gripper_width)
        if window is None:
            print(f"Episode {ep_idx}: No closing window found, skipping.")
            continue

        raw_data = dataset[window[-1]]

        repacked_data = {
            "observation/left_wrist_image": raw_data["left_wrist_img"],
            "observation/right_wrist_image": raw_data["right_wrist_img"], 
            "observation/eye_image": raw_data["left_eye_img"],
            "observation/state": raw_data["state"],
            "actions": raw_data["actions"],
            "prompt": raw_data.get("task", "pick up object")
        }

        flexiv_inputs = single_iphone_vr_flexiv_policy.SingleiPhoneVRFlexivInputs(model_type=config.model.model_type)
        # Apply transform (this handles the format conversion)
        transformed_data = flexiv_inputs(repacked_data)
        
        # Use previous sample's state if available, otherwise use zero state
        current_state = transformed_data["state"]
        sample_prev_state = np.zeros_like(current_state)
        
        # Extract the data for our sample format
        sample = {
            'sample_id': sample_idx,
            'sample_index_in_batch': i,
            'episode_source_id': episode_id,
            'state': current_state,
            'prev_state': sample_prev_state,
            "eye_image": transformed_data["image"]["base_0_rgb"],
            'left_wrist_image': transformed_data["image"]["left_wrist_0_rgb"],
            'right_wrist_image': transformed_data["image"]["right_wrist_0_rgb"],
            'gt_actions': transformed_data["actions"],
            'prompt': transformed_data.get("prompt", "pick up object"),
            'raw_data': raw_data
        }

        # Prepare observation
        obs_input = {
            "observation/state": sample['state'],
            "observation/eye_image": sample['eye_image'],
            "observation/left_wrist_image": sample['left_wrist_image'],
            "observation/right_wrist_image": sample['right_wrist_image'],
            "prompt": sample['prompt']
        }

        # Run inference
        result = policy.infer(obs_input)
        pred_actions = result["actions"]
        gt_actions = raw_data["actions"]

        # Compute accuracy
        acc = compute_sign_accuracy(pred_actions, gt_actions, window)
        print(f"Episode {ep_idx}: Sign accuracy = {acc:.4f}")
        results.append((ep_idx, acc))

    # Save results
    output_path = pathlib.Path(args.output_csv).expanduser().resolve()
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "sign_accuracy"])
        writer.writerows(results)
    print(f"Results saved to {output_path}")

    # Compute and print average accuracy
    valid_accuracies = [acc for _, acc in results if not np.isnan(acc)]
    if valid_accuracies:
        avg_accuracy = sum(valid_accuracies) / len(valid_accuracies)
        print(f"Average Sign Accuracy: {avg_accuracy:.4f}")
    else:
        print("No valid accuracies to compute average.")

if __name__ == "__main__":
    main()