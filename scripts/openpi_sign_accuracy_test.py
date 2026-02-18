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

def load_openpi_policy(config_name: str, checkpoint_path: str):
    """Load OpenPI policy from checkpoint."""
    config = _config.get_config(config_name)
    print(f"Loading policy from {checkpoint_path}")
    policy = _policy_config.create_trained_policy(config, checkpoint_path)
    return policy, config

def load_openpi_dataset(dataset_path: str):
    """Load OpenPI dataset in LeRobot-compatible format."""
    dataset_path = pathlib.Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading OpenPI dataset from: {dataset_path}")

    # Load dataset using LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_path)

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

def load_dataset_from_config(config):
    """Load dataset path from configuration using repo_id."""
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    if not repo_id or repo_id == "fake":
        raise ValueError("No valid repo_id found in config. Cannot load dataset.")
    print(f"Dataset repo_id: {repo_id}")
    return repo_id

def main():
    parser = argparse.ArgumentParser(description="OpenPI Sign Accuracy Evaluation")
    parser.add_argument("--config", default='pi05_iPhoneVRSingle_q3_mouse', help="Configuration name for the policy")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-csv", default="openpi_sign_accuracy.csv", help="Path to save results")
    args = parser.parse_args()

    # Load configuration and policy
    config, policy = load_openpi_config_and_policy(args.config, args.checkpoint)

    # Load dataset path from configuration
    dataset_path = load_dataset_from_config(config)

    # Load dataset
    episodes = load_openpi_dataset(dataset_path)

    results = []
    for ep_idx, episode in episodes.items():
        gripper_width = episode["gripper_width"]
        window = find_closing_window(gripper_width)
        if window is None:
            print(f"Episode {ep_idx}: No closing window found, skipping.")
            continue

        # Prepare observation
        obs = {
            "observation/state": episode["state"],
            "observation/eye_image": episode["eye_image"],
            "observation/left_wrist_image": episode["left_wrist_image"],
            "observation/right_wrist_image": episode["right_wrist_image"],
        }

        # Run inference
        result = policy.infer(obs)
        pred_actions = result["actions"]
        gt_actions = episode["actions"]

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