#!/usr/bin/env python3
"""
Visualization script for bimanual flexiv policy inference.
Supports both dummy data and real dataset comparison.
"""

import argparse
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Any
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PIL import Image

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import bimanual_flexiv_policy
from openpi.shared import download


class FlexivTrajectoryVisualizer:
    """Visualizes bimanual flexiv policy inference trajectories."""
    
    def __init__(self, config_name: str, checkpoint_path: str, output_dir: str = "flexiv_visualization_output"):
        """
        Initialize the trajectory visualizer.
        
        Args:
            config_name: Name of the configuration to use (e.g., "pi05_flexiv_pick")
            checkpoint_path: Path to the model checkpoint
            output_dir: Directory to save visualization outputs
        """
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load configuration and policy
        self.config = _config.get_config(config_name)
        print(f"Loading policy from {checkpoint_path}")
        self.policy = _policy_config.create_trained_policy(self.config, checkpoint_path)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_real_dataset_episodes(self, episode_id: int = 0, num_samples: int = 5) -> List[Dict]:
        """Load samples from a specific episode directly from LeRobot dataset."""
        print(f"Loading episode {episode_id} with {num_samples} samples from LeRobot...")
        
        try:
            # Import LeRobot dataset
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            # Get the repo_id from config
            data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
            repo_id = data_config.repo_id
            
            if repo_id is None or repo_id == "fake":
                raise ValueError("No valid repo_id found in config. Cannot load real dataset.")
            
            print(f"Loading from repo: {repo_id}")
            
            # Load dataset with proper delta_timestamps for action chunks
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
            
            dataset_meta = LeRobotDatasetMetadata(repo_id)
            action_horizon = self.config.model.action_horizon
            
            # Create delta_timestamps based on fps and action_sequence_keys
            delta_timestamps = {
                key: [t / dataset_meta.fps for t in range(action_horizon)] 
                for key in data_config.action_sequence_keys
            }
            
            print(f"Dataset FPS: {dataset_meta.fps}, Action horizon: {action_horizon}")
            print(f"Delta timestamps: {delta_timestamps}")
            
            # Create dataset with specific episode
            dataset = LeRobotDataset(
                repo_id, 
                delta_timestamps=delta_timestamps,
                episodes=[episode_id]
            )
            
            # Get BimanualFlexivInputs and BimanualFlexivOutputs transforms
            flexiv_inputs = bimanual_flexiv_policy.BimanualFlexivInputs(model_type=self.config.model.model_type)
            flexiv_outputs = bimanual_flexiv_policy.BimanualFlexivOutputs()
            
            episodes = []
            dataset_len = len(dataset)
            
            # Sample multiple samples from this episode
            num_samples_to_take = min(num_samples, dataset_len)
            if dataset_len > num_samples_to_take:
                # Evenly distribute samples across the episode
                step = dataset_len // num_samples_to_take
                sample_indices = [i * step for i in range(num_samples_to_take)]
            else:
                sample_indices = list(range(dataset_len))
            
            print(f"Sampling {len(sample_indices)} samples from episode {episode_id}: {sample_indices}")
            
            for i, sample_idx in enumerate(sample_indices):
                try:
                    # Get raw data from LeRobot dataset
                    raw_data = dataset[sample_idx]
                    
                    # Apply repack transform to match expected format
                    # Based on LeRobotBimanualFlexivDataConfig repack_transform
                    repacked_data = {
                        "observation/left_wrist_image": raw_data["left_wrist_image"],
                        "observation/right_wrist_image": raw_data["right_wrist_image"], 
                        "observation/state": raw_data["state"],
                        "actions": raw_data["actions"],
                        "prompt": raw_data.get("task", "pick up object")
                    }
                    
                    # Apply BimanualFlexivInputs transform (this handles the format conversion)
                    transformed_data = flexiv_inputs(repacked_data)
                    
                    # Extract the data for our episode format
                    episode = {
                        'episode_id': i,
                        'sample_id': sample_idx,
                        'episode_source_id': episode_id,
                        'state': transformed_data["state"],
                        'left_wrist_image': transformed_data["image"]["left_wrist_0_rgb"],
                        'right_wrist_image': transformed_data["image"]["right_wrist_0_rgb"],
                        'gt_actions': transformed_data["actions"],
                        'prompt': transformed_data.get("prompt", "pick up object"),
                        'raw_data': raw_data
                    }
                    
                    episodes.append(episode)
                    print(f"Loaded sample {i+1}/{len(sample_indices)} (dataset index: {sample_idx})")
                    
                except Exception as e:
                    print(f"Failed to load sample {sample_idx}: {e}")
                    continue
                    
            print(f"Successfully loaded {len(episodes)} samples from episode {episode_id}")
            return episodes
            
        except Exception as e:
            print(f"Failed to load real dataset: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Could not load real dataset: {e}") from e
    
    def create_dummy_episodes(self, num_episodes: int = 5) -> List[Dict]:
        """Create dummy episodes for testing when real dataset is not available."""
        episodes = []
        
        for i in range(num_episodes):
            # Create base example
            example = bimanual_flexiv_policy.make_bimanual_flexiv_example()
            
            episode = {
                'episode_id': i,
                'state': example["observation/state"],
                'left_wrist_image': example["observation/left_wrist_image"],
                'right_wrist_image': example["observation/right_wrist_image"],
                'gt_actions': np.random.randn(self.config.model.action_horizon, 14) * 0.1,
                'prompt': example["prompt"]
            }
            episodes.append(episode)
            
        return episodes
    
    def run_inference_on_episode(self, episode: Dict) -> Dict:
        """Run inference on a single episode."""
        # Create input for inference
        obs_input = {
            "observation/state": episode['state'],
            "observation/left_wrist_image": episode['left_wrist_image'],
            "observation/right_wrist_image": episode['right_wrist_image'],
            "prompt": episode['prompt']
        }
        
        print(f"Running inference on episode {episode['episode_id']}...")
        
        try:
            result = self.policy.infer(obs_input)
            predicted_actions = result["actions"]
            inference_time = result.get('policy_timing', {}).get('infer_ms', 0)
            
            return {
                'predicted_actions': predicted_actions,
                'inference_time': inference_time,
                'success': True
            }
            
        except Exception as e:
            print(f"Inference failed on episode {episode['episode_id']}: {e}")
            import traceback
            traceback.print_exc()
            
            # Use zeros as fallback
            fallback_actions = np.zeros((self.config.model.action_horizon, 14))
            return {
                'predicted_actions': fallback_actions,
                'inference_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def plot_trajectory_comparison(self, episode: Dict, inference_result: Dict, 
                                 save_path: Optional[str] = None) -> None:
        """Plot comparison between ground truth and predicted trajectories."""
        gt_actions = episode['gt_actions']
        pred_actions = inference_result['predicted_actions']
        episode_id = episode['episode_id']
        
        # Handle different shapes of gt_actions and pred_actions
        if gt_actions.ndim == 3:  # (batch, horizon, action_dim)
            gt_actions = gt_actions.reshape(-1, gt_actions.shape[-1])
        elif gt_actions.ndim == 2:  # (horizon, action_dim) - single episode
            pass
        
        # Determine the actual action dimension (flexiv uses 14 DOF)
        actual_action_dim = 14
        
        # Truncate gt_actions to actual dimension if it's padded
        if gt_actions.shape[-1] > actual_action_dim:
            gt_actions = gt_actions[:, :actual_action_dim]
        
        # Truncate pred_actions to actual dimension if it's padded  
        if pred_actions.shape[-1] > actual_action_dim:
            pred_actions = pred_actions[:, :actual_action_dim]
        
        if pred_actions.ndim == 2:  # (action_horizon, action_dim) - single prediction
            # Compare the predicted action sequence with the first part of gt
            min_len = min(len(gt_actions), len(pred_actions))
            gt_actions_to_compare = gt_actions[:min_len]
            pred_actions_to_compare = pred_actions[:min_len]
        else:
            # If we have multiple predictions, use the first one
            pred_actions_to_compare = pred_actions
            min_len = min(len(gt_actions), len(pred_actions))
            gt_actions_to_compare = gt_actions[:min_len]
            pred_actions_to_compare = pred_actions[:min_len]
        
        # Ensure same action dimensions
        min_action_dim = min(gt_actions_to_compare.shape[-1], pred_actions_to_compare.shape[-1])
        gt_actions_to_compare = gt_actions_to_compare[:, :min_action_dim]
        pred_actions_to_compare = pred_actions_to_compare[:, :min_action_dim]
        
        # Convert to numpy arrays if they're tensors
        if hasattr(gt_actions_to_compare, 'detach'):  # PyTorch tensor
            gt_actions_to_compare = gt_actions_to_compare.detach().cpu().numpy()
        elif hasattr(gt_actions_to_compare, 'device'):  # JAX array
            gt_actions_to_compare = np.array(gt_actions_to_compare)
        else:
            gt_actions_to_compare = np.array(gt_actions_to_compare)
            
        if hasattr(pred_actions_to_compare, 'detach'):  # PyTorch tensor
            pred_actions_to_compare = pred_actions_to_compare.detach().cpu().numpy()
        elif hasattr(pred_actions_to_compare, 'device'):  # JAX array
            pred_actions_to_compare = np.array(pred_actions_to_compare)
        else:
            pred_actions_to_compare = np.array(pred_actions_to_compare)
        
        # Create subplots for each action dimension
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(min(min_action_dim, len(axes))):
            ax = axes[i]
            
            # Plot ground truth and predictions
            ax.plot(gt_actions_to_compare[:, i], label='Ground Truth', linewidth=2, alpha=0.8, color='blue')
            ax.plot(pred_actions_to_compare[:, i], label='Predicted', linewidth=2, alpha=0.8, 
                   color='red', linestyle='--')
            
            ax.set_ylabel(f'Joint {i}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(14, len(axes)):
            axes[i].set_visible(False)
            
        axes[-4].set_xlabel('Time Step')
        plt.suptitle(f'Trajectory Comparison - Episode {episode_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory comparison to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_action_horizon_comparison(self, episode: Dict, inference_result: Dict,
                                     timestep: int = 0, save_path: Optional[str] = None) -> None:
        """Plot the full action horizon prediction vs ground truth at a specific timestep."""
        gt_actions = episode['gt_actions']
        pred_actions = inference_result['predicted_actions']
        episode_id = episode['episode_id']
        
        if timestep >= len(pred_actions):
            print(f"Timestep {timestep} not available")
            return
            
        # Handle action dimension truncation
        actual_action_dim = 14
        
        # Truncate gt_actions to actual dimension if it's padded
        if gt_actions.shape[-1] > actual_action_dim:
            gt_actions = gt_actions[:, :actual_action_dim]
        
        # Get the predicted action horizon at this timestep
        pred_horizon = pred_actions[timestep] if pred_actions.ndim > 1 else pred_actions  # (action_horizon, action_dim)
        
        # Truncate pred_horizon to actual dimension if it's padded  
        if pred_horizon.shape[-1] > actual_action_dim:
            pred_horizon = pred_horizon[:, :actual_action_dim]
        
        # Get corresponding ground truth (if available)
        gt_horizon = gt_actions[timestep:timestep + pred_horizon.shape[0]]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        actual_plot_dim = min(actual_action_dim, pred_horizon.shape[-1])
        
        for i in range(min(actual_plot_dim, len(axes))):
            ax = axes[i]
            
            # Plot predicted horizon
            if pred_horizon.ndim == 2:
                ax.plot(pred_horizon[:, i], label='Predicted Horizon', linewidth=2, 
                       color='red', marker='o', markersize=4)
            else:
                ax.plot([pred_horizon[i]], label='Predicted Horizon', linewidth=2, 
                       color='red', marker='o', markersize=4)
            
            # Plot ground truth if available
            if len(gt_horizon) > 0 and gt_horizon.shape[-1] > i:
                ax.plot(range(len(gt_horizon)), gt_horizon[:, i], 
                       label='Ground Truth', linewidth=2, color='blue', 
                       marker='s', markersize=4, alpha=0.7)
            
            ax.set_ylabel(f'Joint {i}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(14, len(axes)):
            axes[i].set_visible(False)
            
        axes[-4].set_xlabel('Horizon Step')
        plt.suptitle(f'Action Horizon at Timestep {timestep} - Episode {episode_id}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved action horizon plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_images_sequence(self, episode: Dict, save_path: Optional[str] = None) -> None:
        """Plot input images."""
        left_image = episode['left_wrist_image']
        right_image = episode['right_wrist_image']
        episode_id = episode['episode_id']
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Left wrist image
        if left_image is not None:
            axes[0].imshow(left_image)
            axes[0].set_title('Left Wrist Image', fontsize=12)
            axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'No Left Image', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].axis('off')
            
        # Right wrist image  
        if right_image is not None:
            axes[1].imshow(right_image)
            axes[1].set_title('Right Wrist Image', fontsize=12)
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No Right Image', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')
            
        plt.suptitle(f'Input Images - Episode {episode_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved images to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_3d_trajectory_comparison(self, episode: Dict, inference_result: Dict,
                                    save_path: Optional[str] = None) -> None:
        """Plot 3D trajectory comparison for left and right hands."""
        gt_actions = episode['gt_actions']
        pred_actions = inference_result['predicted_actions']
        initial_state = episode['state']
        episode_id = episode['episode_id']
        
        # Convert to numpy arrays if needed
        if hasattr(gt_actions, 'detach'):
            gt_actions = gt_actions.detach().cpu().numpy()
        else:
            gt_actions = np.array(gt_actions)
            
        if hasattr(pred_actions, 'detach'):
            pred_actions = pred_actions.detach().cpu().numpy()
        else:
            pred_actions = np.array(pred_actions)
            
        if hasattr(initial_state, 'detach'):
            initial_state = initial_state.detach().cpu().numpy()
        else:
            initial_state = np.array(initial_state)
        
        # Handle different shapes
        if gt_actions.ndim == 3:
            gt_actions = gt_actions.reshape(-1, gt_actions.shape[-1])
        
        # Truncate to 14 dimensions
        actual_action_dim = 14
        if gt_actions.shape[-1] > actual_action_dim:
            gt_actions = gt_actions[:, :actual_action_dim]
        if pred_actions.shape[-1] > actual_action_dim:
            pred_actions = pred_actions[:, :actual_action_dim]
        if initial_state.shape[-1] > actual_action_dim:
            initial_state = initial_state[:actual_action_dim]
        
        # Ensure same length
        min_len = min(len(gt_actions), len(pred_actions))
        gt_actions = gt_actions[:min_len]
        pred_actions = pred_actions[:min_len]
        
        # Extract initial positions (first 3 dims for left hand, dims 7-9 for right hand)
        left_hand_init_pos = initial_state[:3]  # xyz for left hand
        right_hand_init_pos = initial_state[7:10]  # xyz for right hand
        
        # Compute cumulative trajectories from delta actions
        # GT trajectories
        gt_left_deltas = gt_actions[:, :3]  # Left hand xyz deltas
        gt_right_deltas = gt_actions[:, 7:10]  # Right hand xyz deltas
        
        gt_left_trajectory = np.zeros((min_len + 1, 3))
        gt_right_trajectory = np.zeros((min_len + 1, 3))
        
        gt_left_trajectory[0] = left_hand_init_pos
        gt_right_trajectory[0] = right_hand_init_pos
        
        for i in range(min_len):
            gt_left_trajectory[i + 1] = gt_left_trajectory[i] + gt_left_deltas[i]
            gt_right_trajectory[i + 1] = gt_right_trajectory[i] + gt_right_deltas[i]
        
        # Predicted trajectories
        pred_left_deltas = pred_actions[:, :3]  # Left hand xyz deltas
        pred_right_deltas = pred_actions[:, 7:10]  # Right hand xyz deltas
        
        pred_left_trajectory = np.zeros((min_len + 1, 3))
        pred_right_trajectory = np.zeros((min_len + 1, 3))
        
        pred_left_trajectory[0] = left_hand_init_pos
        pred_right_trajectory[0] = right_hand_init_pos
        
        for i in range(min_len):
            pred_left_trajectory[i + 1] = pred_left_trajectory[i] + pred_left_deltas[i]
            pred_right_trajectory[i + 1] = pred_right_trajectory[i] + pred_right_deltas[i]
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Combined view
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot GT trajectories (dashed lines)
        ax1.plot(gt_left_trajectory[:, 0], gt_left_trajectory[:, 1], gt_left_trajectory[:, 2], 
                'r--', linewidth=2, alpha=0.8, label='GT Left Hand')
        ax1.plot(gt_right_trajectory[:, 0], gt_right_trajectory[:, 1], gt_right_trajectory[:, 2], 
                'b--', linewidth=2, alpha=0.8, label='GT Right Hand')
        
        # Plot Predicted trajectories (solid lines)
        ax1.plot(pred_left_trajectory[:, 0], pred_left_trajectory[:, 1], pred_left_trajectory[:, 2], 
                'r-', linewidth=2, alpha=0.8, label='Pred Left Hand')
        ax1.plot(pred_right_trajectory[:, 0], pred_right_trajectory[:, 1], pred_right_trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.8, label='Pred Right Hand')
        
        # Mark start and end points
        ax1.scatter(*gt_left_trajectory[0], color='red', s=100, marker='o', alpha=0.8, label='Start Left')
        ax1.scatter(*gt_right_trajectory[0], color='blue', s=100, marker='o', alpha=0.8, label='Start Right')
        ax1.scatter(*gt_left_trajectory[-1], color='red', s=100, marker='s', alpha=0.8, label='End Left')
        ax1.scatter(*gt_right_trajectory[-1], color='blue', s=100, marker='s', alpha=0.8, label='End Right')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Combined 3D Trajectories')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Left hand only
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(gt_left_trajectory[:, 0], gt_left_trajectory[:, 1], gt_left_trajectory[:, 2], 
                'r--', linewidth=3, alpha=0.8, label='GT Left Hand')
        ax2.plot(pred_left_trajectory[:, 0], pred_left_trajectory[:, 1], pred_left_trajectory[:, 2], 
                'r-', linewidth=3, alpha=0.8, label='Pred Left Hand')
        
        ax2.scatter(*gt_left_trajectory[0], color='red', s=100, marker='o', alpha=0.8)
        ax2.scatter(*gt_left_trajectory[-1], color='red', s=100, marker='s', alpha=0.8)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Left Hand Trajectory')
        ax2.legend()
        
        # Plot 3: Right hand only
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(gt_right_trajectory[:, 0], gt_right_trajectory[:, 1], gt_right_trajectory[:, 2], 
                'b--', linewidth=3, alpha=0.8, label='GT Right Hand')
        ax3.plot(pred_right_trajectory[:, 0], pred_right_trajectory[:, 1], pred_right_trajectory[:, 2], 
                'b-', linewidth=3, alpha=0.8, label='Pred Right Hand')
        
        ax3.scatter(*gt_right_trajectory[0], color='blue', s=100, marker='o', alpha=0.8)
        ax3.scatter(*gt_right_trajectory[-1], color='blue', s=100, marker='s', alpha=0.8)
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('Right Hand Trajectory')
        ax3.legend()
        
        plt.suptitle(f'3D Hand Trajectories - Episode {episode_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D trajectory plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def compute_metrics(self, episode: Dict, inference_result: Dict) -> Dict[str, float]:
        """Compute trajectory comparison metrics."""
        gt_actions = episode['gt_actions']
        pred_actions = inference_result['predicted_actions']
        
        # Handle different shapes
        if gt_actions.ndim == 3:  # (batch, horizon, action_dim)
            gt_actions = gt_actions.reshape(-1, gt_actions.shape[-1])
        
        # Determine the actual action dimension (flexiv uses 14 DOF)
        actual_action_dim = 14
        
        # Truncate gt_actions to actual dimension if it's padded
        if gt_actions.shape[-1] > actual_action_dim:
            gt_actions = gt_actions[:, :actual_action_dim]
        
        # Truncate pred_actions to actual dimension if it's padded  
        if pred_actions.shape[-1] > actual_action_dim:
            pred_actions = pred_actions[:, :actual_action_dim]
        
        if pred_actions.ndim == 2:  # (action_horizon, action_dim) - single prediction
            # Compare the predicted action sequence with the first part of gt
            min_len = min(len(gt_actions), len(pred_actions))
            gt_actions_to_compare = gt_actions[:min_len]
            pred_actions_to_compare = pred_actions[:min_len]
        else:
            min_len = min(len(gt_actions), len(pred_actions))
            gt_actions_to_compare = gt_actions[:min_len]
            pred_actions_to_compare = pred_actions[:min_len]
        
        # Ensure same action dimensions
        min_action_dim = min(gt_actions_to_compare.shape[-1], pred_actions_to_compare.shape[-1])
        gt_actions_to_compare = gt_actions_to_compare[:, :min_action_dim]
        pred_actions_to_compare = pred_actions_to_compare[:, :min_action_dim]
        
        # Convert to numpy arrays if they're tensors
        if hasattr(gt_actions_to_compare, 'detach'):  # PyTorch tensor
            gt_actions_to_compare = gt_actions_to_compare.detach().cpu().numpy()
        elif hasattr(gt_actions_to_compare, 'device'):  # JAX array
            gt_actions_to_compare = np.array(gt_actions_to_compare)
        else:
            gt_actions_to_compare = np.array(gt_actions_to_compare)
            
        if hasattr(pred_actions_to_compare, 'detach'):  # PyTorch tensor
            pred_actions_to_compare = pred_actions_to_compare.detach().cpu().numpy()
        elif hasattr(pred_actions_to_compare, 'device'):  # JAX array
            pred_actions_to_compare = np.array(pred_actions_to_compare)
        else:
            pred_actions_to_compare = np.array(pred_actions_to_compare)
        
        # Compute metrics using numpy
        diff = gt_actions_to_compare - pred_actions_to_compare
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))

        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'trajectory_length': min_len,
            'actual_action_dim': min_action_dim,
            'inference_time': inference_result['inference_time']
        }
    
    def visualize_episodes(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Visualize multiple episodes."""
        results = {
            'episodes': [],
            'aggregate_metrics': {}
        }
        
        all_metrics = []
        
        for episode in episodes:
            print(f"\nProcessing episode {episode['episode_id']}...")
            
            # Run inference
            inference_result = self.run_inference_on_episode(episode)
            
            # Compute metrics
            metrics = self.compute_metrics(episode, inference_result)
            all_metrics.append(metrics)
            
            # Create sample-specific output directory
            sample_dir = self.output_dir / f"sample_{episode['sample_id']}"
            sample_dir.mkdir(exist_ok=True)
            
            # Generate plots
            self.plot_trajectory_comparison(
                episode, inference_result,
                save_path=sample_dir / "trajectory_comparison.png"
            )
            
            self.plot_action_horizon_comparison(
                episode, inference_result, timestep=0,
                save_path=sample_dir / "action_horizon_t0.png"
            )
            
            self.plot_images_sequence(
                episode,
                save_path=sample_dir / "input_images.png"
            )
            
            # Store episode and inference result for batch 3D plotting
            episode['inference_result'] = inference_result
            
            self.plot_3d_trajectory_comparison(
                episode, inference_result,
                save_path=sample_dir / "3d_trajectories.png"
            )
            
            # Store results
            episode_result = {
                'episode_id': episode['episode_id'],
                'metrics': metrics
            }
            results['episodes'].append(episode_result)
        
        # Compute aggregate metrics
        if all_metrics:
            results['aggregate_metrics'] = {
                'mean_mse': np.mean([m['mse'] for m in all_metrics]),
                'std_mse': np.std([m['mse'] for m in all_metrics]),
                'mean_mae': np.mean([m['mae'] for m in all_metrics]),
                'std_mae': np.std([m['mae'] for m in all_metrics]),
                'mean_inference_time': np.mean([m['inference_time'] for m in all_metrics]),
                'total_episodes': len(episodes)
            }
            
            # Plot aggregate metrics
            self.plot_aggregate_metrics(all_metrics)
            
            # Plot combined 3D trajectories for all samples from the same episode
            if episodes:
                self.plot_combined_3d_trajectories(episodes, save_path=self.output_dir / "combined_3d_trajectories.png")
        
        print(f"\nVisualization complete! Results saved to {self.output_dir}")
        return results
    
    def plot_combined_3d_trajectories(self, episodes: List[Dict], save_path: Optional[str] = None) -> None:
        """Plot 3D trajectories for multiple samples from the same episode - K rows x 3 columns layout."""
        if not episodes:
            return
            
        episode_source_id = episodes[0].get('episode_source_id', 'Unknown')
        num_samples = len(episodes)
        
        # Create figure with K rows x 3 columns
        fig = plt.figure(figsize=(18, 6 * num_samples))
        
        for sample_idx, episode in enumerate(episodes):
            if 'inference_result' not in episode:
                continue
                
            inference_result = episode['inference_result']
            gt_actions = episode['gt_actions']
            pred_actions = inference_result['predicted_actions']
            initial_state = episode['state']
            frame_id = episode.get('sample_id', sample_idx)
            
            # Convert to numpy arrays if needed
            if hasattr(gt_actions, 'detach'):
                gt_actions = gt_actions.detach().cpu().numpy()
            else:
                gt_actions = np.array(gt_actions)
                
            if hasattr(pred_actions, 'detach'):
                pred_actions = pred_actions.detach().cpu().numpy()
            else:
                pred_actions = np.array(pred_actions)
                
            if hasattr(initial_state, 'detach'):
                initial_state = initial_state.detach().cpu().numpy()
            else:
                initial_state = np.array(initial_state)
            
            # Handle different shapes and truncate to 14 dimensions
            if gt_actions.ndim == 3:
                gt_actions = gt_actions.reshape(-1, gt_actions.shape[-1])
            
            actual_action_dim = 14
            if gt_actions.shape[-1] > actual_action_dim:
                gt_actions = gt_actions[:, :actual_action_dim]
            if pred_actions.shape[-1] > actual_action_dim:
                pred_actions = pred_actions[:, :actual_action_dim]
            if initial_state.shape[-1] > actual_action_dim:
                initial_state = initial_state[:actual_action_dim]
            
            # Ensure same length
            min_len = min(len(gt_actions), len(pred_actions))
            gt_actions = gt_actions[:min_len]
            pred_actions = pred_actions[:min_len]
            
            # Extract initial positions
            left_hand_init_pos = initial_state[:3]  # xyz for left hand
            right_hand_init_pos = initial_state[7:10]  # xyz for right hand
            
            # Compute cumulative trajectories from delta actions
            # GT trajectories
            gt_left_deltas = gt_actions[:, :3]
            gt_right_deltas = gt_actions[:, 7:10]
            
            gt_left_trajectory = np.zeros((min_len + 1, 3))
            gt_right_trajectory = np.zeros((min_len + 1, 3))
            
            gt_left_trajectory[0] = left_hand_init_pos
            gt_right_trajectory[0] = right_hand_init_pos
            
            for j in range(min_len):
                gt_left_trajectory[j + 1] = gt_left_trajectory[j] + gt_left_deltas[j]
                gt_right_trajectory[j + 1] = gt_right_trajectory[j] + gt_right_deltas[j]
            
            # Predicted trajectories
            pred_left_deltas = pred_actions[:, :3]
            pred_right_deltas = pred_actions[:, 7:10]
            
            pred_left_trajectory = np.zeros((min_len + 1, 3))
            pred_right_trajectory = np.zeros((min_len + 1, 3))
            
            pred_left_trajectory[0] = left_hand_init_pos
            pred_right_trajectory[0] = right_hand_init_pos
            
            for j in range(min_len):
                pred_left_trajectory[j + 1] = pred_left_trajectory[j] + pred_left_deltas[j]
                pred_right_trajectory[j + 1] = pred_right_trajectory[j] + pred_right_deltas[j]
            
            # Create subplots for this sample (row sample_idx)
            # Column 1: Combined view
            ax1 = fig.add_subplot(num_samples, 3, sample_idx * 3 + 1, projection='3d')
            
            # Plot GT trajectories (dashed lines)
            ax1.plot(gt_left_trajectory[:, 0], gt_left_trajectory[:, 1], gt_left_trajectory[:, 2], 
                    'r--', linewidth=2, alpha=0.8, label='GT Left')
            ax1.plot(gt_right_trajectory[:, 0], gt_right_trajectory[:, 1], gt_right_trajectory[:, 2], 
                    'b--', linewidth=2, alpha=0.8, label='GT Right')
            
            # Plot Predicted trajectories (solid lines)
            ax1.plot(pred_left_trajectory[:, 0], pred_left_trajectory[:, 1], pred_left_trajectory[:, 2], 
                    'r-', linewidth=2, alpha=0.8, label='Pred Left')
            ax1.plot(pred_right_trajectory[:, 0], pred_right_trajectory[:, 1], pred_right_trajectory[:, 2], 
                    'b-', linewidth=2, alpha=0.8, label='Pred Right')
            
            # Mark start points
            ax1.scatter(*gt_left_trajectory[0], color='red', s=60, marker='o', alpha=0.8)
            ax1.scatter(*gt_right_trajectory[0], color='blue', s=60, marker='s', alpha=0.8)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'Combined - Sample {frame_id}')
            if sample_idx == 0:  # Only show legend on first row
                ax1.legend(fontsize=8)
            
            # Column 2: Left hand only
            ax2 = fig.add_subplot(num_samples, 3, sample_idx * 3 + 2, projection='3d')
            
            ax2.plot(gt_left_trajectory[:, 0], gt_left_trajectory[:, 1], gt_left_trajectory[:, 2], 
                    'r--', linewidth=3, alpha=0.8, label='GT Left')
            ax2.plot(pred_left_trajectory[:, 0], pred_left_trajectory[:, 1], pred_left_trajectory[:, 2], 
                    'r-', linewidth=3, alpha=0.8, label='Pred Left')
            
            ax2.scatter(*gt_left_trajectory[0], color='red', s=60, marker='o', alpha=0.8)
            ax2.scatter(*gt_left_trajectory[-1], color='red', s=60, marker='s', alpha=0.8)
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Z (m)')
            ax2.set_title(f'Left Hand - Sample {frame_id}')
            if sample_idx == 0:
                ax2.legend(fontsize=8)
            
            # Column 3: Right hand only
            ax3 = fig.add_subplot(num_samples, 3, sample_idx * 3 + 3, projection='3d')
            
            ax3.plot(gt_right_trajectory[:, 0], gt_right_trajectory[:, 1], gt_right_trajectory[:, 2], 
                    'b--', linewidth=3, alpha=0.8, label='GT Right')
            ax3.plot(pred_right_trajectory[:, 0], pred_right_trajectory[:, 1], pred_right_trajectory[:, 2], 
                    'b-', linewidth=3, alpha=0.8, label='Pred Right')
            
            ax3.scatter(*gt_right_trajectory[0], color='blue', s=60, marker='o', alpha=0.8)
            ax3.scatter(*gt_right_trajectory[-1], color='blue', s=60, marker='s', alpha=0.8)
            
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            ax3.set_title(f'Right Hand - Sample {frame_id}')
            if sample_idx == 0:
                ax3.legend(fontsize=8)
        
        plt.suptitle(f'3D Hand Trajectories - Episode {episode_source_id} ({num_samples} samples)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined 3D trajectory plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_aggregate_metrics(self, all_metrics: List[Dict]) -> None:
        """Plot aggregate metrics across all episodes."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSE distribution
        mse_values = [m['mse'] for m in all_metrics]
        axes[0, 0].hist(mse_values, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('MSE Distribution')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        
        # MAE distribution  
        mae_values = [m['mae'] for m in all_metrics]
        axes[0, 1].hist(mae_values, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('MAE Distribution')
        axes[0, 1].set_xlabel('MAE')
        axes[0, 1].set_ylabel('Frequency')
        
        # Inference time distribution
        time_values = [m['inference_time'] for m in all_metrics]
        axes[1, 0].hist(time_values, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Inference Time Distribution')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Frequency')

        plt.suptitle('Aggregate Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / "aggregate_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregate metrics plot to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize bimanual flexiv policy inference")
    parser.add_argument("-cfg", "--config", type=str, default="pi05_flexiv_pick",
                       help="Configuration name")
    parser.add_argument("-ckpt", "--checkpoint", type=str, required=True,
                       help="Path to model checkpoint or GCS path")
    parser.add_argument("--episode-id", type=int, default=0,
                       help="Episode ID to visualize")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize from the episode")
    parser.add_argument("--output-dir", type=str, default="flexiv_visualization_output",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Download checkpoint if it's a GCS path
    if args.checkpoint.startswith("gs://"):
        print(f"Downloading checkpoint from {args.checkpoint}")
        checkpoint_path = download.maybe_download(args.checkpoint)
    else:
        checkpoint_path = args.checkpoint
    
    # Create visualizer
    visualizer = FlexivTrajectoryVisualizer(
        config_name=args.config,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir
    )
    
    # Load samples from specified episode
    print(f"Loading {args.num_samples} samples from episode {args.episode_id}...")
    episodes = visualizer.load_real_dataset_episodes(episode_id=args.episode_id, num_samples=args.num_samples)
    
    # Run visualization
    results = visualizer.visualize_episodes(episodes)
    
    # Print summary
    if results['aggregate_metrics']:
        metrics = results['aggregate_metrics']
        print(f"\n=== Summary Statistics ===")
        print(f"Episodes processed: {metrics['total_episodes']}")
        print(f"Mean MSE: {metrics['mean_mse']:.6f} ± {metrics['std_mse']:.6f}")
        print(f"Mean MAE: {metrics['mean_mae']:.6f} ± {metrics['std_mae']:.6f}")
        print(f"Mean Inference Time: {metrics['mean_inference_time']:.2f} ms")


if __name__ == "__main__":
    main()
