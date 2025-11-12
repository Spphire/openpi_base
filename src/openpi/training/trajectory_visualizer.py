#!/usr/bin/env python3
"""
Trajectory visualization utilities for training process.
Lightweight version that saves visualizations to local files.
"""

import os
import pathlib
from typing import Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from openpi.training import checkpoints as _checkpoints
import jax
import jax.numpy as jnp
import openpi.shared.array_typing as at
import openpi.models.model as _model
import logging
import openpi.transforms as transforms

class TrainingTrajectoryVisualizer:
    """Lightweight trajectory visualizer that saves to local files."""
    
    def __init__(self, action_dim: int = 14, save_interval: int = 5000, norm_stats_dict=None):
        """
        Initialize the training trajectory visualizer.
        
        Args:
            action_dim: Dimension of actions (14 for bimanual flexiv)
            save_interval: How often to save trajectory visualizations (in steps)
        """
        self.action_dim = action_dim
        self.save_interval = save_interval
        self.norm_stats_dict = norm_stats_dict
        if self.norm_stats_dict is not None:
            norm_stats = _checkpoints.load_norm_stats(self.norm_stats_dict['assets_dirs'], self.norm_stats_dict['asset_id'])
            self.unnormalizer = transforms.Unnormalize(norm_stats, use_quantiles=self.norm_stats_dict['use_quantiles'])
        else:
            self.unnormalizer = None
            logging.warning("No norm stats path provided, not using norm stats for trajectory visualization")
        self.output_dir = None  # Will be set later by training script
        
        # Setup plotting style
        plt.style.use('default')  # Use default style for consistency
        
    def set_output_dir(self, output_dir: str):
        """Set the output directory for saving visualizations."""
        self.output_dir = pathlib.Path(output_dir) / "trajectory_visualizations"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def should_save_trajectories(self, step: int) -> bool:
        """Check if we should save trajectories at this step."""
        return step % self.save_interval == 0 and step > 0
    
    def maybe_save_trajectories(
        self, 
        step: int, 
        model: Any,
        rng: Any,
        observation: Any,
        actions: Any,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Maybe save trajectory visualizations if it's time to do so.
        
        Args:
            step: Training step number
            model: The trained model
            rng: Random key for inference
            observation: Observation data
            actions: Ground truth actions
            checkpoint_dir: Checkpoint directory path (for setting output dir)
            
        Returns:
            Dictionary of trajectory metrics (empty if no visualization saved)
        """
        # Set output directory if provided and not already set
        if checkpoint_dir is not None and self.output_dir is None:
            self.set_output_dir(checkpoint_dir)
        
        # Check if we should save trajectories at this step
        if not self.should_save_trajectories(step):
            return {}
        
        try:
            # Set model to eval mode for inference
            model.eval()
            pred_rng = jax.random.fold_in(rng, 42)  # Use different RNG for prediction
            predicted_actions = model.sample_actions(pred_rng, observation)
            
            # Save trajectories and return metrics
            metrics = self.save_trajectories_to_files(step, actions, predicted_actions)
            
            # Set back to train mode
            model.train()
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Failed to save trajectory visualizations: {e}")
            # Set back to train mode in case of error
            try:
                model.train()
            except:
                pass
            return {}
    
    def create_trajectory_comparison_plot(
        self, 
        gt_actions: np.ndarray, 
        pred_actions: np.ndarray,
        sample_idx: int = 0,
        max_samples: int = 3
    ) -> plt.Figure:
        """
        Create trajectory comparison plot for a batch of samples.
        
        Args:
            gt_actions: Ground truth actions [batch, horizon, action_dim]
            pred_actions: Predicted actions [batch, horizon, action_dim] 
            sample_idx: Which sample to visualize
            max_samples: Maximum number of samples to show
            
        Returns:
            matplotlib Figure object
        """
        # Note: gt_actions and pred_actions should already be numpy arrays
        # and unnormalized (if applicable) before calling this function
        
        # Handle different shapes
        if gt_actions.ndim == 3:  # (batch, horizon, action_dim)
            batch_size = gt_actions.shape[0]
        else:  # (horizon, action_dim) - single sample
            gt_actions = gt_actions[None, ...]  # Add batch dimension
            pred_actions = pred_actions[None, ...]
            batch_size = 1
            
        # Truncate to actual action dimension
        if gt_actions.shape[-1] > self.action_dim:
            gt_actions = gt_actions[:, :, :self.action_dim]
        if pred_actions.shape[-1] > self.action_dim:
            pred_actions = pred_actions[:, :, :self.action_dim]
            
        # Determine how many samples to show
        num_samples = min(max_samples, batch_size)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot first sample's trajectory comparison
        sample_idx = min(sample_idx, batch_size - 1)
        gt_sample = gt_actions[sample_idx]  # (horizon, action_dim)
        pred_sample = pred_actions[sample_idx]  # (horizon, action_dim)
        
        # Ensure same length
        min_len = min(len(gt_sample), len(pred_sample))
        gt_sample = gt_sample[:min_len]
        pred_sample = pred_sample[:min_len]
        
        for i in range(min(self.action_dim, len(axes))):
            ax = axes[i]
            
            # Plot ground truth and predictions
            ax.plot(gt_sample[:, i], label='Ground Truth', linewidth=2, alpha=0.8, color='blue')
            ax.plot(pred_sample[:, i], label='Predicted', linewidth=2, alpha=0.8, 
                   color='red', linestyle='--')
            
            # Add labels based on action dimension
            if i < 7:  # Left hand
                if i < 3:
                    ax.set_ylabel(f'Left XYZ[{i}]')
                elif i < 6:
                    ax.set_ylabel(f'Left Rot[{i-3}]')
                else:
                    ax.set_ylabel('Left Gripper')
            else:  # Right hand
                if i < 10:
                    ax.set_ylabel(f'Right XYZ[{i-7}]')
                elif i < 13:
                    ax.set_ylabel(f'Right Rot[{i-10}]')
                else:
                    ax.set_ylabel('Right Gripper')
                    
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(self.action_dim, len(axes)):
            axes[i].set_visible(False)
            
        axes[-4].set_xlabel('Time Step')
        plt.suptitle(f'Trajectory Comparison - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_3d_trajectory_plot(
        self,
        gt_actions: np.ndarray,
        pred_actions: np.ndarray, 
        sample_idx: int = 0
    ) -> plt.Figure:
        """
        Create 3D trajectory comparison plot.
        
        Args:
            gt_actions: Ground truth actions [batch, horizon, action_dim]
            pred_actions: Predicted actions [batch, horizon, action_dim]
            sample_idx: Which sample to visualize
            
        Returns:
            matplotlib Figure object
        """
        # Note: gt_actions and pred_actions should already be numpy arrays
        # and unnormalized (if applicable) before calling this function
            
        # Handle batch dimension
        if gt_actions.ndim == 3:
            batch_size = gt_actions.shape[0]
        else:
            gt_actions = gt_actions[None, ...]
            pred_actions = pred_actions[None, ...]
            batch_size = 1
            
        sample_idx = min(sample_idx, batch_size - 1)
        
        # Extract sample data
        gt_sample = gt_actions[sample_idx]  # (horizon, action_dim)
        pred_sample = pred_actions[sample_idx]  # (horizon, action_dim)
        
        # Truncate to action dimension
        if gt_sample.shape[-1] > self.action_dim:
            gt_sample = gt_sample[:, :self.action_dim]
        if pred_sample.shape[-1] > self.action_dim:
            pred_sample = pred_sample[:, :self.action_dim]
            
        # Ensure same length
        min_len = min(len(gt_sample), len(pred_sample))
        gt_sample = gt_sample[:min_len]
        pred_sample = pred_sample[:min_len]
        
        # Each hand: [x, y, z, rx, ry, rz, gripper_width]
        
        # Extract absolute positions from actions
        # New action format: first 7 dims for left hand, last 7 dims for right hand
        # Each hand action: [abs_x, abs_y, abs_z, abs_rx, abs_ry, abs_rz, abs_gripper_width]
        # GT trajectories - actions are already absolute positions
        gt_left_positions = gt_sample[:, :3]  # Left hand xyz absolute positions (first 3 of first 7)
        gt_right_positions = gt_sample[:, 7:10]  # Right hand xyz absolute positions (first 3 of last 7)
        
        # Predicted trajectories - actions are already absolute positions
        pred_left_positions = pred_sample[:, :3]  # Left hand xyz absolute positions (first 3 of first 7)
        pred_right_positions = pred_sample[:, 7:10]  # Right hand xyz absolute positions (first 3 of last 7)
        
        gt_left_trajectory = gt_left_positions
        gt_right_trajectory = gt_right_positions
        pred_left_trajectory = pred_left_positions
        pred_right_trajectory = pred_right_positions
        
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
        
        # Mark start points
        ax1.scatter(*gt_left_trajectory[0], color='red', s=100, marker='o', alpha=0.8)
        ax1.scatter(*gt_right_trajectory[0], color='blue', s=100, marker='o', alpha=0.8)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Combined 3D Trajectories')
        ax1.legend(fontsize=8)
        
        # Plot 2: Left hand only
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(gt_left_trajectory[:, 0], gt_left_trajectory[:, 1], gt_left_trajectory[:, 2], 
                'r--', linewidth=3, alpha=0.8, label='GT Left Hand')
        ax2.plot(pred_left_trajectory[:, 0], pred_left_trajectory[:, 1], pred_left_trajectory[:, 2], 
                'r-', linewidth=3, alpha=0.8, label='Pred Left Hand')
        
        ax2.scatter(*gt_left_trajectory[0], color='red', s=100, marker='o', alpha=0.8)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Left Hand Trajectory')
        ax2.legend(fontsize=8)
        
        # Plot 3: Right hand only
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(gt_right_trajectory[:, 0], gt_right_trajectory[:, 1], gt_right_trajectory[:, 2], 
                'b--', linewidth=3, alpha=0.8, label='GT Right Hand')
        ax3.plot(pred_right_trajectory[:, 0], pred_right_trajectory[:, 1], pred_right_trajectory[:, 2], 
                'b-', linewidth=3, alpha=0.8, label='Pred Right Hand')
        
        ax3.scatter(*gt_right_trajectory[0], color='blue', s=100, marker='o', alpha=0.8)
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('Right Hand Trajectory')
        ax3.legend(fontsize=8)
        
        plt.suptitle(f'3D Hand Trajectories - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_trajectories_to_files(
        self,
        step: int,
        gt_actions: at.Array,
        pred_actions: at.Array,
    ) -> Dict[str, float]:
        """
        Save trajectory visualizations to local files.
        
        Args:
            step: Training step
            gt_actions: Ground truth actions
            pred_actions: Predicted actions  
            
        Returns:
            Dictionary of trajectory metrics
        """
        if not self.should_save_trajectories(step) or self.output_dir is None:
            return {}
            
        metrics = {}
        
        try:
            # Convert to numpy arrays
            if hasattr(gt_actions, 'device'):  # JAX array
                gt_actions = np.array(gt_actions)
            if hasattr(pred_actions, 'device'):  # JAX array
                pred_actions = np.array(pred_actions)


            # Unnormalize actions if norm stats are provided
            if self.unnormalizer is not None and self.unnormalizer.norm_stats is not None:
                gt_actions = transforms.apply_tree(
                    {"actions": gt_actions},
                    self.unnormalizer.norm_stats,
                    self.unnormalizer._unnormalize_quantile if self.unnormalizer.use_quantiles else self.unnormalizer._unnormalize,
                    strict=False,
                )["actions"]
                pred_actions = transforms.apply_tree(
                    {"actions": pred_actions},
                    self.unnormalizer.norm_stats,
                    self.unnormalizer._unnormalize_quantile if self.unnormalizer.use_quantiles else self.unnormalizer._unnormalize,
                    strict=False,
                )["actions"]

            # Compute trajectory metrics
            metrics = compute_trajectory_metrics(gt_actions, pred_actions)
            
            # Create step-specific directory
            step_dir = self.output_dir / f"step_{step:08d}"
            step_dir.mkdir(exist_ok=True)
            
            # Save trajectory comparison plot
            traj_fig = self.create_trajectory_comparison_plot(
                gt_actions, pred_actions, sample_idx=0
            )
            traj_path = step_dir / "trajectory_comparison.png"
            traj_fig.savefig(traj_path, dpi=150, bbox_inches='tight')
            plt.close(traj_fig)
            
            # Save 3D trajectory plot 
            traj_3d_fig = self.create_3d_trajectory_plot(
                gt_actions, pred_actions, sample_idx=0
            )
            traj_3d_path = step_dir / "3d_trajectories.png"
            traj_3d_fig.savefig(traj_3d_path, dpi=150, bbox_inches='tight')
            plt.close(traj_3d_fig)
            
            # Save metrics to text file
            metrics_path = step_dir / "trajectory_metrics.txt"
            with open(metrics_path, 'w') as f:
                f.write(f"Training Step: {step}\n")
                f.write("=" * 40 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.6f}\n")
            
            print(f"Saved trajectory visualizations to {step_dir}")
                
        except Exception as e:
            print(f"Warning: Failed to save trajectory plots: {e}")
            
        return metrics


def compute_trajectory_metrics(
    gt_actions: at.Array | np.ndarray, 
    pred_actions: at.Array | np.ndarray
) -> Dict[str, float]:
    """
    Compute trajectory comparison metrics.
    
    Args:
        gt_actions: Ground truth actions
        pred_actions: Predicted actions
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if hasattr(gt_actions, 'device'):  # JAX array
        gt_actions = np.array(gt_actions)
    if hasattr(pred_actions, 'device'):  # JAX array
        pred_actions = np.array(pred_actions)
        
    # Handle different shapes and ensure same dimensions
    if gt_actions.ndim == 3:  # (batch, horizon, action_dim)
        gt_actions = gt_actions.reshape(-1, gt_actions.shape[-1])
    if pred_actions.ndim == 3:
        pred_actions = pred_actions.reshape(-1, pred_actions.shape[-1])
        
    # Truncate to same dimensions
    min_action_dim = min(gt_actions.shape[-1], pred_actions.shape[-1])
    min_len = min(len(gt_actions), len(pred_actions))
    
    gt_actions = gt_actions[:min_len, :min_action_dim]
    pred_actions = pred_actions[:min_len, :min_action_dim]
    
    # Compute metrics
    diff = gt_actions - pred_actions
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    
    # Compute per-dimension metrics
    mse_per_dim = np.mean(diff ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    
    # Compute position-specific metrics (first 3 dims of each hand)
    left_pos_mse = np.mean(mse_per_dim[:3]) if min_action_dim >= 3 else 0.0
    right_pos_mse = np.mean(mse_per_dim[7:10]) if min_action_dim >= 10 else 0.0
    
    return {
        "trajectory_mse": float(mse),
        "trajectory_mae": float(mae),
        "left_hand_pos_mse": float(left_pos_mse),
        "right_hand_pos_mse": float(right_pos_mse),
        "trajectory_length": min_len,
        "action_dim": min_action_dim
    }


# Global trajectory visualizer instance
_global_trajectory_visualizer = None


def initialize_trajectory_visualizer(action_dim: int = 14, save_interval: int = 1000, norm_stats_dict=None):
    """Initialize the global trajectory visualizer."""
    global _global_trajectory_visualizer
    _global_trajectory_visualizer = TrainingTrajectoryVisualizer(
        action_dim=action_dim,
        save_interval=save_interval,
        norm_stats_dict=norm_stats_dict,
    )


def maybe_save_training_trajectories(
    step: int,
    model: Any,
    rng: Any,
    observation: Any,
    actions: Any,
    checkpoint_dir: Optional[str] = None,
    save_interval: int = 1000,
    norm_stats_dict: Optional[dict] = None,
    num_train_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Hook function to maybe save trajectory visualizations during training.
    
    This is a simple function that can be called from the training loop
    without major modifications to the training code.
    
    Args:
        step: Training step number
        model: The trained model
        rng: Random key for inference
        observation: Observation data
        actions: Ground truth actions
        checkpoint_dir: Checkpoint directory path
        save_interval: How often to save (in steps)
        norm_stats_dict: Normalization stats dictionary
        num_train_steps: Total number of training steps. If provided, will save at the last step even if it doesn't match save_interval.
        
    Returns:
        Dictionary of trajectory metrics (empty if no visualization saved)
    """
    global _global_trajectory_visualizer
    
    if _global_trajectory_visualizer is None:
        # Initialize with default parameters if not already initialized
        initialize_trajectory_visualizer(save_interval=save_interval, norm_stats_dict=norm_stats_dict)
    
    # Check if we should save: either at regular intervals or at the last step
    should_save = _global_trajectory_visualizer.should_save_trajectories(step)
    if not should_save and num_train_steps is not None and step == num_train_steps - 1:
        # Force save at the last training step
        should_save = True
    
    if not should_save:
        return {}
    
    return _global_trajectory_visualizer.maybe_save_trajectories(
        step, model, rng, observation, actions, checkpoint_dir
    )
