"""
Shared utilities for DAgger serve and backend services.

This module contains common functions for data format conversion and validation
that are used by both the gateway service and the backend inference service.

Author: Wendi Chen
"""
import base64
import logging
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


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
    return np.concatenate((x, y, z), axis=1)


def matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot).as_rotvec()


def get_robot_type_from_config(workspace_config: str) -> str:
    config_lower = workspace_config.lower()
    if "bimanual" in config_lower or "iphonebimanual" in config_lower:
        return "bimanual_iphone_flexiv"
    elif "single" in config_lower or "iphonesingle" in config_lower:
        return "single_iphone_flexiv"
    return "single_iphone_flexiv"


def deserialize_observations(observations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize observations received via JSON/HTTP.
    Converts serialized data back to numpy arrays.
    
    Supports multiple formats:
    1. Image sequences (rgb_sequence): list of base64 strings, each representing one frame
    2. Low-dim sequences (low_dim_sequence): list of numeric lists
    3. Single images (rgb): single base64 string
    4. Single low-dim (low_dim): single numeric list
    
    Args:
        observations: Raw observations dict from HTTP request
        
    Returns:
        Deserialized observations with numpy arrays
    """
    result = {}
    
    for key, value in observations.items():
        is_image_key = 'img' in key.lower() or 'image' in key.lower()
        
        if is_image_key:
            try:
                # Case 1: List of base64 strings (rgb_sequence - multiple frames)
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                    frames = []
                    for i, frame_data in enumerate(value):
                        image_bytes = base64.b64decode(frame_data)
                        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        frames.append(image_array)
                        logger.debug(f"Decoded frame {i} for {key}, size: {len(image_array)}")
                    result[key] = np.stack(frames, axis=0)
                    logger.debug(f"Stacked {len(frames)} frames for {key}, final shape: {result[key].shape}")
                    
                # Case 2: Single base64 string (rgb - single frame)
                elif isinstance(value, str):
                    image_bytes = base64.b64decode(value)
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    result[key] = image_array
                    logger.debug(f"Decoded single frame for {key}, shape: {image_array.shape}")
                    
                # Case 3: Numeric nested list (already decoded, convert to array)
                elif isinstance(value, (list, tuple)):
                    arr = np.array(value, dtype=np.uint8)
                    result[key] = arr
                    logger.debug(f"Converted numeric list to array for {key}, shape: {arr.shape}")
                    
                else:
                    result[key] = value
                    logger.debug(f"Kept {key} as-is (type: {type(value)})")
                    
            except Exception as e:
                logger.warning(f"Failed to deserialize image data for {key}: {e}, keeping as-is")
                result[key] = value
                
        elif isinstance(value, list):
            try:
                # Check if this is a sequence of lists (low_dim_sequence)
                if len(value) > 0 and isinstance(value[0], (list, tuple)):
                    # List of lists - convert to 2D array
                    arr = np.array(value, dtype=np.float32)
                    result[key] = arr
                    logger.debug(f"Converted low_dim_sequence for {key}, shape: {arr.shape}")
                else:
                    # Single list - convert to 1D array
                    arr = np.array(value, dtype=np.float32)
                    result[key] = arr
                    logger.debug(f"Converted low_dim for {key}, shape: {arr.shape}")
            except Exception as e:
                logger.warning(f"Failed to convert {key} to array: {e}, keeping as-is")
                result[key] = value
                
        elif isinstance(value, dict):
            # Recursively deserialize nested dicts
            result[key] = deserialize_observations(value)
            
        else:
            result[key] = value
    
    return result


def convert_single_frame_to_lerobot_format(
    observations: Dict[str, Any],
    robot_type: str,
    task: str = "",
) -> Dict[str, Any]:
    """
    Convert a single frame of raw observation data to LeRobot-compatible format.
    Uses the same conversion logic as convert_episode_to_lerobot_format in iphone_zarr_to_lerobot.py.
    
    Args:
        observations: Dict containing raw observation data with keys like 
                      'left_robot_tcp_pose', 'left_robot_gripper_width', 
                      'left_wrist_img', etc.
        robot_type: Either "single_iphone_flexiv" or "bimanual_iphone_flexiv"
        task: Task description string
    
    Returns:
        Dict with converted data including 'state' and image keys
    """
    result = {}
    
    if "state" in observations:
        state = np.asarray(observations["state"])
        expected_dim = 7 if robot_type == "single_iphone_flexiv" else 14
        if state.shape[-1] == expected_dim:
            result["state"] = state.astype(np.float32)
            for key in ["left_wrist_img", "right_wrist_img", "task", "prompt"]:
                if key in observations:
                    result[key] = observations[key]
            if task and "task" not in result and "prompt" not in result:
                result["task"] = task
            return result
    
    if "left_robot_tcp_pose" in observations:
        left_pose_9d = np.asarray(observations["left_robot_tcp_pose"])
        squeeze_output = False
        if left_pose_9d.ndim == 1:
            left_pose_9d = left_pose_9d[np.newaxis, :]
            squeeze_output = True
            
        left_pose = np.zeros((left_pose_9d.shape[0], 6), dtype=np.float32)
        for i in range(left_pose.shape[0]):
            left_pose[i][:3] = left_pose_9d[i][:3]
            left_pose[i][3:] = matrix_to_rotvec(ortho6d_to_matrix(left_pose_9d[i][3:9]))
        
        left_gripper = np.asarray(observations.get("left_robot_gripper_width", [[0.0]]))
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, np.newaxis]
        left_state = np.concatenate([left_pose, left_gripper], axis=1)
        
        if robot_type == "single_iphone_flexiv":
            state = left_state
        else:
            right_pose_9d = np.asarray(observations["right_robot_tcp_pose"])
            if right_pose_9d.ndim == 1:
                right_pose_9d = right_pose_9d[np.newaxis, :]
            right_pose = np.zeros((right_pose_9d.shape[0], 6), dtype=np.float32)
            for i in range(right_pose.shape[0]):
                right_pose[i][:3] = right_pose_9d[i][:3]
                right_pose[i][3:] = matrix_to_rotvec(ortho6d_to_matrix(right_pose_9d[i][3:9]))
            
            right_gripper = np.asarray(observations.get("right_robot_gripper_width", [[0.0]]))
            if right_gripper.ndim == 1:
                right_gripper = right_gripper[:, np.newaxis]
            right_state = np.concatenate([right_pose, right_gripper], axis=1)
            state = np.concatenate([left_state, right_state], axis=1)
        
        if squeeze_output:
            state = state.squeeze(0)
        result["state"] = state.astype(np.float32)
    
    if "left_wrist_img" in observations:
        result["left_wrist_img"] = observations["left_wrist_img"]
    if "right_wrist_img" in observations:
        result["right_wrist_img"] = observations["right_wrist_img"]
    
    if task:
        result["task"] = task
    elif "task" in observations:
        result["task"] = observations["task"]
    elif "prompt" in observations:
        result["task"] = observations["prompt"]
    
    return result


class TaskObsSpec:
    """
    Validates observation format based on openpi config.
    Dynamically extracts expected keys from config's repack_transforms.
    """

    def __init__(self, workspace_config: str):
        from openpi.training import config as _config
        
        self.workspace_config = workspace_config
        self.robot_type = get_robot_type_from_config(workspace_config)
        self.action_dim = None
        self.action_horizon = None
        self.obs_spec = self._load_obs_spec(_config)
        logger.info(f"TaskObsSpec initialized for {workspace_config}, robot_type: {self.robot_type}, action_dim: {self.action_dim}, action_horizon: {self.action_horizon}")

    def _extract_source_keys_from_repack(self, repack_transforms) -> Dict[str, str]:
        """
        Extract source keys (client-side keys) from repack_transforms.
        RepackTransform structure: {target_key: source_key}
        Returns: {source_key: target_key}
        """
        source_keys = {}
        for transform in repack_transforms.inputs:
            if hasattr(transform, "structure"):
                structure = transform.structure
                self._flatten_structure(structure, source_keys)
        return source_keys

    def _flatten_structure(self, structure: Any, result: Dict[str, str], prefix: str = ""):
        if isinstance(structure, dict):
            for target_key, value in structure.items():
                full_target = f"{prefix}/{target_key}" if prefix else target_key
                if isinstance(value, str):
                    result[value] = full_target
                elif isinstance(value, dict):
                    self._flatten_structure(value, result, full_target)

    def _load_obs_spec(self, _config) -> Dict[str, Dict[str, Any]]:
        try:
            train_config = _config.get_config(self.workspace_config)
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

            obs_spec = {}

            source_to_target = self._extract_source_keys_from_repack(data_config.repack_transforms)

            action_dim = train_config.model.action_dim
            action_horizon = train_config.model.action_horizon
            state_dim = action_dim
            self.action_dim = action_dim
            self.action_horizon = action_horizon

            for source_key, target_key in source_to_target.items():
                key_lower = source_key.lower()
                target_lower = target_key.lower()

                if "image" in key_lower or "img" in key_lower or "wrist" in key_lower:
                    obs_spec[source_key] = {"shape": [224, 224, 3], "type": "rgb"}
                elif "state" in key_lower:
                    obs_spec[source_key] = {"shape": [state_dim], "type": "low_dim"}
                elif "action" in key_lower:
                    obs_spec[source_key] = {"shape": [action_dim], "type": "low_dim"}
                elif "task" in key_lower or "prompt" in key_lower:
                    obs_spec[source_key] = {"type": "text"}
                elif "gripper" in key_lower:
                    obs_spec[source_key] = {"shape": [1], "type": "low_dim"}
                elif "joint" in key_lower or "position" in key_lower:
                    obs_spec[source_key] = {"shape": [7], "type": "low_dim"}
                else:
                    obs_spec[source_key] = {"type": "unknown"}

            if "prompt" not in obs_spec and "task" not in obs_spec:
                obs_spec["prompt"] = {"type": "text"}

            logger.info(f"Loaded obs spec from config {self.workspace_config}: {list(obs_spec.keys())}")
            return obs_spec

        except Exception as e:
            logger.warning(f"Failed to load obs spec from config {self.workspace_config}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return {
                "state": {"shape": [7], "type": "low_dim"},
                "image": {"shape": [224, 224, 3], "type": "rgb"},
                "prompt": {"type": "text"},
            }

    def process_observations(self, observations: Dict[str, Any], task: str = "") -> tuple[Dict[str, Any], bool, str]:
        """
        Process, reshape and validate observations in one unified function.
        
        Checks if data (in (n, x) or (x,) format) can be reshaped to match task config,
        performs the reshape, and returns validation status.
        
        Args:
            observations: Raw observations dict
            task: Task description string
            
        Returns:
            Tuple of (processed_observations, is_valid, error_message)
        """
        if not observations:
            return {}, False, "Observations cannot be empty"
        
        errors = []
        
        # First convert to LeRobot format
        result = convert_single_frame_to_lerobot_format(observations, self.robot_type, task)
        
        # Process and validate each key according to obs_spec
        processed_result = {}
        
        for key, spec in self.obs_spec.items():
            spec_type = spec.get("type", "unknown")
            expected_shape = spec.get("shape")
            
            # Skip text fields and action fields for validation
            if spec_type == "text":
                if key in result:
                    processed_result[key] = result[key]
                continue
            if "action" in key.lower():
                continue
            
            # Find the key in result (with alternative key formats)
            obs_key = key
            if key not in result:
                alt_keys = [
                    key.replace("/", "."),
                    key.replace("observation/", ""),
                    key.split("/")[-1],
                ]
                found_key = None
                for alt_key in alt_keys:
                    if alt_key in result:
                        found_key = alt_key
                        break
                
                if found_key:
                    obs_key = found_key
                else:
                    errors.append(f"Missing key: {key}")
                    continue
            
            value = result[obs_key]
            
            # Process based on type
            if spec_type == "rgb" and expected_shape:
                reshaped, reshape_error = self._try_reshape_image(value, expected_shape, obs_key)
                if reshape_error:
                    errors.append(reshape_error)
                else:
                    processed_result[obs_key] = reshaped
            elif spec_type == "low_dim" and expected_shape:
                reshaped, shape_error = self._try_reshape_low_dim(value, expected_shape, obs_key)
                if shape_error:
                    errors.append(shape_error)
                else:
                    processed_result[obs_key] = reshaped
            else:
                processed_result[obs_key] = value
        
        # Copy any keys not in obs_spec
        for key, value in result.items():
            if key not in processed_result:
                processed_result[key] = value
        
        is_valid = len(errors) == 0
        error_msg = " | ".join(errors) if errors else "Valid"
        
        return processed_result, is_valid, error_msg
    
    def _try_reshape_image(self, image: np.ndarray, expected_shape: list, key: str) -> tuple[np.ndarray, str]:
        """
        Validate and reshape image to expected [H, W, C].
        Only supports ndim 1 or 2. When ndim is 2, first dim must be 1 and will be squeezed.
        
        Returns:
            Tuple of (image, error_message). error_message is empty if valid.
        """
        expected_h, expected_w, expected_c = expected_shape
        expected_total = expected_h * expected_w * expected_c
        
        try:
            if image.ndim == 1:
                if image.shape[0] == expected_total:
                    reshaped = image.reshape(expected_h, expected_w, expected_c)
                    return reshaped, ""
                else:
                    return image, f"{key}: cannot reshape flattened array of size {image.shape[0]} to {expected_shape}"
            
            if image.ndim == 2:
                if image.shape[0] != 1:
                    return image, f"{key}: when ndim is 2, first dim must be 1, got shape {image.shape}"
                flat_size = image.shape[1]
                if flat_size == expected_total:
                    reshaped = image.reshape(expected_h, expected_w, expected_c)
                    return reshaped, ""
                else:
                    return image, f"{key}: cannot reshape 2D array {image.shape} to {expected_shape}"
            
            return image, f"{key}: image ndim must be 1 or 2, got {image.ndim}"
                
        except Exception as e:
            return image, f"{key}: failed to validate image: {e}"
    
    def _try_reshape_low_dim(self, data: np.ndarray, expected_shape: list, key: str) -> tuple[np.ndarray, str]:
        """
        Validate and reshape low_dim data to expected shape.
        Only supports ndim 1 or 2. When ndim is 2, first dim must be 1 and will be squeezed.
        
        Returns:
            Tuple of (data, error_message). error_message is empty if valid.
        """
        expected_dim = expected_shape[0] if expected_shape else None
        
        try:
            if data.ndim == 1:
                return data, ""
            
            if data.ndim == 2:
                if data.shape[0] != 1:
                    return data, f"{key}: when ndim is 2, first dim must be 1, got shape {data.shape}"
                squeezed = data.squeeze(0)
                return squeezed, ""
            
            return data, f"{key}: low_dim ndim must be 1 or 2, got {data.ndim}"
                
        except Exception as e:
            return data, f"{key}: failed to validate low_dim: {e}"