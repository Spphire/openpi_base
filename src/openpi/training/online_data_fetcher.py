"""
Online data fetcher for fetching new episodes from cloud storage.

Author: Wendi Chen
"""
import os
import sys
sys.path.append(os.getcwd())

import hashlib
import logging
import tarfile
import tempfile
from collections import deque
from typing import Any, Dict, List, Optional

import lz4.frame
import numpy as np
import requests
import zarr

from preprocess_data.iphone_zarr_to_lerobot import convert_episode_to_lerobot_format
from preprocess_data.umi_utils import ActionType, convert_data_to_zarr


class AdaptiveSampler:
    """
    Implements SOP-style adaptive sampling between online and offline data.

    Formula: ω_on = exp(α·l̄_on) / (exp(α·l̄_on) + exp(l̄_off))
    where α > 1 is a boost factor to prioritize online data.
    """

    def __init__(
        self,
        window_size: int = 200,
        boost_factor: float = 1.5,
        min_online_ratio: float = 0.2,
        max_online_ratio: float = 0.8,
        initial_online_weight: float = 0.5,
    ):
        self.window_size = window_size
        self.boost_factor = boost_factor
        self.min_online_ratio = min_online_ratio
        self.max_online_ratio = max_online_ratio

        self.online_losses: deque = deque(maxlen=window_size)
        self.offline_losses: deque = deque(maxlen=window_size)

        self._online_weight = initial_online_weight

    def add_loss(self, loss: float, is_online: bool):
        if is_online:
            self.online_losses.append(loss)
        else:
            self.offline_losses.append(loss)

    def update_weights(self) -> float:
        if len(self.online_losses) < 10 or len(self.offline_losses) < 10:
            return self._online_weight

        l_on = np.mean(list(self.online_losses))
        l_off = np.mean(list(self.offline_losses))

        exp_on = np.exp(np.clip(self.boost_factor * l_on, -50, 50))
        exp_off = np.exp(np.clip(l_off, -50, 50))

        omega_on = exp_on / (exp_on + exp_off)
        omega_on = np.clip(omega_on, self.min_online_ratio, self.max_online_ratio)

        self._online_weight = omega_on
        return self._online_weight

    @property
    def online_weight(self) -> float:
        return self._online_weight

    @property
    def offline_weight(self) -> float:
        return 1.0 - self._online_weight

    def get_stats(self) -> Dict[str, Any]:
        return {
            "online_weight": self._online_weight,
            "offline_weight": 1.0 - self._online_weight,
            "online_loss_mean": np.mean(list(self.online_losses)) if self.online_losses else 0.0,
            "offline_loss_mean": np.mean(list(self.offline_losses)) if self.offline_losses else 0.0,
            "online_loss_count": len(self.online_losses),
            "offline_loss_count": len(self.offline_losses),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "online_losses": list(self.online_losses),
            "offline_losses": list(self.offline_losses),
            "_online_weight": self._online_weight,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.online_losses = deque(state_dict["online_losses"], maxlen=self.window_size)
        self.offline_losses = deque(state_dict["offline_losses"], maxlen=self.window_size)
        self._online_weight = state_dict["_online_weight"]


class OnlineDataFetcher:
    """
    Fetches new episodes from cloud storage and converts them to LeRobot-compatible format.
    Uses convert_data_to_zarr for data processing, similar to umi_base implementation.
    """

    def __init__(
        self,
        datacloud_endpoint: str,
        identifier: str,
        query_filter: Optional[dict] = None,
        robot_type: str = "single_iphone_flexiv",
        fps: int = 10,
        task_description: str = "",
        features: Optional[Dict[str, Dict]] = None,
        # convert_data_to_zarr parameters
        use_absolute_action: bool = True,
        action_type: str = "left_arm_6DOF_gripper_width",
        temporal_downsample_ratio: int = 0,
        use_dino: bool = False,
        episode_clip_head_seconds: float = 0.0,
        episode_clip_tail_seconds: float = 0.0,
        gripper_width_bias: float = 0.0,
        gripper_width_scale: float = 1.0,
    ):
        self.datacloud_endpoint = datacloud_endpoint
        self.identifier = identifier
        self.query_filter = query_filter or {}
        self.robot_type = robot_type
        self.fps = fps
        self.task_description = task_description
        self.features = features or self._default_features()

        # convert_data_to_zarr parameters
        self.use_absolute_action = use_absolute_action
        self.action_type = ActionType[action_type]
        self.temporal_downsample_ratio = temporal_downsample_ratio
        self.use_dino = use_dino
        self.episode_clip_head_seconds = episode_clip_head_seconds
        self.episode_clip_tail_seconds = episode_clip_tail_seconds
        self.gripper_width_bias = gripper_width_bias
        self.gripper_width_scale = gripper_width_scale

        self._fetched_uuids: set = set()

    def _default_features(self) -> Dict[str, Dict]:
        if self.robot_type == "single_iphone_flexiv":
            return {
                "left_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            }
        else:  # bimanual
            return {
                "left_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "right_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
            }

    def fetch_new_episodes(self) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Fetch new episodes from cloud storage that haven't been fetched yet.
        Returns list of episode data dicts or None if no new data.
        Each episode dict contains arrays for each feature key with shape (T, ...).
        """
        try:
            list_recordings_request = {
                "identifier": self.identifier,
                "query_filter": self.query_filter,
                "limit": 10000,
                "skip": 0,
            }
            url = f"{self.datacloud_endpoint}/v1/logs"
            response = requests.post(
                url,
                json=list_recordings_request,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                logging.warning(f"Failed to list recordings: {response.text}")
                return None

            records = response.json().get("data", [])
            all_uuids = set(record["uuid"] for record in records)
            new_uuids = all_uuids - self._fetched_uuids

            if not new_uuids:
                return None

            logging.info(f"Found {len(new_uuids)} new episodes to fetch")

            episodes_data = self._download_and_process(list(new_uuids))

            if episodes_data:
                self._fetched_uuids.update(new_uuids)

            return episodes_data

        except Exception as e:
            logging.error(f"Error fetching new episodes: {e}")
            return None

    def _download_and_process(self, uuids: List[str]) -> Optional[List[Dict[str, np.ndarray]]]:
        """Download and process episodes into LeRobot-compatible format using convert_data_to_zarr."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "downloaded_records.tar.lz4")

            try:
                data_request = {
                    "identifier": self.identifier,
                    "uuids": uuids,
                }
                response = requests.post(
                    f"{self.datacloud_endpoint}/v1/download_records",
                    json=data_request,
                    stream=True,
                    timeout=300,
                )

                if response.status_code != 200:
                    logging.error(f"Failed to download records: {response.text}")
                    return None

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                server_sha256sum = response.headers.get("X-File-SHA256")
                if server_sha256sum:
                    sha256_hash = hashlib.sha256()
                    with open(filename, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)
                    file_sha256sum = sha256_hash.hexdigest()
                    if file_sha256sum != server_sha256sum:
                        logging.error("SHA256 checksum mismatch")
                        return None

                extract_dir = os.path.join(temp_dir, "downloaded_records")
                os.makedirs(extract_dir, exist_ok=True)

                with lz4.frame.open(filename, "rb") as lz4_file:
                    with tarfile.open(fileobj=lz4_file, mode="r|") as tar:
                        tar.extractall(path=extract_dir)

                # Use convert_data_to_zarr to process downloaded data
                zarr_output_dir = os.path.join(temp_dir, "zarr_output")
                zarr_path = convert_data_to_zarr(
                    input_dir=extract_dir,
                    output_dir=zarr_output_dir,
                    temporal_downsample_ratio=self.temporal_downsample_ratio,
                    use_absolute_action=self.use_absolute_action,
                    action_type=self.action_type,
                    use_dino=self.use_dino,
                    episode_clip_head_seconds=self.episode_clip_head_seconds,
                    episode_clip_tail_seconds=self.episode_clip_tail_seconds,
                    gripper_width_bias=self.gripper_width_bias,
                    gripper_width_scale=self.gripper_width_scale,
                )

                if zarr_path and os.path.exists(zarr_path):
                    return self._load_zarr_to_episodes(zarr_path)

                return None

            except Exception as e:
                logging.error(f"Error downloading/processing episodes: {e}")
                import traceback

                logging.error(traceback.format_exc())
                return None

    def _load_zarr_to_episodes(self, zarr_path: str) -> Optional[List[Dict[str, np.ndarray]]]:
        """Load zarr data and convert to list of episode dicts in LeRobot format."""
        try:
            src_root = zarr.group(zarr_path)

            # Load meta data
            episode_ends = src_root["meta"]["episode_ends"][:]

            # Load data
            data = {}
            for key in src_root["data"].keys():
                data[key] = src_root["data"][key][:]

            # Split into episodes
            episodes = []
            episode_starts = [0] + list(episode_ends[:-1])

            for start_idx, end_idx in zip(episode_starts, episode_ends):
                raw_episode_data = {key: data[key][start_idx:end_idx] for key in data.keys()}

                # Convert to LeRobot format
                episode_data = convert_episode_to_lerobot_format(
                    episode_data=raw_episode_data,
                    robot_type=self.robot_type,
                    task=self.task_description,
                )
                episodes.append(episode_data)

            logging.info(f"Loaded {len(episodes)} episodes from zarr data")
            return episodes

        except Exception as e:
            logging.error(f"Error loading zarr data: {e}")
            import traceback

            logging.error(traceback.format_exc())
            return None

    @property
    def fetched_count(self) -> int:
        return len(self._fetched_uuids)
