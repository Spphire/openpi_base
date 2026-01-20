"""
DAgger backend inference service for OpenPi.

This service provides the actual inference backend that manages inference sessions,
handles checkpoint hot-reloading, and processes inference requests.

Author: Wendi Chen
"""
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

sys.path.insert(0, str(Path(__file__).parent))
from dagger_utils import TaskObsSpec, deserialize_observations
import scipy.spatial.transform as st
from openpi.training.dsl_pose_utils import mat_to_rot6d

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def convert_action_rotation_to_6d(action: np.ndarray) -> np.ndarray:
    """
    Convert action rotation from euler angles (xyz order) to 6D representation.
    
    Input shape: [..., 7] where 7 = 3 (pos) + 3 (euler_xyz) + 1 (gripper)
    Output shape: [..., 10] where 10 = 3 (pos) + 6 (rot_6d) + 1 (gripper)
    """
    pos = action[..., :3]
    euler_xyz = action[..., 3:6]
    gripper = action[..., 6:7]
    
    # Convert euler angles (xyz order) to rotation matrix
    original_shape = euler_xyz.shape[:-1]
    euler_xyz_flat = euler_xyz.reshape(-1, 3)
    
    rot_matrices = st.Rotation.from_euler('xyz', euler_xyz_flat, degrees=False).as_matrix()
    rot_6d = mat_to_rot6d(rot_matrices)
    
    rot_6d = rot_6d.reshape(original_shape + (6,))
    
    # Concatenate: pos (3) + rot_6d (6) + gripper (1) = 10
    result = np.concatenate([pos, rot_6d, gripper], axis=-1)
    return result

class InferenceRequest(BaseModel):
    request_id: str
    device_id: str
    workspace_config: str  # maps to config_name in openpi
    task_config: str  # maps to task description/prompt
    checkpoint_path: str
    observations: Dict[str, Any]
    debug: bool = False
    rotation_rep: str = "6d"  # "6d" or "rpy", default to 6d representation


class InferenceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InferenceSession:
    def __init__(
        self,
        session_id: str,
        workspace_config: str,
        task_config: str,
        checkpoint_path: str,
    ):
        self.session_id = session_id
        self.workspace_config = workspace_config
        self.task_config = task_config
        self.initial_checkpoint_path = checkpoint_path  # Immutable, used for session identification
        self.checkpoint_path = checkpoint_path  # Mutable, can be updated via hot-reload
        self.last_access = time.time()
        self.policy: Optional[_policy.Policy] = None
        self.lock = threading.Lock()
        self.initialized = False
        # Reference counting for thread-safe cleanup
        self._ref_count = 0
        self._ref_lock = threading.Lock()

    def acquire(self):
        """Increment reference count to prevent cleanup while in use."""
        with self._ref_lock:
            self._ref_count += 1

    def release(self):
        """Decrement reference count."""
        with self._ref_lock:
            self._ref_count -= 1

    def is_in_use(self) -> bool:
        """Check if session is currently being used."""
        with self._ref_lock:
            return self._ref_count > 0

    def _initialize_session(self):
        """Initialize policy for this session."""
        try:
            logger.info(f"Initializing session {self.session_id}")
            logger.info(f"  workspace_config: {self.workspace_config}")
            logger.info(f"  checkpoint_path: {self.checkpoint_path}")

            # Get training config
            train_config = _config.get_config(self.workspace_config)

            # Create policy from checkpoint
            checkpoint_dir = Path(self.checkpoint_path)
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

            # Get data config to extract repack_transforms for inference
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

            self.policy = _policy_config.create_trained_policy(
                train_config,
                checkpoint_dir,
                repack_transforms=data_config.repack_transforms,
                default_prompt=self.task_config if self.task_config else None,
            )

            self.initialized = True
            logger.info(f"Session {self.session_id} initialized successfully")

        except Exception as e:
            import traceback

            logger.error(f"Failed to initialize session {self.session_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._cleanup_internal()
            raise

    def predict_action(self, observations: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Perform inference with the loaded policy."""
        with self.lock:
            if not self.initialized:
                self._initialize_session()

            self.last_access = time.time()
            logger.info(f"[Inference] request_id={request_id}, checkpoint={self.checkpoint_path}")

            try:
                # The policy.infer() method expects observations in a specific format
                # We need to convert the observations to the expected format
                result = self.policy.infer(observations)

                return {
                    "action": result["actions"][np.newaxis, ...].tolist(),
                    "session_id": self.session_id,
                    "request_id": request_id,
                }

            except Exception as e:
                import traceback

                logger.error(f"Inference failed for session {self.session_id}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise

    def _cleanup_internal(self):
        """Internal cleanup without lock acquisition. Must be called with self.lock held."""
        logger.info(f"Cleaning up session {self.session_id}")
        if self.policy is not None:
            del self.policy
            self.policy = None
        self.initialized = False

    def cleanup(self, wait_timeout: float = 5.0):
        """
        Clean up resources. Thread-safe: waits for ongoing operations to complete.

        Args:
            wait_timeout: Maximum time to wait for ongoing operations (seconds)
        """
        start_time = time.time()
        while self.is_in_use():
            if time.time() - start_time > wait_timeout:
                logger.warning(f"Session {self.session_id} cleanup timeout, forcing cleanup")
                break
            time.sleep(0.01)

        with self.lock:
            self._cleanup_internal()

    def reload_checkpoint(self, new_checkpoint_path: str) -> bool:
        """
        Hot-reload checkpoint without full re-initialization.
        Thread-safe: acquires lock before reloading.

        Args:
            new_checkpoint_path: Path to the new checkpoint directory

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                logger.info(f"Reloading checkpoint for session {self.session_id}")
                logger.info(f"  Old checkpoint: {self.checkpoint_path}")
                logger.info(f"  New checkpoint: {new_checkpoint_path}")

                if not Path(new_checkpoint_path).exists():
                    logger.error(f"New checkpoint not found: {new_checkpoint_path}")
                    return False

                # For openpi, we need to re-create the policy with new checkpoint
                # Full re-initialization is required because the model params are loaded differently
                train_config = _config.get_config(self.workspace_config)

                # Get data config to extract repack_transforms for inference
                data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

                self.policy = _policy_config.create_trained_policy(
                    train_config,
                    new_checkpoint_path,
                    repack_transforms=data_config.repack_transforms,
                    default_prompt=self.task_config if self.task_config else None,
                )

                # Update checkpoint path
                self.checkpoint_path = new_checkpoint_path
                self.last_access = time.time()

                logger.info(f"Session {self.session_id} checkpoint reloaded successfully")
                return True

            except Exception as e:
                import traceback

                logger.error(f"Failed to reload checkpoint for session {self.session_id}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                return False


class CheckpointUpdateManager:
    """
    Manages checkpoint updates from training server.
    Provides thread-safe checkpoint reception and session updates.
    """

    def __init__(self, checkpoint_dir: str = ".cache/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._update_in_progress = False
        self._current_checkpoint: Optional[str] = None
        self._update_history = []

    def receive_checkpoint(
        self,
        checkpoint_file: bytes,
        workspace_config: str,
        task_config: str,
        sha256: str,
        file_size: int,
        metadata: Dict[str, Any],
        is_archive: bool = False,
    ) -> Optional[str]:
        """
        Receive and validate a checkpoint file.

        Args:
            checkpoint_file: Checkpoint file bytes
            workspace_config: Workspace configuration name
            task_config: Task configuration name
            sha256: Expected SHA256 hash
            file_size: Expected file size
            metadata: Additional metadata (global_step, epoch, etc.)
            is_archive: Whether the file is a tar.gz archive

        Returns:
            Path to saved checkpoint or None if failed
        """
        with self._lock:
            try:
                self._update_in_progress = True

                # Validate file size
                if len(checkpoint_file) != file_size:
                    logger.error(f"Checkpoint size mismatch: {len(checkpoint_file)} != {file_size}")
                    return None

                # Validate SHA256
                computed_sha256 = hashlib.sha256(checkpoint_file).hexdigest()
                if computed_sha256 != sha256:
                    logger.error(f"Checkpoint SHA256 mismatch: {computed_sha256} != {sha256}")
                    return None

                # Generate checkpoint directory name
                global_step = metadata.get("global_step", 0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"{workspace_config}_{task_config}_step{global_step}_{timestamp}"
                checkpoint_path = self.checkpoint_dir / checkpoint_name

                if is_archive:
                    # Extract tar.gz archive
                    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                        tmp_file.write(checkpoint_file)
                        tmp_path = tmp_file.name

                    try:
                        checkpoint_path.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(tmp_path, "r:gz") as tar:
                            tar.extractall(path=checkpoint_path)
                        # If the archive contains a single directory, move its contents up
                        contents = list(checkpoint_path.iterdir())
                        if len(contents) == 1 and contents[0].is_dir():
                            inner_dir = contents[0]
                            for item in inner_dir.iterdir():
                                shutil.move(str(item), str(checkpoint_path))
                            inner_dir.rmdir()
                    finally:
                        os.unlink(tmp_path)
                else:
                    # Write checkpoint file directly
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    ckpt_file = checkpoint_path / "checkpoint.ckpt"
                    with open(ckpt_file, "wb") as f:
                        f.write(checkpoint_file)

                logger.info(f"Checkpoint saved to: {checkpoint_path}")

                # Update state
                self._current_checkpoint = str(checkpoint_path)
                self._update_history.append(
                    {
                        "path": str(checkpoint_path),
                        "timestamp": timestamp,
                        "metadata": metadata,
                    }
                )

                # Keep only last 5 checkpoints
                self._cleanup_old_checkpoints()

                return str(checkpoint_path)

            except Exception as e:
                logger.error(f"Failed to receive checkpoint: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return None
            finally:
                self._update_in_progress = False

    def _cleanup_old_checkpoints(self, keep_n: int = 5):
        """Remove old checkpoints, keeping only the most recent N."""
        try:
            if len(self._update_history) > keep_n:
                old_entries = self._update_history[:-keep_n]
                self._update_history = self._update_history[-keep_n:]

                for entry in old_entries:
                    old_path = Path(entry["path"])
                    if old_path.exists():
                        shutil.rmtree(old_path)
                        logger.info(f"Removed old checkpoint: {old_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

    @property
    def current_checkpoint(self) -> Optional[str]:
        with self._lock:
            return self._current_checkpoint

    @property
    def is_updating(self) -> bool:
        with self._lock:
            return self._update_in_progress


class SessionManager:
    def __init__(self, max_sessions: int = 2, session_timeout: int = 300):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: OrderedDict[str, InferenceSession] = OrderedDict()
        self.lock = threading.Lock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def _generate_session_id(self, workspace_config: str, task_config: str, checkpoint_path: str) -> str:
        """Generate a unique session ID based on configuration."""
        config_str = f"{workspace_config}#{task_config}#{checkpoint_path}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _evict_oldest_session(self):
        """Evict the oldest session to make room for new one."""
        if self.sessions:
            oldest_session_id = next(iter(self.sessions))
            oldest_session = self.sessions.pop(oldest_session_id)
            oldest_session.cleanup()
            logger.info(f"Evicted oldest session: {oldest_session_id}")

    def get_or_create_session(
        self, workspace_config: str, task_config: str, checkpoint_path: str
    ) -> InferenceSession:
        """Get existing session or create new one."""
        session_id = self._generate_session_id(workspace_config, task_config, checkpoint_path)

        with self.lock:
            # Check if session already exists
            if session_id in self.sessions:
                # Move to end (most recently used)
                session = self.sessions.pop(session_id)
                self.sessions[session_id] = session
                return session

            # Check if we need to evict sessions
            while len(self.sessions) >= self.max_sessions:
                self._evict_oldest_session()

            # Create new session
            session = InferenceSession(session_id, workspace_config, task_config, checkpoint_path)
            self.sessions[session_id] = session

            logger.info(f"Created new session: {session_id}")
            return session

    def _cleanup_loop(self):
        """Background thread to clean up expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []

                with self.lock:
                    for session_id, session in list(self.sessions.items()):
                        if current_time - session.last_access > self.session_timeout:
                            expired_sessions.append(session_id)

                    for session_id in expired_sessions:
                        session = self.sessions.pop(session_id)
                        session.cleanup()
                        logger.info(f"Cleaned up expired session: {session_id}")

                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(10)

    def update_all_sessions_checkpoint(
        self,
        workspace_config: str,
        task_config: str,
        new_checkpoint_path: str,
    ) -> Dict[str, bool]:
        """
        Update checkpoint for all matching sessions.

        Session ID is based on initial_checkpoint_path and remains unchanged
        after checkpoint hot-reload, allowing session reuse.

        Args:
            workspace_config: Target workspace config
            task_config: Target task config
            new_checkpoint_path: Path to new checkpoint

        Returns:
            Dict mapping session_id to update success status
        """
        results = {}

        with self.lock:
            for session_id, session in list(self.sessions.items()):
                if session.workspace_config == workspace_config and session.task_config == task_config:
                    success = session.reload_checkpoint(new_checkpoint_path)
                    results[session_id] = success

        return results

    def get_sessions_info(self) -> Dict[str, Dict]:
        """Get information about all active sessions."""
        with self.lock:
            info = {}
            for session_id, session in self.sessions.items():
                info[session_id] = {
                    "workspace_config": session.workspace_config,
                    "task_config": session.task_config,
                    "initial_checkpoint_path": session.initial_checkpoint_path,
                    "checkpoint_path": session.checkpoint_path,
                    "initialized": session.initialized,
                    "last_access": session.last_access,
                }
            return info


# Global managers
session_manager = SessionManager(max_sessions=2, session_timeout=300)
checkpoint_update_manager = CheckpointUpdateManager()

# FastAPI app
app = FastAPI(title="OpenPi DAgger Backend Inference Service", version="1.0.0")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "openpi-dagger-backend"}


@app.post("/backend/inference", response_model=InferenceResponse)
async def backend_inference(request: InferenceRequest):
    session = None
    try:
        logger.info(f"Received inference request for device {request.device_id}, request_id: {request.request_id}")

        # Deserialize observations (convert serialized data back to numpy arrays)
        deserialized_observations = deserialize_observations(request.observations)
        
        # Create task spec for observation processing
        task_spec = TaskObsSpec(request.workspace_config)
        
        # Process, reshape and validate observations (handles 9D pose -> 6D pose conversion)
        converted_observations, is_valid, error_msg = task_spec.process_observations(
            deserialized_observations, 
            task=request.task_config
        )
        
        if not is_valid:
            logger.warning(f"Observation validation warning: {error_msg}")
        
        # Add dummy actions field to satisfy transforms requirement
        # Shape: [action_horizon, action_dim]
        if task_spec.action_dim is not None and task_spec.action_horizon is not None:
            converted_observations["actions"] = np.zeros((task_spec.action_horizon, task_spec.action_dim), dtype=np.float32)

        # Get or create session
        session = session_manager.get_or_create_session(
            request.workspace_config,
            request.task_config,
            request.checkpoint_path,
        )

        # Acquire reference to prevent cleanup during inference
        session.acquire()

        # Perform inference with converted observations
        result = session.predict_action(converted_observations, request.request_id)

        # Convert rotation representation if needed
        if request.rotation_rep == "6d":
            action = np.array(result["action"])
            action_6d = convert_action_rotation_to_6d(action)
            result["action"] = action_6d.tolist()
        # If rotation_rep is "rpy", keep the original 3D axis-angle representation

        return InferenceResponse(success=True, data=result)

    except Exception as e:
        logger.error(f"Backend inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return InferenceResponse(success=False, error=str(e))
    finally:
        if session is not None:
            session.release()


class CheckpointUpdateResponse(BaseModel):
    success: bool
    checkpoint_path: Optional[str] = None
    updated_sessions: Optional[Dict[str, bool]] = None
    error: Optional[str] = None


@app.post("/backend/update_checkpoint", response_model=CheckpointUpdateResponse)
async def update_checkpoint(
    checkpoint_file: UploadFile = File(...),
    workspace_config: str = Form(...),
    task_config: str = Form(...),
    sha256: str = Form(...),
    file_size: str = Form(...),
    global_step: str = Form("0"),
    epoch: str = Form("0"),
    is_archive: str = Form("false"),
):
    """
    Receive a checkpoint file from the training server and update all matching sessions.

    This endpoint:
    1. Validates and saves the checkpoint file
    2. Updates all active sessions with matching workspace/task config
    3. Returns update status for each affected session
    """
    try:
        logger.info("Received checkpoint update request")
        logger.info(f"  workspace_config: {workspace_config}")
        logger.info(f"  task_config: {task_config}")
        logger.info(f"  file_size: {file_size}")
        logger.info(f"  global_step: {global_step}")

        # Read checkpoint file
        checkpoint_bytes = await checkpoint_file.read()

        # Receive and validate checkpoint
        metadata = {
            "global_step": int(global_step),
            "epoch": int(epoch),
        }

        checkpoint_path = checkpoint_update_manager.receive_checkpoint(
            checkpoint_file=checkpoint_bytes,
            workspace_config=workspace_config,
            task_config=task_config,
            sha256=sha256,
            file_size=int(file_size),
            metadata=metadata,
            is_archive=is_archive.lower() == "true",
        )

        if checkpoint_path is None:
            return CheckpointUpdateResponse(success=False, error="Failed to receive or validate checkpoint")

        # Update all matching sessions
        updated_sessions = session_manager.update_all_sessions_checkpoint(
            workspace_config=workspace_config,
            task_config=task_config,
            new_checkpoint_path=checkpoint_path,
        )

        logger.info(f"Checkpoint update complete. Updated sessions: {updated_sessions}")

        return CheckpointUpdateResponse(
            success=True,
            checkpoint_path=checkpoint_path,
            updated_sessions=updated_sessions,
        )

    except Exception as e:
        import traceback

        logger.error(f"Checkpoint update failed: {e}")
        logger.error(traceback.format_exc())
        return CheckpointUpdateResponse(success=False, error=str(e))


@app.get("/backend/sessions")
async def get_sessions():
    """Get information about all active inference sessions."""
    return {
        "sessions": session_manager.get_sessions_info(),
        "current_checkpoint": checkpoint_update_manager.current_checkpoint,
        "update_in_progress": checkpoint_update_manager.is_updating,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
