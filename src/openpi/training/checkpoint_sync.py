"""
Checkpoint synchronization client for pushing checkpoints from training server to inference server.

Author: Wendi Chen
"""
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class CheckpointSyncServer:
    """
    HTTP client for pushing checkpoints from training server (Machine A)
    to inference server (Machine B).

    Supports:
    - File transfer with multipart upload
    - SHA256 verification
    - Retry mechanism
    - Timeout handling
    """

    def __init__(
        self,
        inference_server_url: str,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.inference_server_url = inference_server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._session = requests.Session()

    def _compute_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def push_checkpoint(
        self,
        checkpoint_path: str,
        workspace_config: str,
        task_config: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push a checkpoint file to the inference server.

        Args:
            checkpoint_path: Path to the checkpoint file or directory
            workspace_config: Workspace configuration name (maps to config_name in openpi)
            task_config: Task configuration name
            metadata: Optional metadata dict (e.g., training step, epoch)

        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return False

        # For openpi, checkpoint_path is typically a directory containing params/
        # We need to handle both file and directory cases
        if checkpoint_path.is_dir():
            # Create a tar.gz of the checkpoint directory
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "checkpoint.tar.gz"
                shutil.make_archive(
                    str(archive_path).replace(".tar.gz", ""),
                    "gztar",
                    checkpoint_path.parent,
                    checkpoint_path.name,
                )
                return self._push_file(
                    file_path=str(archive_path),
                    workspace_config=workspace_config,
                    task_config=task_config,
                    metadata=metadata,
                    is_archive=True,
                )
        else:
            return self._push_file(
                file_path=str(checkpoint_path),
                workspace_config=workspace_config,
                task_config=task_config,
                metadata=metadata,
                is_archive=False,
            )

    def _push_file(
        self,
        file_path: str,
        workspace_config: str,
        task_config: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_archive: bool = False,
    ) -> bool:
        file_size = os.path.getsize(file_path)
        sha256 = self._compute_sha256(file_path)

        logger.info(f"Pushing checkpoint: {file_path}")
        logger.info(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        logger.info(f"  SHA256: {sha256}")

        for attempt in range(self.max_retries):
            try:
                success = self._upload_checkpoint(
                    file_path=file_path,
                    workspace_config=workspace_config,
                    task_config=task_config,
                    sha256=sha256,
                    file_size=file_size,
                    metadata=metadata or {},
                    is_archive=is_archive,
                )
                if success:
                    logger.info(f"Checkpoint pushed successfully to {self.inference_server_url}")
                    return True

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        logger.error(f"Failed to push checkpoint after {self.max_retries} attempts")
        return False

    def _upload_checkpoint(
        self,
        file_path: str,
        workspace_config: str,
        task_config: str,
        sha256: str,
        file_size: int,
        metadata: Dict[str, Any],
        is_archive: bool = False,
    ) -> bool:
        url = f"{self.inference_server_url}/backend/update_checkpoint"

        with open(file_path, "rb") as f:
            files = {"checkpoint_file": (os.path.basename(file_path), f, "application/octet-stream")}
            data = {
                "workspace_config": workspace_config,
                "task_config": task_config,
                "sha256": sha256,
                "file_size": str(file_size),
                "global_step": str(metadata.get("global_step", 0)),
                "epoch": str(metadata.get("epoch", 0)),
                "is_archive": str(is_archive).lower(),
            }

            response = self._session.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                return True
            else:
                logger.error(f"Server returned error: {result.get('error', 'Unknown error')}")
                return False
        else:
            logger.error(f"Server returned status {response.status_code}: {response.text}")
            return False

    def check_server_health(self) -> bool:
        """Check if the inference server is healthy and reachable."""
        try:
            response = self._session.get(
                f"{self.inference_server_url}/health",
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
