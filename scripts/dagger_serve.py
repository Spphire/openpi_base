"""
DAgger serve gateway for OpenPi.

This service provides the external API gateway that validates requests and forwards
them to the backend inference service. The API is compatible with umi_base.

Author: Wendi Chen
"""
import base64
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenPi DAgger Inference Server", version="1.0.0")

# Backend URL - can be configured via environment variable
BACKEND_URL = "http://localhost:8080"


class InferenceRequest(BaseModel):
    request_id: str = Field(..., description="Request ID from client (UUID string)")
    device_id: str = Field(..., description="Device ID")
    workspace_config: str = Field(..., description="Workspace configuration name (maps to config name in openpi)")
    task_config: str = Field(..., description="Task configuration name (maps to task description/prompt)")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    observations: Dict[str, Any] = Field(..., description="Observation data matching task requirements")
    debug: bool = Field(default=False, description="Enable debug mode")


class InferenceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskObsSpec:
    """
    Validates observation format based on openpi config.
    """

    def __init__(self, workspace_config: str):
        self.workspace_config = workspace_config
        self.obs_spec = self._load_obs_spec()

    def _load_obs_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Load observation specification from openpi config.
        For openpi, we infer the expected format from the data config transforms.
        """
        try:
            train_config = _config.get_config(self.workspace_config)
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

            # Infer observation spec from model config
            # This is a simplified version - actual spec depends on robot type
            obs_spec = {}

            # Common patterns based on robot type in config name
            if "iPhoneSingle" in self.workspace_config or "single_iphone" in self.workspace_config:
                obs_spec = {
                    "observation/state": {"shape": [7], "type": "low_dim"},
                    "observation/left_wrist_image": {"shape": [224, 224, 3], "type": "rgb"},
                }
            elif "iPhoneBimanual" in self.workspace_config or "bimanual_iphone" in self.workspace_config:
                obs_spec = {
                    "observation/state": {"shape": [14], "type": "low_dim"},
                    "observation/left_wrist_image": {"shape": [224, 224, 3], "type": "rgb"},
                    "observation/right_wrist_image": {"shape": [224, 224, 3], "type": "rgb"},
                }
            elif "aloha" in self.workspace_config.lower():
                obs_spec = {
                    "images": {"shape": [3, 224, 224], "type": "rgb_dict"},  # Multiple cameras
                    "state": {"shape": [14], "type": "low_dim"},
                }
            elif "droid" in self.workspace_config.lower():
                obs_spec = {
                    "observation/image": {"shape": [224, 224, 3], "type": "rgb"},
                    "observation/wrist_image": {"shape": [224, 224, 3], "type": "rgb"},
                    "observation/joint_position": {"shape": [7], "type": "low_dim"},
                    "observation/gripper_position": {"shape": [1], "type": "low_dim"},
                }
            elif "libero" in self.workspace_config.lower():
                obs_spec = {
                    "image": {"shape": [224, 224, 3], "type": "rgb"},
                    "wrist_image": {"shape": [224, 224, 3], "type": "rgb"},
                    "state": {"shape": [7], "type": "low_dim"},
                }
            else:
                # Default spec
                obs_spec = {
                    "state": {"shape": [7], "type": "low_dim"},
                    "image": {"shape": [224, 224, 3], "type": "rgb"},
                }

            # Add prompt field
            obs_spec["prompt"] = {"type": "text"}

            return obs_spec

        except Exception as e:
            logger.warning(f"Failed to load obs spec from config {self.workspace_config}: {e}")
            # Return a minimal default spec
            return {
                "state": {"shape": [7], "type": "low_dim"},
                "image": {"shape": [224, 224, 3], "type": "rgb"},
            }

    def validate_observations(self, observations: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate observations against the expected spec.
        Returns (is_valid, error_message).
        """
        # For now, we do basic validation
        # More detailed validation can be added based on specific requirements

        if not observations:
            return False, "Observations cannot be empty"

        for key, spec in self.obs_spec.items():
            if key == "prompt":
                # Prompt is optional
                continue

            if key not in observations:
                # Check alternative key formats
                alt_keys = [
                    key.replace("/", "."),
                    key.replace("observation/", ""),
                    key.split("/")[-1],
                ]
                found = False
                for alt_key in alt_keys:
                    if alt_key in observations:
                        found = True
                        break
                if not found:
                    # Not a critical error, just log
                    logger.debug(f"Observation key {key} not found, but may be optional")

        return True, "Valid"


def decode_observations(observations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode base64-encoded images and convert observations to numpy arrays.
    """
    decoded = {}

    for key, value in observations.items():
        if isinstance(value, str):
            # Might be a base64-encoded image
            try:
                decoded_bytes = base64.b64decode(value)
                # Try to decode as image
                img_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
                # Reshape if possible (assume 224x224x3 for standard images)
                if len(img_array) == 224 * 224 * 3:
                    img_array = img_array.reshape(224, 224, 3)
                decoded[key] = img_array
            except Exception:
                # Not base64, keep as string
                decoded[key] = value
        elif isinstance(value, list):
            # Convert lists to numpy arrays
            if isinstance(value[0], str):
                # List of base64-encoded images (multi-frame)
                frames = []
                for frame_b64 in value:
                    try:
                        decoded_bytes = base64.b64decode(frame_b64)
                        img_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
                        if len(img_array) == 224 * 224 * 3:
                            img_array = img_array.reshape(224, 224, 3)
                        frames.append(img_array)
                    except Exception:
                        frames.append(frame_b64)
                decoded[key] = np.stack(frames) if all(isinstance(f, np.ndarray) for f in frames) else value
            elif isinstance(value[0], list):
                # Nested list (multi-frame low-dim data)
                decoded[key] = np.array(value)
            else:
                # Simple list (single-frame low-dim data)
                decoded[key] = np.array(value)
        elif isinstance(value, dict):
            # Nested dict (e.g., images dict)
            decoded[key] = decode_observations(value)
        else:
            decoded[key] = value

    return decoded


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "openpi-dagger-serve"}


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    try:
        # Validate workspace configuration exists
        try:
            _config.get_config(request.workspace_config)
        except ValueError as e:
            logger.error(f"Invalid workspace config: {request.workspace_config}")
            raise HTTPException(status_code=400, detail=f"Invalid workspace config: {e}")

        # Validate checkpoint exists
        checkpoint_path = Path(request.checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {request.checkpoint_path}")
            raise HTTPException(status_code=400, detail=f"Checkpoint not found: {request.checkpoint_path}")

        # Validate observation format
        task_spec = TaskObsSpec(request.workspace_config)
        is_valid, error_msg = task_spec.validate_observations(request.observations)

        if not is_valid:
            logger.error(f"Invalid observation format: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid observation format: {error_msg}")

        # Decode observations
        decoded_observations = decode_observations(request.observations)

        # Forward request to backend API
        backend_payload = {
            "request_id": request.request_id,
            "device_id": request.device_id,
            "workspace_config": request.workspace_config,
            "task_config": request.task_config,
            "checkpoint_path": request.checkpoint_path,
            "observations": decoded_observations,
            "debug": request.debug,
        }

        try:
            backend_url = f"{BACKEND_URL}/backend/inference"
            response = requests.post(
                backend_url,
                json=backend_payload,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    return InferenceResponse(success=True, data=result.get("data"))
                else:
                    return InferenceResponse(success=False, error=result.get("error", "Unknown error"))
            else:
                return InferenceResponse(
                    success=False, error=f"Backend service error: {response.status_code} - {response.text}"
                )

        except requests.exceptions.Timeout:
            return InferenceResponse(success=False, error="Backend service timeout")
        except requests.exceptions.ConnectionError as e:
            return InferenceResponse(success=False, error=f"Failed to connect to backend service: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        return InferenceResponse(success=False, error=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenPi DAgger Serve Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8072, help="Port to bind to")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8080", help="Backend service URL")

    args = parser.parse_args()

    BACKEND_URL = args.backend_url
    logger.info(f"Starting DAgger Serve gateway on {args.host}:{args.port}")
    logger.info(f"Backend URL: {BACKEND_URL}")

    uvicorn.run(app, host=args.host, port=args.port)
