"""
DAgger serve gateway for OpenPi.

This service provides the external API gateway that validates requests and forwards
them to the backend inference service. The API is compatible with umi_base.

Author: Wendi Chen
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openpi.training import config as _config

sys.path.insert(0, str(Path(__file__).parent))
from dagger_utils import TaskObsSpec, deserialize_observations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenPi DAgger Inference Server", version="1.0.0")


BACKEND_URL = None


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

        # Deserialize observations (convert base64 images and lists to numpy arrays)
        deserialized_observations = deserialize_observations(request.observations)
        
        # Create task spec for processing and validation
        task_spec = TaskObsSpec(request.workspace_config)
        
        # Process, reshape and validate observations in one step
        converted_observations, is_valid, error_msg = task_spec.process_observations(
            deserialized_observations, 
            task=request.task_config
        )

        if not is_valid:
            logger.error(f"Invalid observation format: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid observation format: {error_msg}")

        # Forward request to backend API with raw observations (backend will handle conversion again)
        backend_payload = {
            "request_id": request.request_id,
            "device_id": request.device_id,
            "workspace_config": request.workspace_config,
            "task_config": request.task_config,
            "checkpoint_path": request.checkpoint_path,
            "observations": request.observations,
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
                return InferenceResponse(success=True, data=response.json())
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
