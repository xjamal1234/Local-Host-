import asyncio
import base64
import json
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from ...core.config import MAX_JPEG_KB
from ...core.errors import ErrorCode, create_ws_error_response, NoorError
from ...core.di import container
from ...core.logger import log_info, log_error, log_debug
from ...domain.dtos import FrameMetadata, GuidanceResponse
from ...domain.enums import MessageType

router = APIRouter(tags=["guidance"])


def normalize_frame_meta_keys(frame_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize camelCase keys to snake_case for FrameMetadata DTO."""
    normalized = {}
    key_mapping = {
        "rotationDegrees": "rotation_degrees",
        "jpegQuality": "jpeg_quality",
        "timestamp": "ts",
        "width": "w", 
        "height": "h"
    }
    
    for key, value in frame_meta.items():
        normalized_key = key_mapping.get(key, key)
        normalized[normalized_key] = value
    
    return normalized


def validate_jpeg_size(image_bytes: bytes) -> None:
    """Validate JPEG size against MAX_JPEG_KB limit."""
    size_kb = len(image_bytes) / 1024
    if size_kb > MAX_JPEG_KB:
        raise NoorError(
            ErrorCode.FRAME_TOO_LARGE,
            f"Image size {size_kb:.1f}KB exceeds limit of {MAX_JPEG_KB}KB"
        )


def decode_jpeg_image(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to OpenCV BGR array."""
    try:
        # Decode JPEG to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise NoorError(ErrorCode.INVALID_FORMAT, "Invalid JPEG format")
        
        return image
    except Exception as e:
        raise NoorError(ErrorCode.INVALID_FORMAT, f"Failed to decode JPEG: {str(e)}")


def create_frame_metadata(frame_meta_dict: Optional[Dict[str, Any]], seq: int = 1) -> FrameMetadata:
    """Create FrameMetadata DTO from dictionary."""
    if frame_meta_dict:
        normalized = normalize_frame_meta_keys(frame_meta_dict)
        return FrameMetadata(
            type=MessageType.FRAME_META,
            seq=normalized.get("seq", seq),
            ts=normalized.get("ts", int(asyncio.get_event_loop().time() * 1000)),
            w=normalized.get("w", 640),
            h=normalized.get("h", 480),
            rotation_degrees=normalized.get("rotation_degrees", 0),
            jpeg_quality=normalized.get("jpeg_quality", 85)
        )
    else:
        return FrameMetadata(
            type=MessageType.FRAME_META,
            seq=seq,
            ts=int(asyncio.get_event_loop().time() * 1000),
            w=640,
            h=480,
            rotation_degrees=0,
            jpeg_quality=85
        )


async def run_inference_in_executor(image_bytes: bytes, frame_meta: FrameMetadata, expected_class: Optional[int] = None) -> GuidanceResponse:
    """Run YOLO inference in thread executor to avoid blocking event loop."""
    def _inference_worker():
        try:
            # Decode image
            image = decode_jpeg_image(image_bytes)
            
            # Get guidance engine from DI container
            guidance_engine = container.get_guidance_engine()
            
            # Convert frame metadata to dict format expected by engine
            metadata_dict = {
                "seq": frame_meta.seq,
                "ts": frame_meta.ts,
                "w": frame_meta.w,
                "h": frame_meta.h,
                "rotation_degrees": frame_meta.rotation_degrees,
                "jpeg_quality": frame_meta.jpeg_quality
            }
            
            # Run inference (this will be async in the main thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(guidance_engine.analyze_frame(image_bytes, metadata_dict, expected_class))
            finally:
                loop.close()
                
        except Exception as e:
            log_error("inference_worker", f"Inference failed: {str(e)}")
            raise NoorError(ErrorCode.INFERENCE_ERROR, f"Inference failed: {str(e)}")
    
    # Run in thread executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _inference_worker)


@router.post("/analyze")
async def analyze_frame(
    request: Request,
    image: Optional[UploadFile] = File(None),
    frame_meta: Optional[str] = Form(None),
    expected: Optional[int] = Form(None)
):
    """
    Analyze a single JPEG frame and return guidance response.
    
    Supports two input modes:
    1. multipart/form-data: image file + optional frame_meta JSON string
    2. application/json: image_b64 + optional frame_meta object
    """
    request_id = str(uuid.uuid4())
    log_info("guidance_analyze", f"Request received: {request_id}")
    
    try:
        # Determine content type and extract image data
        content_type = request.headers.get("content-type", "")
        image_bytes = None
        frame_meta_dict = None
        expected_class = None
        
        if "multipart/form-data" in content_type:
            # Handle multipart form data
            if not image:
                raise NoorError(ErrorCode.INVALID_MSG, "Missing required 'image' field")
            
            if image.content_type not in ["image/jpeg", "image/jpg"]:
                raise NoorError(ErrorCode.INVALID_FORMAT, "Only JPEG images are supported")
            
            image_bytes = await image.read()
            
            # Parse frame_meta if provided
            if frame_meta:
                try:
                    frame_meta_dict = json.loads(frame_meta)
                except json.JSONDecodeError:
                    raise NoorError(ErrorCode.INVALID_MSG, "Invalid frame_meta JSON format")
        
        elif "application/json" in content_type:
            # Handle JSON with base64 image
            try:
                body = await request.json()
            except Exception:
                raise NoorError(ErrorCode.INVALID_MSG, "Invalid JSON payload")
            
            if "image_b64" not in body:
                raise NoorError(ErrorCode.INVALID_MSG, "Missing required 'image_b64' field")
            
            try:
                image_bytes = base64.b64decode(body["image_b64"])
            except Exception:
                raise NoorError(ErrorCode.INVALID_FORMAT, "Invalid base64 image data")
            
            frame_meta_dict = body.get("frame_meta")
            expected_class = body.get("expected")
        
        else:
            raise NoorError(ErrorCode.INVALID_FORMAT, "Unsupported content type. Use multipart/form-data or application/json")
        
        # Validate image size
        validate_jpeg_size(image_bytes)
        
        # JPEG sanity check - attempt to decode the frame
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded_image is None:
                # Invalid JPEG - return HTTP 400
                log_info("rest_frame_decode_failed", "Invalid JPEG image received", request_id=request_id)
                raise NoorError(ErrorCode.INVALID_MSG, "invalid JPEG image")
        except Exception as e:
            if isinstance(e, NoorError):
                raise  # Re-raise NoorError as-is
            # Decode error - return HTTP 400
            log_info("rest_frame_decode_failed", f"JPEG decode error: {str(e)}", request_id=request_id)
            raise NoorError(ErrorCode.INVALID_MSG, "invalid JPEG image")
        
        log_info("guidance_analyze", f"Image decoded successfully: {len(image_bytes)} bytes")
        
        # Create frame metadata
        frame_metadata = create_frame_metadata(frame_meta_dict)
        log_info("guidance_analyze", f"Frame metadata: seq={frame_metadata.seq}, size={frame_metadata.w}x{frame_metadata.h}")
        
        # Run inference in executor
        log_info("guidance_analyze", "Starting inference...")
        guidance_response = await run_inference_in_executor(image_bytes, frame_metadata, expected_class)
        log_info("guidance_analyze", f"Inference completed: {guidance_response.dir.value}")
        
        # Return guidance response in WebSocket format
        response_data = {
            "type": "guidance",
            "class": guidance_response.dir.value,  # Changed from "dir" to "class"
            "coverage": guidance_response.coverage,
            "skew_deg": guidance_response.skew_deg,
            "conf": guidance_response.conf,
            "ready": guidance_response.ready
        }
        
        log_info("guidance_analyze", f"Guidance returned: {response_data}")
        return JSONResponse(content=response_data, status_code=200)
        
    except NoorError as e:
        log_error("guidance_analyze", f"NoorError: {e.error_code.value} - {e.message}")
        
        # Map error codes to HTTP status codes
        status_code = 500
        if e.error_code == ErrorCode.FRAME_TOO_LARGE:
            status_code = 413
        elif e.error_code in [ErrorCode.INVALID_MSG, ErrorCode.INVALID_FORMAT]:
            status_code = 422
        elif e.error_code in [ErrorCode.MODEL_LOAD_ERROR, ErrorCode.INFERENCE_ERROR, ErrorCode.CUDA_ERROR]:
            status_code = 500
        
        error_response = create_ws_error_response(e.error_code, e.message, e.ref)
        return JSONResponse(content=error_response, status_code=status_code)
        
    except Exception as e:
        log_error("guidance_analyze", f"Unexpected error: {str(e)}")
        error_response = create_ws_error_response(ErrorCode.SERVER_ERROR, f"Internal server error: {str(e)}")
        return JSONResponse(content=error_response, status_code=500)
