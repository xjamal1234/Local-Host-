import uuid
from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    """Standardized error codes for NOOR system."""
    
    # WebSocket Errors
    WS_TIMEOUT = "WS_TIMEOUT"
    WS_CONNECTION_LOST = "WS_CONNECTION_LOST"
    WS_INVALID_MESSAGE = "WS_INVALID_MESSAGE"
    
    # System Errors
    MODEL_BUSY = "MODEL_BUSY"
    SERVER_ERROR = "SERVER_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    
    # Input Validation Errors
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"
    LOW_LIGHT = "LOW_LIGHT"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # General Errors
    INVALID_MSG = "INVALID_MSG"
    TIMEOUT = "TIMEOUT"
    
    # YOLO/ML Model Errors
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    CUDA_ERROR = "CUDA_ERROR"


class ErrorMessages:
    """Error message templates."""
    
    ERROR_MESSAGES = {
        ErrorCode.WS_TIMEOUT: "WebSocket connection timed out",
        ErrorCode.WS_CONNECTION_LOST: "WebSocket connection was lost",
        ErrorCode.WS_INVALID_MESSAGE: "Invalid WebSocket message format",
        ErrorCode.MODEL_BUSY: "AI model is currently busy, please try again",
        ErrorCode.SERVER_ERROR: "Internal server error occurred",
        ErrorCode.RATE_LIMIT: "Rate limit exceeded, please slow down",
        ErrorCode.FRAME_TOO_LARGE: "Image frame size exceeds maximum limit",
        ErrorCode.LOW_LIGHT: "Image lighting is too low for processing",
        ErrorCode.INVALID_FORMAT: "Invalid data format provided",
        ErrorCode.INVALID_MSG: "Invalid message format",
        ErrorCode.TIMEOUT: "Request timed out",
        ErrorCode.MODEL_LOAD_ERROR: "Failed to load YOLO model",
        ErrorCode.INFERENCE_ERROR: "YOLO inference failed",
        ErrorCode.CUDA_ERROR: "GPU or CUDA issue",
    }
    
    @classmethod
    def get_message(cls, error_code: ErrorCode) -> str:
        """Get error message for given error code."""
        return cls.ERROR_MESSAGES.get(error_code, "Unknown error occurred")


class NoorError(Exception):
    """Base exception class for NOOR system."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.message = message or ErrorMessages.get_message(error_code)
        self.details = details or {}
        self.ref = str(uuid.uuid4())
        super().__init__(self.message)


class WebSocketError(NoorError):
    """WebSocket specific errors."""
    pass


class ValidationError(NoorError):
    """Input validation errors."""
    pass


class SystemError(NoorError):
    """System and infrastructure errors."""
    pass


class YoloError(NoorError):
    """YOLO model specific errors."""
    pass


def create_error_response(
    error_code: ErrorCode,
    message: Optional[str] = None,
    ref: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response format."""
    return {
        "error": {
            "code": error_code.value,
            "msg": message or ErrorMessages.get_message(error_code),
            "ref": ref or str(uuid.uuid4())
        }
    }


def create_ws_error_response(
    error_code: ErrorCode,
    message: Optional[str] = None,
    ref: Optional[str] = None
) -> Dict[str, Any]:
    """Create WebSocket error response format."""
    return {
        "type": "error",
        "code": error_code.value,
        "msg": message or ErrorMessages.get_message(error_code),
        "ref": ref or str(uuid.uuid4())
    }
