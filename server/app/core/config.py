import os


def check_cuda_availability():
    """Check if CUDA is available on the system."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # If torch is not available, check for nvidia-smi as fallback
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False


# Server Configuration
PORT = int(os.getenv("PORT", "8080"))
WS_PATH = os.getenv("WS_PATH", "/ws/guidance")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
ENV = os.getenv("ENV", "dev")

# GPU Configuration
USE_CUDA = os.getenv("USE_CUDA", "True") == "True" and check_cuda_availability()

# WebSocket Configuration
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "3"))
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "10"))
MAX_FPS = int(os.getenv("MAX_FPS", "24"))
MAX_JPEG_KB = int(os.getenv("MAX_JPEG_KB", "120"))
QUEUE_MAX = int(os.getenv("QUEUE_MAX", "10"))
CADENCE_MS = int(os.getenv("CADENCE_MS", "250"))

# Development Configuration  
USE_STUBS = os.getenv("USE_STUBS", "False") == "True"  # Default to real implementations for Phase 3

# Application Metadata
APP_VERSION = os.getenv("APP_VERSION", "1.0")
BUILD_HASH = os.getenv("BUILD_HASH", "dummy-hash")

# Final Capture Configuration
FINAL_CAPTURE_CLASS = int(os.getenv("NOOR_FINAL_CAPTURE_CLASS", "5"))
FINAL_CAPTURE_MIN_COUNT = int(os.getenv("FINAL_CAPTURE_MIN_COUNT", "3"))
FINAL_CAPTURE_MIN_FREQ = float(os.getenv("FINAL_CAPTURE_MIN_FREQ", "0.60"))
FINAL_FRAME_DIR = os.getenv("FINAL_FRAME_DIR", "server/app/static/final_captures")
STOP_BEHAVIOR = os.getenv("STOP_BEHAVIOR", "server_close")  # or "wait_ack"
ACK_TIMEOUT_MS = int(os.getenv("ACK_TIMEOUT_MS", "1500"))

# YOLO Evaluation Logging Configuration
ENABLE_YOLO_EVAL_LOGS = os.getenv("NOOR_ENABLE_YOLO_EVAL_LOGS", "true").lower() in ["true", "1", "yes"]
YOLO_EVAL_TOPK = max(1, min(8, int(os.getenv("NOOR_YOLO_EVAL_TOPK", "3"))))  # Clamp to [1, 8]

# Guidance Vote Mode Configuration
GUIDANCE_VOTE_MODE = os.getenv("NOOR_GUIDANCE_VOTE_MODE", "ema").lower()
if GUIDANCE_VOTE_MODE not in ("ema", "majority"):
    GUIDANCE_VOTE_MODE = "ema"
