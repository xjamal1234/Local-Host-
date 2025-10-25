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
WS_IDLE_TIMEOUT_SEC = int(os.getenv("NOOR_WS_IDLE_TIMEOUT_SEC", "120"))  # Single source of truth for idle timeout
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

# OCR Configuration
OCR_ENGINE = os.getenv("NOOR_OCR_ENGINE", "easyocr")
OCR_LANGS = os.getenv("NOOR_OCR_LANGS", "en,ar")
OCR_OUTPUT_DIR = os.getenv("NOOR_OCR_OUTPUT_DIR", "server/app/static/final_captures")
OCR_SAVE_JSON = os.getenv("NOOR_OCR_SAVE_JSON", "true").lower() in ["true", "1", "yes"]

# OpenAI Layout (Phase 6) Configuration
# Load API key from hardcoded secret_config.py file
try:
    from .secret_config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = ""
    # If secret_config.py is missing, log error but don't crash
    try:
        from .logger import log_error
        log_error("secret_config_missing", "secret_config.py not found; OpenAI features will be disabled")
    except Exception:
        pass
OPENAI_MODEL = os.getenv("NOOR_OPENAI_MODEL", "gpt-4o-2024-08-06")  # Responses-capable model with Structured Outputs
OPENAI_API_BASE = os.getenv("NOOR_OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_RESPONSES_ENDPOINT = os.getenv("NOOR_OPENAI_RESPONSES_ENDPOINT", "/responses")
OPENAI_TIMEOUT_SEC = int(os.getenv("NOOR_OPENAI_TIMEOUT_SEC", "45"))
OPENAI_MAX_RETRIES = int(os.getenv("NOOR_OPENAI_MAX_RETRIES", "2"))

# Chat Intent Extraction Configuration (reuses OpenAI config above)
CHAT_INTENT_TIMEOUT_SEC = int(os.getenv("NOOR_CHAT_INTENT_TIMEOUT_SEC", "30"))
CHAT_INTENT_MODEL = os.getenv("NOOR_CHAT_INTENT_MODEL", OPENAI_MODEL)  # Default to same model as layout

# Chat Security Configuration
NOOR_CHAT_TOKEN = os.getenv("NOOR_CHAT_TOKEN", "")  # If set, requires authentication for chat
CHAT_RATE_LIMIT_MESSAGES = int(os.getenv("NOOR_CHAT_RATE_LIMIT_MESSAGES", "5"))  # Messages per window
CHAT_RATE_LIMIT_WINDOW_SEC = int(os.getenv("NOOR_CHAT_RATE_LIMIT_WINDOW_SEC", "10"))  # Window in seconds

# Where final frames + OCR + layout JSON are stored
FINAL_FRAME_DIR = os.getenv("NOOR_FINAL_FRAME_DIR", "server/app/static/final_captures")

if not OPENAI_API_KEY:
    # Soft warning; do not crash on startup
    try:
        from .logger import log_error
        log_error("openai_api_key_missing", "OPENAI_API_KEY is empty; Phase 6 layout will be disabled until set")
    except Exception:
        pass
