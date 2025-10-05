import sys
import json
from loguru import logger
from .config import ENV, LOG_LEVEL


def setup_logger():
    """Configure loguru logger based on environment."""
    # Remove default handler
    logger.remove()
    
    if ENV == "prod":
        # Production: JSON structured logs
        def json_formatter(record):
            log_entry = {
                "ts": record["time"].isoformat(),
                "level": record["level"].name,
                "event": record["name"],
                "msg": record["message"],
                "requestId": record["extra"].get("requestId"),
                "sessionId": record["extra"].get("sessionId"),
            }
            
            # Add error details if present
            if record["exception"]:
                log_entry["error"] = {
                    "code": record["extra"].get("error_code", "UNKNOWN_ERROR"),
                    "msg": str(record["exception"]),
                    "ref": record["extra"].get("error_ref")
                }
            
            return json.dumps(log_entry)
        
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            format=json_formatter,
            level=LOG_LEVEL,
            rotation="1 MB",
            retention="30 days",
            compression="zip"
        )
    else:
        # Development: Pretty human-readable logs
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=LOG_LEVEL,
            colorize=True
        )
    
    return logger


def log_error(error_code: str, message: str, error_ref: str = None, request_id: str = None, session_id: str = None):
    """Log an error with standardized format."""
    logger.bind(
        error_code=error_code,
        error_ref=error_ref,
        requestId=request_id,
        sessionId=session_id
    ).error(message)


def log_info(event: str, message: str, request_id: str = None, session_id: str = None):
    """Log an info message with context."""
    logger.bind(
        requestId=request_id,
        sessionId=session_id
    ).info(f"{event}: {message}")


def log_debug(event: str, message: str, request_id: str = None, session_id: str = None):
    """Log a debug message with context."""
    logger.bind(
        requestId=request_id,
        sessionId=session_id
    ).debug(f"{event}: {message}")


# Initialize logger on import
app_logger = setup_logger()
