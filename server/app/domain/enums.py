from enum import Enum


class GuidanceDirection(Enum):
    """Guidance directions for camera positioning."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ROTATE_CW = "rotateCW"
    ROTATE_CCW = "rotateCCW"
    CLOSER = "closer"
    FARTHER = "farther"
    STEADY = "steady"
    # Semantic directions for YOLO guidance (0..6)
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    PAPER_FACE_ONLY = "paper_face_only"
    PERFECT = "perfect"
    NO_DOCUMENT = "no_document"
    
    # Class-based directions for YOLO temporal aggregation (legacy)
    CLASS_0 = "class_0"  # top-left corner seen
    CLASS_1 = "class_1"  # top-right corner seen
    CLASS_2 = "class_2"  # bottom-left corner seen
    CLASS_3 = "class_3"  # bottom-right corner seen
    CLASS_4 = "class_4"  # partial framing
    CLASS_5 = "class_5"  # perfect framing
    CLASS_6 = "class_6"  # center-edges
    CLASS_7 = "class_7"  # no document detected


class MessageType(Enum):
    """WebSocket message types."""
    FRAME_META = "frame_meta"
    CAPTURE = "capture"
    CANCEL = "cancel"
    GUIDANCE = "guidance"
    HEARTBEAT = "hb"
    OCR_PROGRESS = "ocr_progress"
    OCR_DONE = "ocr_done"
    ERROR = "error"


class CaptureReason(Enum):
    """Reasons for capture trigger."""
    AUTO = "auto"
    MANUAL = "manual"


class LoadHint(Enum):
    """Server load indicators."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Language(Enum):
    """Supported languages."""
    ARABIC = "ar"
    ENGLISH = "en"
