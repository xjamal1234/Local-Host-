from dataclasses import dataclass
from typing import Optional, Dict, Any
from .enums import GuidanceDirection, MessageType, CaptureReason, LoadHint, Language


@dataclass
class FrameMetadata:
    """Metadata for incoming video frames."""
    type: MessageType
    seq: int
    ts: int  # timestamp in milliseconds
    w: int   # width
    h: int   # height
    rotation_degrees: int
    jpeg_quality: int


@dataclass
class CaptureRequest:
    """Request to capture the current frame."""
    type: MessageType
    reason: CaptureReason
    best_seq: Optional[int] = None


@dataclass
class GuidanceResponse:
    """Guidance response for camera positioning."""
    type: MessageType
    dir: GuidanceDirection
    magnitude: float  # 0.0 to 1.0
    coverage: float   # 0.0 to 1.0
    skew_deg: float
    conf: float       # confidence 0.0 to 1.0
    ready: bool


@dataclass
class HeartbeatResponse:
    """WebSocket heartbeat response."""
    type: MessageType
    rtt_ms: int
    load_hint: LoadHint
    last_seq: int


@dataclass
class OcrProgressResponse:
    """OCR processing progress update."""
    type: MessageType
    stage: str
    pct: int  # 0 to 100


@dataclass
class OcrDoneResponse:
    """OCR processing completion notification."""
    type: MessageType
    doc_id: str
    pages: int
    lang: Language
    handoff: str
    meta: Dict[str, Any]


@dataclass
class ErrorResponse:
    """Error response format."""
    type: MessageType
    code: str
    msg: str
    ref: Optional[str] = None


@dataclass
class HealthResponse:
    """Health check response."""
    status: str


@dataclass
class ReadinessResponse:
    """Readiness check response."""
    ready: bool


@dataclass
class VersionResponse:
    """Version information response."""
    version: str
    build: str
