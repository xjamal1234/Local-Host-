import asyncio
import uuid
from typing import Dict, Any
from ..domain.ports import IOcrEngine
from ..domain.dtos import OcrDoneResponse
from ..domain.enums import MessageType, Language


class OcrEngineStub(IOcrEngine):
    """Stub implementation of OCR engine for testing."""
    
    def __init__(self):
        self.available = True
    
    async def process_image(self, image_data: bytes) -> OcrDoneResponse:
        """Return mock OCR processing result."""
        # Simulate OCR processing time
        await asyncio.sleep(1.0)
        
        return OcrDoneResponse(
            type=MessageType.OCR_DONE,
            doc_id=str(uuid.uuid4()),
            pages=1,
            lang=Language.ENGLISH,
            handoff="internal",
            meta={
                "w": 800,
                "h": 600,
                "text_blocks": 5,
                "confidence": 0.95
            }
        )
    
    async def is_available(self) -> bool:
        """Return availability status."""
        return self.available
