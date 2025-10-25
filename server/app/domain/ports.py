from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .dtos import GuidanceResponse, OcrDoneResponse


class IGuidanceEngine(ABC):
    """Interface for guidance engine implementations."""
    
    @abstractmethod
    async def analyze_frame(self, frame_data: bytes, metadata: Dict[str, Any]) -> GuidanceResponse:
        """Analyze frame and return guidance response."""
        pass
    
    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if guidance engine is ready to process frames."""
        pass


class IOcrEngine(ABC):
    """Interface for OCR engine implementations."""
    
    @abstractmethod
    async def process_image(self, image_data: bytes) -> OcrDoneResponse:
        """Process image and extract text with OCR."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if OCR engine is available."""
        pass


class ILayoutEngine(ABC):
    """Interface for layout processing engines."""
    
    @abstractmethod
    async def structure_text(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure OCR text into organized layout."""
        pass
    
    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if layout engine is ready."""
        pass


class IChatEngine(ABC):
    """Interface for chat/conversation engines."""
    
    @abstractmethod
    async def create_session(self, layout_data: Dict[str, Any]) -> str:
        """Create a new chat session with document context."""
        pass
    
    @abstractmethod
    async def ask_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """Ask a question about the document."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if chat engine is available."""
        pass
