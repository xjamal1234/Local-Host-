from typing import Dict, Any
from ..core.di import container
from ..core.logger import log_info, log_debug


class OcrProcessingUseCase:
    """Use case for handling OCR image processing."""
    
    def __init__(self):
        self.ocr_engine = container.get_ocr_engine()
    
    async def process_captured_image(self, image_data: bytes, session_id: str = None) -> Dict[str, Any]:
        """Process captured image with OCR."""
        log_info("ocr_processing", f"Starting OCR processing", session_id=session_id)
        
        # Check if OCR engine is available
        is_available = await self.ocr_engine.is_available()
        if not is_available:
            raise RuntimeError("OCR engine is not available")
        
        # Process image
        ocr_result = await self.ocr_engine.process_image(image_data)
        
        log_info("ocr_processing", f"OCR processing completed", session_id=session_id)
        
        return {
            "session_id": session_id,
            "ocr_result": ocr_result,
            "processing_completed": True,
            "ready_for_layout": True
        }
    
    async def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """Get current OCR processing status."""
        log_debug("ocr_processing", f"Checking OCR status", session_id=session_id)
        
        # In a real implementation, this would track actual processing status
        return {
            "session_id": session_id,
            "status": "completed",  # Mock status
            "progress_pct": 100
        }
