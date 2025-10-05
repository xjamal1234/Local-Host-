from typing import Dict, Any
from ..core.di import container
from ..core.logger import log_info, log_debug


class GuidedCaptureUseCase:
    """Use case for managing guided capture orchestration."""
    
    def __init__(self):
        self.guidance_engine = container.get_guidance_engine()
    
    async def start_guidance_session(self, session_id: str) -> Dict[str, Any]:
        """Start a new guided capture session."""
        log_info("guided_capture", f"Starting guidance session", session_id=session_id)
        
        # Check if guidance engine is ready
        is_ready = await self.guidance_engine.is_ready()
        
        return {
            "session_id": session_id,
            "guidance_ready": is_ready,
            "status": "active" if is_ready else "waiting"
        }
    
    async def process_frame_guidance(self, session_id: str, frame_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process frame and return guidance response."""
        log_debug("guided_capture", f"Processing frame for guidance", session_id=session_id)
        
        # Get guidance from engine
        guidance_response = await self.guidance_engine.analyze_frame(frame_data, metadata)
        
        return {
            "session_id": session_id,
            "guidance": guidance_response,
            "frame_processed": True
        }
    
    async def finalize_capture(self, session_id: str, best_frame_seq: int = None) -> Dict[str, Any]:
        """Finalize the capture process."""
        log_info("guided_capture", f"Finalizing capture", session_id=session_id)
        
        return {
            "session_id": session_id,
            "best_frame_seq": best_frame_seq,
            "capture_completed": True,
            "ready_for_ocr": True
        }
