from typing import Dict, Any
from ..core.di import container
from ..core.logger import log_info, log_debug


class GptLayoutUseCase:
    """Use case for managing text layout and structuring with GPT."""
    
    def __init__(self):
        self.layout_engine = container.get_layout_engine()
    
    async def structure_ocr_text(self, ocr_result: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Structure OCR text into organized layout using GPT."""
        log_info("gpt_layout", f"Starting layout structuring", session_id=session_id)
        
        # Check if layout engine is ready
        is_ready = await self.layout_engine.is_ready()
        if not is_ready:
            raise RuntimeError("Layout engine is not ready")
        
        # Structure the text
        structured_layout = await self.layout_engine.structure_text(ocr_result)
        
        log_info("gpt_layout", f"Layout structuring completed", session_id=session_id)
        
        return {
            "session_id": session_id,
            "layout_result": structured_layout,
            "structuring_completed": True,
            "ready_for_chat": True
        }
    
    async def get_layout_summary(self, layout_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Get summary of the structured layout."""
        log_debug("gpt_layout", f"Generating layout summary", session_id=session_id)
        
        # Mock summary generation
        sections = layout_data.get("layout_json", {}).get("sections", [])
        
        return {
            "session_id": session_id,
            "summary": {
                "total_sections": len(sections),
                "section_types": [section.get("type") for section in sections],
                "estimated_reading_time": f"{len(sections) * 30} seconds"
            }
        }
