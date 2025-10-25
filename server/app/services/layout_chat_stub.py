import asyncio
import uuid
from typing import Dict, Any
from ..domain.ports import ILayoutEngine, IChatEngine


class LayoutEngineStub(ILayoutEngine):
    """Stub implementation of layout engine for testing."""
    
    def __init__(self):
        self.ready = True
    
    async def structure_text(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock structured layout."""
        # Simulate layout processing time
        await asyncio.sleep(0.5)
        
        return {
            "layout_json": {
                "sections": [
                    {
                        "type": "heading",
                        "text": "Sample Document Title",
                        "confidence": 0.98
                    },
                    {
                        "type": "paragraph",
                        "text": "This is a sample paragraph extracted from the document.",
                        "confidence": 0.95
                    }
                ],
                "meta": {
                    "total_sections": 2,
                    "processing_time_ms": 500
                }
            },
            "chat_id": str(uuid.uuid4())
        }
    
    async def is_ready(self) -> bool:
        """Return readiness status."""
        return self.ready


class ChatEngineStub(IChatEngine):
    """Stub implementation of chat engine for testing."""
    
    def __init__(self):
        self.available = True
        self.sessions = {}
    
    async def create_session(self, layout_data: Dict[str, Any]) -> str:
        """Create mock chat session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = layout_data
        return session_id
    
    async def ask_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """Return mock answer to question."""
        # Simulate chat processing time
        await asyncio.sleep(0.3)
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "answer": f"Mock answer for: {question}",
            "citations": [
                {
                    "section": "paragraph",
                    "text": "This is a sample paragraph extracted from the document.",
                    "confidence": 0.95
                }
            ],
            "session_id": session_id
        }
    
    async def is_available(self) -> bool:
        """Return availability status."""
        return self.available
