from typing import Dict, Any
from ..core.di import container
from ..core.logger import log_info, log_debug


class GptChatUseCase:
    """Use case for handling GPT-powered chat interactions."""
    
    def __init__(self):
        self.chat_engine = container.get_chat_engine()
    
    async def create_chat_session(self, layout_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Create a new chat session with document context."""
        log_info("gpt_chat", f"Creating chat session", session_id=session_id)
        
        # Check if chat engine is available
        is_available = await self.chat_engine.is_available()
        if not is_available:
            raise RuntimeError("Chat engine is not available")
        
        # Create chat session
        chat_session_id = await self.chat_engine.create_session(layout_data)
        
        log_info("gpt_chat", f"Chat session created: {chat_session_id}", session_id=session_id)
        
        return {
            "session_id": session_id,
            "chat_session_id": chat_session_id,
            "document_loaded": True,
            "ready_for_questions": True
        }
    
    async def ask_question(self, chat_session_id: str, question: str, session_id: str = None) -> Dict[str, Any]:
        """Ask a question about the document."""
        log_debug("gpt_chat", f"Processing question: {question[:50]}...", session_id=session_id)
        
        # Get answer from chat engine
        answer_result = await self.chat_engine.ask_question(chat_session_id, question)
        
        log_debug("gpt_chat", f"Question answered", session_id=session_id)
        
        return {
            "session_id": session_id,
            "chat_session_id": chat_session_id,
            "question": question,
            "answer_result": answer_result,
            "question_processed": True
        }
    
    async def get_chat_history(self, chat_session_id: str, session_id: str = None) -> Dict[str, Any]:
        """Get chat history for a session."""
        log_debug("gpt_chat", f"Retrieving chat history", session_id=session_id)
        
        # Mock chat history
        return {
            "session_id": session_id,
            "chat_session_id": chat_session_id,
            "history": [
                {
                    "question": "What is this document about?",
                    "answer": "This appears to be a sample document with structured content.",
                    "timestamp": "2025-09-15T20:40:00Z"
                }
            ],
            "total_questions": 1
        }
