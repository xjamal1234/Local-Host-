"""
REST API endpoint for chat functionality.

POST /api/v1/chat - Send chat message and get response
"""

from fastapi import APIRouter, HTTPException, status, Header
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from app.core.logger import log_info, log_error
from app.core.di import container
from app.core.config import NOOR_CHAT_TOKEN
from app.core.chat.help_text import ARABIC_HELP_TEXT, AUTH_FAILED_MESSAGE


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""
    docId: str = Field(..., description="4-digit document ID (e.g., '0007')")
    text: str = Field(..., description="User query text")
    locale: Optional[str] = Field("ar", description="Locale hint (default: 'ar')")


class ChatResponse(BaseModel):
    """Chat response model."""
    text: str = Field(..., description="Arabic text response")
    meta: Dict[str, Any] = Field(..., description="Metadata including intent and args")


# Create router
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
) -> ChatResponse:
    """
    Process a chat message and return a response.
    
    This endpoint extracts the user's intent using OpenAI Responses API with
    Structured Outputs, then executes the intent deterministically against the
    GPT JSON file for the specified document.
    
    Requires authentication via Bearer token if NOOR_CHAT_TOKEN is set.
    
    Args:
        request: Chat request with docId and text
        authorization: Optional Bearer token for authentication
        
    Returns:
        ChatResponse with Arabic text and metadata
        
    Raises:
        HTTPException: If authentication fails, document not found, or processing fails
    """
    # Authentication check if token is required
    if NOOR_CHAT_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            log_error("chat_auth_failed", "Missing or invalid Authorization header", request_id=request.docId)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=AUTH_FAILED_MESSAGE
            )
        
        token = authorization.replace("Bearer ", "").strip()
        if token != NOOR_CHAT_TOKEN:
            log_error("chat_auth_failed", "Invalid authentication token", request_id=request.docId)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=AUTH_FAILED_MESSAGE
            )
    
    try:
        # Get chat services
        intents_client = container.get_intents_client()
        json_executor = container.get_json_executor()
        
        log_info("chat_api_request", f"Chat request for docId: {request.docId}", request_id=request.docId)
        
        # Try GPT intent extraction first
        intent_result = None
        fallback_used = False
        
        try:
            intent_result = await intents_client.extract_intent(
                request.text,
                locale=request.locale or "ar",
                session_id=request.docId  # Use docId as session_id for logging
            )
            intent = intent_result.get("intent")
            args = intent_result.get("args", {})
            
            # If GPT returns unsupported, try fallback
            if intent == "unsupported":
                from app.core.chat.fallback_intents import detect_intent_fallback
                fallback_result = detect_intent_fallback(request.text, reason="gpt_unsupported")
                intent = fallback_result.get("intent")
                args = fallback_result.get("args", {})
                fallback_used = True
                log_info("intent_fallback_used", f"Fallback used for unsupported GPT result", request_id=request.docId)
                
        except Exception as gpt_error:
            # GPT failed, use fallback
            from app.core.chat.fallback_intents import detect_intent_fallback
            fallback_result = detect_intent_fallback(request.text, reason="gpt_error")
            intent = fallback_result.get("intent")
            args = fallback_result.get("args", {})
            fallback_used = True
            log_info("intent_fallback_used", f"Fallback used due to GPT error: {str(gpt_error)}", request_id=request.docId)
        
        log_info("chat_intent_extracted", f"Intent: {intent}, Args: {args}, Fallback: {fallback_used}", request_id=request.docId)
        
        # If intent is still unsupported after fallback, use central help text
        if intent == "unsupported":
            response_text = ARABIC_HELP_TEXT
        else:
            # Execute intent locally on gpt_####.json
            response_text = json_executor.execute(intent, args, request.docId, session_id=request.docId)
        
        log_info("chat_execution_done", f"Execution complete for intent: {intent}", request_id=request.docId)
        
        # Build response
        response = ChatResponse(
            text=response_text,
            meta={
                "intent": intent,
                "args": args
            }
        )
        
        log_info("chat_reply_sent", f"Chat reply sent via API", request_id=request.docId)
        
        return response
        
    except FileNotFoundError as e:
        log_error("chat_api_file_not_found", f"Document not found: {str(e)}", request_id=request.docId)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستند {request.docId} غير موجود. الرجاء التأكد من رقم المستند."
        )
        
    except Exception as e:
        log_error("chat_api_error", f"Chat API error: {str(e)}", request_id=request.docId)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="حدث خطأ أثناء معالجة طلبك. الرجاء المحاولة مرة أخرى."
        )

