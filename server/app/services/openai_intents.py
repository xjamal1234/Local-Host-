"""
OpenAI intent extraction using Responses API with Structured Outputs.

This module extracts user intents from natural language queries using OpenAI's
Responses API with strict JSON Schema conformance (Structured Outputs).
"""

import json
import httpx
from typing import Dict, Any, Optional

from app.core.logger import log_info, log_error, log_debug
from app.core.chat.intents_schema import INTENT_EXTRACTION_SCHEMA, IntentExtraction
from app.core.chat.prompt_templates import build_few_shot_messages


class OpenAIIntentsClient:
    """Client for extracting intents using OpenAI Responses API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-2024-08-06",  # Responses-capable model with Structured Outputs
        base_url: str = "https://api.openai.com/v1",
        responses_endpoint: str = "/responses",
        timeout_sec: int = 30,
        max_retries: int = 2
    ):
        """
        Initialize OpenAI intents client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (must support Structured Outputs)
            base_url: API base URL
            responses_endpoint: Responses API endpoint (default: /responses)
            timeout_sec: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.responses_endpoint = responses_endpoint
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        
        if not self.api_key:
            log_error("openai_intents_no_key", "OpenAI API key not provided for intent extraction")
    
    async def extract_intent(
        self,
        text: str,
        locale: str = "ar",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract intent from user text using OpenAI Responses API with Structured Outputs.
        
        Args:
            text: User query text
            locale: Locale hint (default: "ar")
            session_id: Optional session ID for logging
            
        Returns:
            Dictionary with 'intent' and 'args' keys, or unsupported intent on error
        """
        if not self.api_key:
            log_error("openai_intents_no_key", "Cannot extract intent without API key", session_id=session_id)
            return {
                "intent": "unsupported",
                "args": {"original_query": text}
            }
        
        # Build few-shot messages
        messages = build_few_shot_messages(text, locale)
        
        # Prepare request payload for Chat Completions API with response_format
        # Switch back to Chat Completions API as Responses API has compatibility issues
        payload = {
            "model": self.model,
            "messages": messages,  # Chat Completions API uses 'messages'
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_extraction",
                    "schema": INTENT_EXTRACTION_SCHEMA,
                    "strict": True  # Enforce strict schema conformance
                }
            },
            "temperature": 0.0,  # Deterministic output
            "max_tokens": 200  # Chat Completions API uses 'max_tokens'
        }
        
        log_debug("intent_extraction_start", f"Extracting intent from: {text[:100]}...", session_id=session_id)
        
        # Call OpenAI Chat Completions API with Structured Outputs
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract intent from Chat Completions API response
                    # Chat Completions API structure: { "choices": [{"message": {"content": "..."}}] }
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            if content:
                                intent_data = json.loads(content)
                            else:
                                raise ValueError("Empty content in response")
                        else:
                            raise ValueError("No message content in choice")
                    else:
                        raise ValueError("No choices in response")
                    
                    # Validate against schema using Pydantic
                    validated = IntentExtraction(**intent_data)
                    
                    log_info(
                        "chat_intent_extracted",
                        f"Intent: {validated.intent}, Model: {self.model}",
                        session_id=session_id
                    )
                    
                    return validated.model_dump()
                        
            except httpx.TimeoutException as e:
                log_error(
                    "intent_extraction_timeout",
                    f"Timeout on attempt {attempt}/{self.max_retries}: {str(e)}",
                    session_id=session_id
                )
                if attempt == self.max_retries:
                    break
                    
            except httpx.HTTPStatusError as e:
                # Enhanced error logging with redacted API key
                body = e.response.text[:300] if e.response.text else "No response body"
                log_error(
                    "intent_extraction_http_error",
                    f"HTTP {e.response.status_code} model={self.model} body_preview={body}",
                    session_id=session_id
                )
                if attempt == self.max_retries:
                    break
                    
            except (json.JSONDecodeError, ValueError) as e:
                log_error(
                    "intent_extraction_parse_error",
                    f"Failed to parse intent on attempt {attempt}/{self.max_retries}: {str(e)}",
                    session_id=session_id
                )
                if attempt == self.max_retries:
                    break
                    
            except Exception as e:
                log_error(
                    "intent_extraction_error",
                    f"Unexpected error on attempt {attempt}/{self.max_retries}: {str(e)}",
                    session_id=session_id
                )
                if attempt == self.max_retries:
                    break
        
        # If all retries fail, return unsupported
        log_error("intent_extraction_failed", "All attempts failed, returning unsupported", session_id=session_id)
        return {
            "intent": "unsupported",
            "args": {"original_query": text}
        }

