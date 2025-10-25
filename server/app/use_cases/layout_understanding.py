"""Use case to run GPT layout understanding after OCR completion."""

import os
import json
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from app.core.logger import log_info, log_error
from app.services.openai_responses_client import OpenAIResponsesClient


class RunLayoutUnderstandingUseCase:
    def __init__(self, client: OpenAIResponsesClient):
        self.client = client
        # Store API key at initialization time to avoid issues with background processes
        self.api_key = client.api_key

    def _save_json(self, path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _get_layout_json_path(self, image_abs_path: str, is_error: bool = False) -> str:
        """Generate layout JSON file path colocated with the image."""
        image_path_obj = Path(image_abs_path)
        directory = image_path_obj.parent
        basename = image_path_obj.stem
        
        if is_error:
            json_filename = f"{basename}.layout.error.json"
        else:
            json_filename = f"{basename}.layout.json"
        
        return str(directory / json_filename)

    def _save_error_json(self, json_path: str, session_id: str, error_msg: str) -> None:
        """Save error information as JSON when GPT layout processing fails."""
        try:
            error_data = {
                "error": "LAYOUT_FAILED",
                "message": error_msg
            }
            self._save_json(json_path, error_data)
            log_info("gpt_layout_error_saved", f"GPT layout error JSON saved: {json_path}", session_id=session_id)
        except Exception as e:
            log_error("gpt_layout_error_save_failed", f"Failed to save GPT layout error JSON: {str(e)}", session_id=session_id)

    def run(self, session_id: str, image_abs_path: str, ocr_json_abs_path: str) -> Dict[str, Any]:
        """
        Run GPT layout understanding on OCR results.
        
        Args:
            session_id: WebSocket session identifier
            image_abs_path: Path to the image file
            ocr_json_abs_path: Path to the OCR JSON file
            
        Returns:
            GPT layout result dictionary only (no file writing)
        """
        try:
            # Check if API key is available
            if not self.api_key:
                error_msg = "OPENAI_API_KEY not set"
                log_error("gpt_layout_skipped", error_msg, session_id=session_id)
                return {"error": "LAYOUT_FAILED", "message": error_msg}
            
            # Validate OCR JSON presence
            if not os.path.exists(ocr_json_abs_path):
                log_error("gpt_layout_failed", f"Skipping layout; OCR missing: {ocr_json_abs_path}", session_id=session_id)
                return {"error": "LAYOUT_SKIPPED_NO_OCR", "message": f"OCR JSON missing: {ocr_json_abs_path}"}

            # Load OCR JSON text
            with open(ocr_json_abs_path, 'r', encoding='utf-8') as f:
                ocr_json_text = f.read()

            # Load and encode image as base64
            import base64
            with open(image_abs_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            log_info("gpt_layout_start", f"Starting layout understanding", session_id=session_id)

            # Call OpenAI Responses API
            # Ensure the client has the API key (in case it was lost in background process)
            self.client.api_key = self.api_key
            raw_text, parsed = self.client.call_layout(session_id=session_id, image_data=image_data, ocr_json_text=ocr_json_text)

            # Ensure schema version field
            parsed.setdefault("schema_version", "noor_layout_v1")

            log_info("gpt_layout_success", f"GPT layout understanding completed successfully", session_id=session_id)
            return parsed

        except Exception as e:
            log_error("gpt_layout_failed", f"Layout understanding failed: {str(e)}", session_id=session_id)
            return {"error": "LAYOUT_FAILED", "message": str(e)}


