"""OpenAI Responses API client for NOOR Phase 6 (Layout Understanding)."""

from typing import Tuple, Dict, Any, Optional
import json
import time
import requests

from app.core.logger import log_info, log_error, log_debug
from app.core import config
from app.services.schemas.noor_layout_schema import NOOR_LAYOUT_V1


class OpenAIResponsesClient:
    def __init__(self, api_key: str, base_url: str, responses_endpoint: str, timeout_sec: int, max_retries: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.responses_endpoint = responses_endpoint
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def call_layout(self, session_id: str, image_data: str, ocr_json_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call OpenAI Responses API to produce strict JSON per NOOR layout schema.
        image_data should be base64 encoded image data.
        Returns (raw_text, parsed_obj).
        Raises on error.
        """
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")

        payload = {
            "model": config.OPENAI_MODEL,
            "instructions": (
                "You are a document layout and text-structuring assistant. "
                "Use the image and the provided OCR JSON to produce a strict JSON "
                "that follows the given schema. Correct obvious OCR errors. "
                "Preserve paragraphs→lines→words. Classify paragraphs (title, subtitle, heading, list, page_number, etc.). "
                "Return ONLY JSON. Also set source.image_path to 'inline' and source.ocr_json_path to 'inline'."
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"OCR JSON below (as-is, may contain errors):\n{ocr_json_text}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_data}"}
                    ]
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "noor_layout_v1",
                    "schema": NOOR_LAYOUT_V1,
                    "strict": True
                }
            },
            "temperature": 0.1
        }

        url = f"{self.base_url}{self.responses_endpoint}"
        attempt = 0
        backoff = 1.5
        last_error: Optional[str] = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                log_info("gpt_layout_request", f"Calling Responses API (attempt {attempt})", session_id=session_id)
                resp = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_sec)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    raise RuntimeError(last_error)

                data = resp.json()
                # Extract structured output text
                raw_text = data["output"][0]["content"][0]["text"]

                # Parse JSON
                parsed_obj = json.loads(raw_text)
                return raw_text, parsed_obj
            except Exception as e:
                last_error = str(e)
                if attempt > self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"OpenAI Responses API failed after {self.max_retries} retries: {last_error}")


