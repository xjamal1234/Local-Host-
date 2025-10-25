"""
OCR Processing Use Case for orchestrating text extraction and JSON formatting.
Handles document ID generation, file persistence, and structured output formatting.
"""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from app.core.logger import log_info, log_error, log_debug
from app.engines.ocr_model import OcrEngine
from app.core import config


class OcrProcessingUseCase:
    """Use case for processing final captured frames with OCR."""
    
    def __init__(self, ocr_engine: OcrEngine, config_obj):
        """
        Initialize OCR processing use case.
        
        Args:
            ocr_engine: Initialized OCR engine instance
            config_obj: Configuration object with OCR settings
        """
        self.ocr_engine = ocr_engine
        self.config = config_obj
        
        # Ensure output directory exists if saving is enabled
        if self.config.OCR_SAVE_JSON:
            os.makedirs(self.config.OCR_OUTPUT_DIR, exist_ok=True)
            log_debug("ocr_uc_init", f"OCR output directory ensured: {self.config.OCR_OUTPUT_DIR}")
    
    def _generate_doc_id(self, session_id: str) -> str:
        """
        Generate document ID from session ID and timestamp.
        
        Args:
            session_id: WebSocket session identifier
            
        Returns:
            Generated document ID in format: NOOR_<sessionId>_<unixTs>
        """
        timestamp = int(time.time())
        return f"NOOR_{session_id}_{timestamp}"
    
    def _get_json_path(self, image_path: str, doc_id: str) -> str:
        """
        Generate JSON file path colocated with the image.
        
        Args:
            image_path: Path to the saved image file
            doc_id: Document identifier
            
        Returns:
            Path for the JSON file (same directory, same basename, .json extension)
        """
        # Convert to Path object for easier manipulation
        image_path_obj = Path(image_path)
        
        # Get the directory and basename without extension
        directory = image_path_obj.parent
        basename = image_path_obj.stem  # filename without extension
        
        # Create JSON path in the same directory
        json_filename = f"{basename}.json"
        json_path = directory / json_filename
        
        return str(json_path)
    
    def _save_error_json(self, json_path: str, doc_id: str, error_msg: str) -> None:
        """
        Save error information as JSON when OCR fails.
        
        Args:
            json_path: Path where to save the error JSON
            doc_id: Document identifier
            error_msg: Error description
        """
        try:
            error_data = {
                "docId": doc_id,
                "error": "OCR_FAILED",
                "message": error_msg
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            log_info("ocr_error_saved", f"Error JSON saved: {json_path}")
            
        except Exception as e:
            log_error("ocr_error_save_failed", f"Failed to save error JSON: {str(e)}")
    
    def _save_success_json(self, json_path: str, ocr_result: Dict[str, Any], doc_id: str) -> None:
        """
        Save successful OCR results as JSON.
        
        Args:
            json_path: Path where to save the JSON
            ocr_result: OCR processing results
            doc_id: Document identifier
        """
        try:
            # Build the final JSON structure
            final_json = {
                "docId": doc_id,
                "lang": "auto",
                "full_text": ocr_result["full_text"],
                "paragraphs": ocr_result["paragraphs"],
                "metrics": ocr_result["metrics"]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(json_path)
            log_info("ocr_json_saved", 
                    f"OCR JSON saved: {json_path}, size: {file_size} bytes")
            
        except Exception as e:
            log_error("ocr_json_save_failed", f"Failed to save OCR JSON: {str(e)}")
            raise
    
    def run(self, session_id: str, image_path: str) -> Dict[str, Any]:
        """
        Run OCR processing on the final captured frame.
        
        Args:
            session_id: WebSocket session identifier
            image_path: Path to the saved final frame image
            
        Returns:
            OCR result dictionary only (no file writing)
        """
        start_time = time.time()
        doc_id = self._generate_doc_id(session_id)
        
        log_info("ocr_start", f"Starting OCR processing for session {session_id}", 
                session_id=session_id)
        
        try:
            # Verify image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Run OCR processing
            ocr_result = self.ocr_engine.run(image_path)
            
            # Add document ID to result
            ocr_result["docId"] = doc_id
            ocr_result["lang"] = "auto"
            
            processing_time = time.time() - start_time
            
            log_info("ocr_completed", 
                    f"OCR processing completed in {processing_time:.2f}s for session {session_id}",
                    session_id=session_id)
            
            return ocr_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Build error result
            error_result = {
                "docId": doc_id,
                "lang": "auto",
                "full_text": "",
                "paragraphs": [],
                "metrics": {
                    "paragraph_count": 0,
                    "line_count": 0,
                    "word_count": 0
                },
                "error": "OCR_FAILED",
                "message": str(e)
            }
            
            log_error("ocr_failed", 
                     f"OCR processing failed after {processing_time:.2f}s for session {session_id}: {str(e)}",
                     session_id=session_id)
            
            return error_result