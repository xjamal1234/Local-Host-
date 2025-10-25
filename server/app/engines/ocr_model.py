"""
OCR Engine for processing final captured frames using EasyOCR.
Implements paragraph grouping and text extraction with proper unicode handling.
"""

import os
import time
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import easyocr

from app.core.logger import log_info, log_error, log_debug


class OcrEngine:
    """OCR engine using EasyOCR for text extraction and paragraph grouping."""
    
    def __init__(self, langs: List[str], gpu: bool = False):
        """
        Initialize OCR engine with EasyOCR.
        
        Args:
            langs: List of language codes (e.g., ['en', 'ar'])
            gpu: Whether to use GPU acceleration (default: False)
        """
        self.langs = langs
        self.gpu = gpu
        self.reader = None
        
        # Initialize EasyOCR reader with GPU detection
        try:
            import torch
            
            # Auto-detect GPU if not explicitly specified
            if gpu is None:
                use_gpu = torch.cuda.is_available()
            else:
                use_gpu = gpu and torch.cuda.is_available()
            
            self.reader = easyocr.Reader(langs, gpu=use_gpu)
            
            # Log GPU information
            if use_gpu:
                device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
                log_info("ocr_engine_init", 
                        f"EasyOCR initialized with languages: {langs}, GPU: {use_gpu} "
                        f"(Device: {device_name}, Memory: {memory_gb:.1f}GB)")
            else:
                log_info("ocr_engine_init", 
                        f"EasyOCR initialized with languages: {langs}, GPU: {use_gpu} (CPU mode)")
                        
        except Exception as e:
            log_error("ocr_engine_init_failed", f"Failed to initialize EasyOCR: {str(e)}")
            raise
    
    def imread_unicode(self, path: str) -> Optional[np.ndarray]:
        """
        Unicode-safe image reading function.
        
        Args:
            path: Path to image file
            
        Returns:
            BGR image as numpy array, or None if failed
        """
        try:
            # Read raw bytes first to handle unicode paths
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR format
            return image
        except Exception as e:
            log_error("image_read_failed", f"Failed to read image {path}: {str(e)}")
            return None
    
    def _sort_by_top_y(self, results: List[Tuple]) -> List[Tuple]:
        """
        Sort OCR results by top Y coordinate (minimum Y in polygon).
        
        Args:
            results: List of (polygon, text, confidence) tuples
            
        Returns:
            Sorted list by top Y coordinate
        """
        def get_top_y(polygon):
            # Get minimum Y coordinate from polygon
            return min(point[1] for point in polygon)
        
        return sorted(results, key=lambda x: get_top_y(x[0]))
    
    def _compute_vertical_gaps(self, sorted_results: List[Tuple]) -> List[float]:
        """
        Compute vertical gaps between consecutive text lines.
        
        Args:
            sorted_results: OCR results sorted by top Y
            
        Returns:
            List of vertical gaps between consecutive lines
        """
        gaps = []
        for i in range(1, len(sorted_results)):
            # Get bottom Y of previous line
            prev_bottom = max(point[1] for point in sorted_results[i-1][0])
            # Get top Y of current line
            curr_top = min(point[1] for point in sorted_results[i][0])
            # Compute gap
            gap = curr_top - prev_bottom
            gaps.append(max(0, gap))  # Ensure non-negative
        return gaps
    
    def _group_into_paragraphs(self, sorted_results: List[Tuple]) -> List[List[Tuple]]:
        """
        Group text lines into paragraphs based on vertical gaps.
        
        Args:
            sorted_results: OCR results sorted by top Y
            
        Returns:
            List of paragraphs, where each paragraph is a list of (polygon, text, confidence)
        """
        if not sorted_results:
            return []
        
        if len(sorted_results) == 1:
            return [sorted_results]
        
        # Compute vertical gaps
        gaps = self._compute_vertical_gaps(sorted_results)
        
        if not gaps:
            return [sorted_results]
        
        # Calculate average gap
        avg_gap = sum(gaps) / len(gaps)
        
        # Group lines into paragraphs
        paragraphs = []
        current_paragraph = [sorted_results[0]]
        
        for i in range(1, len(sorted_results)):
            gap = gaps[i-1]
            
            # Start new paragraph if gap is significantly larger than average
            if gap > 1.5 * avg_gap and avg_gap > 0:
                paragraphs.append(current_paragraph)
                current_paragraph = [sorted_results[i]]
            else:
                current_paragraph.append(sorted_results[i])
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace.
        Keeps punctuation as returned by OCR.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def run(self, image_path: str) -> Dict[str, Any]:
        """
        Run OCR on the specified image and return structured results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
            - full_text: Complete text with newlines
            - paragraphs: List of paragraph objects
            - metrics: Count statistics
        """
        start_time = time.time()
        
        try:
            # Read image with unicode-safe method
            image = self.imread_unicode(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            log_debug("ocr_start", f"Processing image: {image_path}, size: {image.shape}")
            
            # Run EasyOCR
            results = self.reader.readtext(image)
            
            if not results:
                log_debug("ocr_no_text", "No text detected in image")
                return {
                    "full_text": "",
                    "paragraphs": [],
                    "metrics": {
                        "paragraph_count": 0,
                        "line_count": 0,
                        "word_count": 0
                    }
                }
            
            log_debug("ocr_raw_results", f"Found {len(results)} text regions")
            
            # Sort results by top Y coordinate
            sorted_results = self._sort_by_top_y(results)
            
            # Group into paragraphs
            paragraphs_data = self._group_into_paragraphs(sorted_results)
            
            # Build structured output
            full_text_lines = []
            paragraphs = []
            total_words = 0
            
            for para_idx, paragraph_lines in enumerate(paragraphs_data):
                para_text_lines = []
                lines = []
                
                for line_data in paragraph_lines:
                    polygon, text, confidence = line_data
                    
                    # Tokenize the line
                    words = self._tokenize_text(text)
                    total_words += len(words)
                    
                    # Store line data
                    line_obj = {
                        "text": text,
                        "words": words
                    }
                    lines.append(line_obj)
                    para_text_lines.append(text)
                
                # Paragraph text (join lines with newlines)
                para_text = "\n".join(para_text_lines)
                
                # Store paragraph data
                paragraph_obj = {
                    "text": para_text,
                    "lines": lines
                }
                paragraphs.append(paragraph_obj)
                
                # Add to full text
                full_text_lines.extend(para_text_lines)
            
            # Combine all text
            full_text = "\n".join(full_text_lines)
            
            # Build metrics
            metrics = {
                "paragraph_count": len(paragraphs),
                "line_count": len(sorted_results),
                "word_count": total_words
            }
            
            processing_time = time.time() - start_time
            log_info("ocr_completed", 
                    f"OCR completed in {processing_time:.2f}s: {metrics['paragraph_count']} paragraphs, "
                    f"{metrics['line_count']} lines, {metrics['word_count']} words")
            
            return {
                "full_text": full_text,
                "paragraphs": paragraphs,
                "metrics": metrics
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_error("ocr_failed", 
                     f"OCR processing failed after {processing_time:.2f}s: {str(e)}")
            raise
