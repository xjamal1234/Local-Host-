"""
Deterministic Storage Manager for Phase 4 - Image Processing + Storage
Provides atomic ID allocation and organized folder structure for final captures.
"""

import os
import threading
from pathlib import Path
from typing import Tuple, Dict, Any
import cv2
import numpy as np

from app.core.logger import log_info, log_error, log_debug


class StorageManager:
    """Thread-safe storage manager for organized final captures with numeric IDs."""
    
    def __init__(self):
        """Initialize storage manager with exact base directory."""
        self.base_dir = Path(r"D:\Noor\server\server\app\static\final_captures")
        self.counter_file = self.base_dir / "counter.txt"
        self._lock = threading.Lock()
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counter if missing
        self._initialize_counter()
        
        log_info("storage_manager_init", f"Storage manager initialized with base dir: {self.base_dir}")
    
    def _initialize_counter(self) -> None:
        """Initialize counter.txt if it doesn't exist."""
        if not self.counter_file.exists():
            try:
                self.counter_file.write_text("0", encoding="utf-8")
                log_debug("storage_counter_init", f"Initialized counter file: {self.counter_file}")
            except Exception as e:
                log_error("storage_counter_init_failed", f"Failed to initialize counter: {str(e)}")
                raise IOError(f"Failed to initialize counter file: {str(e)}")
    
    def allocate_capture_id(self) -> Tuple[int, Path]:
        """
        Atomically allocate a new capture ID and create its folder.
        
        Returns:
            Tuple of (capture_id, capture_dir_path)
        """
        with self._lock:
            try:
                # Read current counter
                current_id = int(self.counter_file.read_text(encoding="utf-8").strip())
                new_id = current_id + 1
                
                # Create capture directory (zero-padded to 4 digits)
                capture_dir = self.base_dir / f"{new_id:04d}"
                capture_dir.mkdir(exist_ok=True)
                
                # Update counter atomically
                self.counter_file.write_text(str(new_id), encoding="utf-8")
                
                log_info("storage_id_allocated", f"Allocated capture ID: {new_id:04d}, dir: {capture_dir}")
                return new_id, capture_dir
                
            except Exception as e:
                log_error("storage_id_allocation_failed", f"Failed to allocate capture ID: {str(e)}")
                raise IOError(f"Failed to allocate capture ID: {str(e)}")
    
    def save_scanned_color(self, img_bgr: np.ndarray, capture_id: int, cap_dir: Path) -> Path:
        """
        Save the processed scanned color image as scanned_color_####.jpg.
        
        Args:
            img_bgr: BGR image array from OpenCV
            capture_id: Capture ID number
            cap_dir: Capture directory path
            
        Returns:
            Path to the saved image file
            
        Raises:
            IOError: If saving fails
            ValueError: If image is invalid
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid image data provided")
        
        if not cap_dir.exists():
            raise IOError(f"Capture directory does not exist: {cap_dir}")
        
        try:
            # Create filename
            filename = f"scanned_color_{capture_id:04d}.jpg"
            image_path = cap_dir / filename
            
            # Encode and save image (Unicode-safe on Windows)
            success, encoded_img = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise IOError("Failed to encode image as JPEG")
            
            # Write using OpenCV's Unicode-safe method
            encoded_img.tofile(str(image_path))
            
            log_info("storage_image_saved", f"Saved scanned color image: {image_path}")
            return image_path
            
        except Exception as e:
            log_error("storage_image_save_failed", f"Failed to save scanned color image: {str(e)}")
            raise IOError(f"Failed to save scanned color image: {str(e)}")
    
    def save_json(self, data: Dict[str, Any], prefix: str, capture_id: int, cap_dir: Path) -> Path:
        """
        Save JSON data as {prefix}_####.json with UTF-8 encoding.
        
        Args:
            data: Dictionary to save as JSON
            prefix: Prefix for filename (e.g., "ocr", "gpt")
            capture_id: Capture ID number
            cap_dir: Capture directory path
            
        Returns:
            Path to the saved JSON file
            
        Raises:
            IOError: If saving fails
        """
        if not cap_dir.exists():
            raise IOError(f"Capture directory does not exist: {cap_dir}")
        
        try:
            import json
            
            # Create filename
            filename = f"{prefix}_{capture_id:04d}.json"
            json_path = cap_dir / filename
            
            # Write JSON with UTF-8 encoding (Unicode-safe)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            log_info("storage_json_saved", f"Saved {prefix} JSON: {json_path}")
            return json_path
            
        except Exception as e:
            log_error("storage_json_save_failed", f"Failed to save {prefix} JSON: {str(e)}")
            raise IOError(f"Failed to save {prefix} JSON: {str(e)}")
    
    def normalize_path(self, path: Path) -> str:
        """
        Normalize absolute path for metadata.
        
        Args:
            path: Path to normalize
            
        Returns:
            Resolved absolute path as string
        """
        return str(path.resolve())


# Global storage manager instance
storage_manager = StorageManager()
