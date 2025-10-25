"""
Document Cropper for Phase 4 - Image Processing
Implements LAB+Otsu page detection, perspective correction, and image enhancement.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from app.core.logger import log_info, log_error, log_debug


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
        
    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left point has smallest sum
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point has largest sum
    rect[2] = pts[np.argmax(s)]
    # Top-right point has smallest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left point has largest difference
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation using 4 corner points.
    
    Args:
        image: Input image
        pts: 4 corner points in order (top-left, top-right, bottom-right, bottom-left)
        
    Returns:
        Perspective-corrected image
    """
    rect = order_points(pts)
    
    # Calculate dimensions of new image
    (tl, tr, br, bl) = rect
    
    # Compute width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped


def grayworld_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Apply gray world white balance correction.
    
    Args:
        img: Input BGR image
        
    Returns:
        White-balanced BGR image
    """
    # Convert to float
    img_float = img.astype(np.float32)
    
    # Calculate average for each channel
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    
    # Gray world assumption: average should be gray
    gray_value = (avg_b + avg_g + avg_r) / 3.0
    
    # Calculate scaling factors
    scale_b = gray_value / avg_b if avg_b > 0 else 1.0
    scale_g = gray_value / avg_g if avg_g > 0 else 1.0
    scale_r = gray_value / avg_r if avg_r > 0 else 1.0
    
    # Apply scaling
    img_balanced = img_float.copy()
    img_balanced[:, :, 0] *= scale_b
    img_balanced[:, :, 1] *= scale_g
    img_balanced[:, :, 2] *= scale_r
    
    # Clip values and convert back to uint8
    img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)
    
    return img_balanced


def enhance_color_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel in LAB color space.
    
    Args:
        img: Input BGR image
        
    Returns:
        Enhanced BGR image
    """
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def detect_quad_lab_otsu(bgr_full: np.ndarray, detect_longside: int = 1200, morph_kernel: int = 5) -> Optional[np.ndarray]:
    """
    Detect document quad using LAB+Otsu thresholding and morphology.
    
    Args:
        bgr_full: Full resolution BGR image
        detect_longside: Maximum size for detection (resize if larger)
        morph_kernel: Morphology kernel size
        
    Returns:
        Array of 4 corner points (x, y) or None if no quad found
    """
    # Resize for detection if needed
    h, w = bgr_full.shape[:2]
    if max(h, w) > detect_longside:
        scale = detect_longside / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        bgr_resized = cv2.resize(bgr_full, (new_w, new_h))
    else:
        bgr_resized = bgr_full.copy()
        scale = 1.0
    
    # Convert to LAB and extract L channel
    lab = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    # Apply Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology operations (close then open)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Check if we have 4 points
    if len(approx) == 4:
        # Convert to full-scale coordinates
        points = approx.reshape(4, 2).astype(np.float32)
        if scale != 1.0:
            points = points / scale
        return points
    
    # Fallback: use minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    
    # Convert to full-scale coordinates
    if scale != 1.0:
        box = box / scale
    
    return box


def run_processing_on_bgr(bgr: np.ndarray, auto_rotate: bool = True, rotate_ratio: float = 1.25) -> np.ndarray:
    """
    Main processing function: detect document, apply perspective correction, and enhance.
    
    Args:
        bgr: Input BGR image
        auto_rotate: Whether to auto-rotate if width > height * ratio
        rotate_ratio: Ratio threshold for auto-rotation
        
    Returns:
        Processed BGR image
        
    Raises:
        ValueError: If no document quad is detected
    """
    log_debug("processing_start", f"Starting document processing, image shape: {bgr.shape}")
    
    # Step 1: Detect document quad
    quad_points = detect_quad_lab_otsu(bgr)
    if quad_points is None:
        raise ValueError("Page not found: No document quad detected in image")
    
    log_debug("processing_quad_detected", f"Document quad detected with {len(quad_points)} points")
    
    # Step 2: Apply perspective correction
    warped = four_point_transform(bgr, quad_points)
    log_debug("processing_perspective_applied", f"Perspective correction applied, output shape: {warped.shape}")
    
    # Step 3: Auto-rotate if needed
    h, w = warped.shape[:2]
    if auto_rotate and w > h * rotate_ratio:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        log_debug("processing_rotated", f"Auto-rotated image, new shape: {warped.shape}")
    
    # Step 4: Apply gray world white balance
    balanced = grayworld_white_balance(warped)
    log_debug("processing_white_balanced", "Applied gray world white balance")
    
    # Step 5: Apply CLAHE enhancement
    enhanced = enhance_color_clahe(balanced)
    log_debug("processing_clahe_applied", "Applied CLAHE enhancement")
    
    log_info("processing_completed", f"Document processing completed successfully, final shape: {enhanced.shape}")
    
    return enhanced
