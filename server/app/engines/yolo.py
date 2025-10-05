import os
import asyncio
import cv2
import numpy as np
import torch
import time
import uuid
from collections import deque, Counter
from typing import Dict, Any, Optional, Tuple
from ultralytics import YOLO

from ..domain.ports import IGuidanceEngine
from ..domain.dtos import GuidanceResponse
from ..domain.enums import GuidanceDirection, MessageType
from ..core.config import USE_CUDA, FINAL_CAPTURE_CLASS, FINAL_CAPTURE_MIN_COUNT, FINAL_CAPTURE_MIN_FREQ, FINAL_FRAME_DIR, ENABLE_YOLO_EVAL_LOGS, YOLO_EVAL_TOPK, GUIDANCE_VOTE_MODE
from ..core.logger import log_info, log_error, log_debug
from ..core.errors import ErrorCode, YoloError


class YoloGuidanceEngine(IGuidanceEngine):
    """Real YOLO-based guidance engine for document framing."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize YOLO guidance engine."""
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "..", "models", "best (2).pt"
        )
        self.model = None
        self.class_names = None  # Will be populated from model.names
        self.device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
        self.ready = False
        
        # Frame management
        self.frame_buffer = deque(maxlen=10)  # Store last 10 frames
        self.best_frame = None
        self.best_frame_jpeg = None  # Store original JPEG bytes for best frame
        self.best_frame_score = 0.0
        
        # Temporal aggregation system (2-second window)
        self.class_history = deque(maxlen=100)  # Store class predictions
        self.history_timestamps = deque(maxlen=100)  # Store timestamps
        self.aggregation_window = 2.0  # 2 seconds
        self.last_guidance_time = 0.0
        self.guidance_cooldown = 2.0  # Minimum time between guidance updates (2 seconds)
        
        # Final capture state
        self.finalized = False  # Prevent multiple finalizations
        self.final_capture_dir = FINAL_FRAME_DIR
        
        # Class to action mapping (0..6 semantics)
        # Class semantics:
        # 0 = top-left corner
        # 1 = top-right corner  
        # 2 = bottom-right corner
        # 3 = bottom-left corner
        # 4 = paper face only
        # 5 = perfect / full page (final capture)
        # 6 = no document detected
        self.ACTIONS = {
            0: {"direction": GuidanceDirection.TOP_LEFT, "description": "top-left corner"},
            1: {"direction": GuidanceDirection.TOP_RIGHT, "description": "top-right corner"},
            2: {"direction": GuidanceDirection.BOTTOM_RIGHT, "description": "bottom-right corner"},
            3: {"direction": GuidanceDirection.BOTTOM_LEFT, "description": "bottom-left corner"},
            4: {"direction": GuidanceDirection.PAPER_FACE_ONLY, "description": "paper face only"},
            5: {"direction": GuidanceDirection.PERFECT, "description": "perfect / full page"},
            6: {"direction": GuidanceDirection.NO_DOCUMENT, "description": "no document detected"},
        }
        
        # Model will be loaded lazily on first use
        self._model_loading = False
    
    async def _load_model(self):
        """Load YOLO model asynchronously."""
        try:
            log_info("yolo_load", f"Loading YOLO model from {self.model_path}")
            
            # Load model in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            self.ready = True
            log_info("yolo_load", f"YOLO model loaded successfully on {self.device}")
            
        except Exception as e:
            log_error(ErrorCode.MODEL_LOAD_ERROR.value, f"Failed to load YOLO model: {str(e)}")
            raise YoloError(ErrorCode.MODEL_LOAD_ERROR, str(e))
    
    def _load_model_sync(self) -> YOLO:
        """Synchronous model loading with enhanced GPU detection."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        
        model = YOLO(self.model_path)
        
        # Cache class names from model
        self.class_names = model.names  # dict[int, str] mapping
        
        # Enhanced device detection and logging
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                log_info("yolo_load", f"YOLO model loaded on GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # Enable FP16 for efficiency if supported (temporarily disabled due to data type issues)
                # try:
                #     model.half()
                #     log_info("yolo_load", "FP16 mode enabled for faster inference")
                # except Exception as e:
                #     log_debug("yolo_load", f"FP16 not supported: {e}")
                log_info("yolo_load", "Using FP32 mode for compatibility")
                    
            except Exception as e:
                log_error("CUDA_ERROR", f"Failed to load model on GPU: {e}")
                log_info("yolo_load", "Falling back to CPU")
                model.to("cpu")
                self.device = "cpu"
        else:
            log_info("yolo_load", "CUDA not available, using CPU")
            model.to("cpu")
            self.device = "cpu"
        
        return model
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for YOLO inference."""
        try:
            # Decode JPEG bytes to image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid JPEG image data")
            
            # Convert BGR to RGB (YOLO expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 640x640 as specified
            image_resized = cv2.resize(image_rgb, (640, 640))
            
            # YOLO expects uint8 format, not normalized float32
            return image_resized
            
        except Exception as e:
            log_error(ErrorCode.INVALID_FORMAT.value, f"Image preprocessing failed: {str(e)}")
            raise YoloError(ErrorCode.INVALID_FORMAT, f"Image preprocessing failed: {str(e)}")
    
    async def analyze_frame(self, frame_data: bytes, metadata: Dict[str, Any], expected_class: Optional[int] = None) -> GuidanceResponse:
        """Analyze frame and return guidance response."""
        # Load model if not already loaded
        if not self.ready and not self._model_loading:
            self._model_loading = True
            try:
                await self._load_model()
            except Exception as e:
                self._model_loading = False
                raise e
        
        if not self.ready or self.model is None:
            log_error(ErrorCode.MODEL_BUSY.value, "YOLO model not ready")
            raise YoloError(ErrorCode.MODEL_BUSY, "YOLO model not ready")
        
        try:
            session_id = metadata.get("session_id", "unknown")
            log_debug("yolo_inference", f"Processing frame", session_id=session_id)
            
            # Preprocess image
            processed_image = self.preprocess_image(frame_data)
            
            # Run inference in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run_inference, processed_image
            )
            
            # Add evaluation logging if enabled
            if ENABLE_YOLO_EVAL_LOGS:
                await self._log_evaluation(result, frame_data, session_id, expected_class)
            
            # Generate guidance response
            guidance = self._generate_guidance(result, frame_data, metadata)
            
            # Update frame buffer and best frame tracking
            self._update_frame_buffer(frame_data, guidance, metadata)
            
            log_debug("yolo_inference", f"Guidance generated: {guidance.dir.value}", session_id=session_id)
            return guidance
            
        except YoloError:
            raise
        except Exception as e:
            log_error(ErrorCode.INFERENCE_ERROR.value, f"YOLO inference failed: {str(e)}")
            raise YoloError(ErrorCode.INFERENCE_ERROR, f"YOLO inference failed: {str(e)}")
    
    def _run_inference(self, image: np.ndarray) -> Dict[str, Any]:
        """Run YOLO inference synchronously."""
        try:
            # Run prediction
            results = self.model.predict(image, imgsz=640, conf=0.55, verbose=False)
            
            if not results or len(results) == 0:
                return {"class": 6, "confidence": 0.1, "all_probs": [0.1] * 7}  # Class 6 for no detection
            
            # Extract classification results
            result = results[0]
            
            if hasattr(result, 'probs') and result.probs is not None:
                # Get probabilities for all classes
                probs = result.probs.data.cpu().numpy()
                
                # Get top class and confidence
                top_class = int(np.argmax(probs))
                confidence = float(probs[top_class])
                
                # If confidence is too low, treat as class 7 (no detection)
                if confidence < 0.3:
                    top_class = 7
                    confidence = 0.1
                
                return {
                    "class": top_class,
                    "confidence": confidence,
                    "all_probs": probs.tolist()
                }
            else:
                # Fallback if no classification results
                return {"class": 6, "confidence": 0.1, "all_probs": [0.1] * 7}
                
        except Exception as e:
            log_error(ErrorCode.INFERENCE_ERROR.value, f"YOLO prediction failed: {str(e)}")
            raise
    
    def _get_most_frequent_class(self, current_time: float) -> Tuple[int, float, Dict[str, Any]]:
        """Get the most frequent class in the last 2 seconds using EMA or majority vote."""
        # Filter predictions within the aggregation window
        recent_predictions = []
        for i, timestamp in enumerate(self.history_timestamps):
            if current_time - timestamp <= self.aggregation_window:
                recent_predictions.append(self.class_history[i])
        
        if not recent_predictions:
            return 6, 0.1, self.ACTIONS[6]  # No recent predictions, return class 6
        
        # Count frequency of each class
        class_counts = Counter(recent_predictions)
        
        if GUIDANCE_VOTE_MODE == "majority":
            # Majority vote mode: pick the most common class
            most_frequent_class = class_counts.most_common(1)[0][0]
            frequency = class_counts[most_frequent_class] / len(recent_predictions)
            
            log_debug("temporal_aggregation", 
                     f"Majority vote: class {most_frequent_class} (frequency: {frequency:.2f}, "
                     f"count: {class_counts[most_frequent_class]}/{len(recent_predictions)})")
        else:
            # EMA mode (default): use existing logic
            most_frequent_class = class_counts.most_common(1)[0][0]
            frequency = class_counts[most_frequent_class] / len(recent_predictions)
            
            log_debug("temporal_aggregation", 
                     f"EMA mode: class {most_frequent_class} (frequency: {frequency:.2f}, "
                     f"count: {class_counts[most_frequent_class]}/{len(recent_predictions)})")
        
        return most_frequent_class, frequency, self.ACTIONS[most_frequent_class]
    
    def maybe_finalize(self, session_id: str) -> Dict[str, Any]:
        """Check if class 5 dominates and return best frame for final capture."""
        if self.finalized:
            log_debug("maybe_finalize", f"Already finalized, skipping", session_id=session_id)
            return {"triggered": False, "best_frame_jpeg": None, "save_path": None}
        
        current_time = time.time()
        
        # Get most frequent class in the last 2 seconds
        recent_predictions = []
        for i, timestamp in enumerate(self.history_timestamps):
            if current_time - timestamp <= self.aggregation_window:
                recent_predictions.append(self.class_history[i])
        
        if not recent_predictions:
            return {"triggered": False, "best_frame_jpeg": None, "save_path": None}
        
        # Count frequency of each class
        class_counts = Counter(recent_predictions)
        most_frequent_class = class_counts.most_common(1)[0][0]
        frequency = class_counts[most_frequent_class] / len(recent_predictions)
        count = class_counts[most_frequent_class]
        
        # Check final capture conditions
        log_debug("maybe_finalize", 
                 f"Checking conditions: class={most_frequent_class} (need {FINAL_CAPTURE_CLASS}), "
                 f"count={count} (need >={FINAL_CAPTURE_MIN_COUNT}), "
                 f"freq={frequency:.2f} (need >={FINAL_CAPTURE_MIN_FREQ})",
                 session_id=session_id)
        
        if (most_frequent_class == FINAL_CAPTURE_CLASS and 
            count >= FINAL_CAPTURE_MIN_COUNT and 
            frequency >= FINAL_CAPTURE_MIN_FREQ):
            
            log_info("final_candidate", 
                    f"Class {FINAL_CAPTURE_CLASS} dominates: count={count}, freq={frequency:.2f}, window_sec={self.aggregation_window}",
                    session_id=session_id)
            
            # Get best frame from buffer
            if self.best_frame is not None:
                # Ensure directory exists
                os.makedirs(self.final_capture_dir, exist_ok=True)
                
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"NOOR_{session_id}_{timestamp}.jpg"
                save_path = os.path.join(self.final_capture_dir, filename)
                
                # Save frame (prefer original JPEG bytes if available)
                try:
                    if hasattr(self, 'best_frame_jpeg') and self.best_frame_jpeg is not None:
                        # Use original JPEG bytes
                        with open(save_path, 'wb') as f:
                            f.write(self.best_frame_jpeg)
                    else:
                        # Encode from BGR array
                        success, encoded_img = cv2.imencode('.jpg', self.best_frame, 
                                                          [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if success:
                            with open(save_path, 'wb') as f:
                                f.write(encoded_img.tobytes())
                        else:
                            raise Exception("Failed to encode frame")
                    
                    # Mark as finalized
                    self.finalized = True
                    
                    log_info("final_selected", 
                            f"Best frame saved: path={save_path}, score={self.best_frame_score:.3f}",
                            session_id=session_id)
                    
                    # Return relative path for HTTP access
                    relative_path = f"/static/final_captures/{filename}"
                    
                    return {
                        "triggered": True,
                        "best_frame_jpeg": getattr(self, 'best_frame_jpeg', None),
                        "save_path": relative_path,
                        "confidence": self.best_frame_score
                    }
                    
                except Exception as e:
                    log_error("final_save_failed", 
                             f"Failed to save final frame: {str(e)}",
                             session_id=session_id)
                    return {"triggered": False, "best_frame_jpeg": None, "save_path": None}
            else:
                log_debug("final_candidate", "No best frame available for final capture", session_id=session_id)
                return {"triggered": False, "best_frame_jpeg": None, "save_path": None}
        
        return {"triggered": False, "best_frame_jpeg": None, "save_path": None}
    
    async def _log_evaluation(self, result: Dict[str, Any], frame_data: bytes, session_id: str, expected_class: Optional[int] = None):
        """Log YOLO evaluation metrics for debugging."""
        try:
            # Get image dimensions from decoded frame
            nparr = np.frombuffer(frame_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_h, img_w = image.shape[:2] if image is not None else (0, 0)
            
            # Get probabilities and compute top-k
            all_probs = result.get("all_probs", [])
            if not all_probs:
                return
            
            probs = np.array(all_probs)
            topk = YOLO_EVAL_TOPK
            idxs = np.argsort(-probs)[:topk]
            topk_ids = idxs.tolist()
            topk_confs = [float(probs[i]) for i in idxs]
            top1_id = int(topk_ids[0])
            top1_conf = float(topk_confs[0])
            
            # Get class names if available
            top1_name = self.class_names.get(top1_id, str(top1_id)) if self.class_names else str(top1_id)
            topk_names = [self.class_names.get(cid, str(cid)) if self.class_names else str(cid) for cid in topk_ids]
            
            # Log eval_topk
            log_info("eval_topk", 
                    f"Top-k evaluation: top1_id={top1_id} ({top1_name}), top1_conf={top1_conf:.3f}, "
                    f"topk_ids={topk_ids}, topk_names={topk_names}, topk_confs={[f'{c:.3f}' for c in topk_confs]}, "
                    f"img_w={img_w}, img_h={img_h}",
                    session_id=session_id)
            
            # Log eval_expected if expected class provided
            if expected_class is not None:
                # Validate expected class
                if not isinstance(expected_class, int) or expected_class < 0 or expected_class > 6:
                    log_info("eval_expected_ignored", 
                            f"Invalid expected class: {expected_class} (must be 0-6)",
                            session_id=session_id)
                    return
                
                passed = (top1_id == expected_class)
                log_info("eval_expected", 
                        f"Expected validation: expected={expected_class}, passed={passed}, "
                        f"top1_id={top1_id}, top1_conf={top1_conf:.3f}",
                        session_id=session_id)
                        
        except Exception as e:
            # Don't let evaluation logging break the main flow
            log_debug("eval_logging_error", f"Evaluation logging failed: {str(e)}", session_id=session_id)
    
    def _generate_guidance(
        self, 
        inference_result: Dict[str, Any], 
        frame_data: bytes, 
        metadata: Dict[str, Any]
    ) -> GuidanceResponse:
        """Generate guidance response from YOLO inference with temporal aggregation."""
        
        current_time = time.time()
        predicted_class = inference_result.get("class", 6)  # Default to 6 (no document)
        
        # Defensive remap: if class 7 appears from old code/weights, coerce to 6
        if predicted_class == 7:
            predicted_class = 6
            
        confidence = inference_result.get("confidence", 0.1)
        
        # Add current prediction to history
        self.class_history.append(predicted_class)
        self.history_timestamps.append(current_time)
        
        # Check if we should update guidance (cooldown period)
        if current_time - self.last_guidance_time < self.guidance_cooldown:
            # Return the most recent guidance from temporal aggregation
            most_frequent_class, frequency, action = self._get_most_frequent_class(current_time)
            return GuidanceResponse(
                type=MessageType.GUIDANCE,
                dir=action["direction"],
                magnitude=0.0,  # Set to 0.0 since magnitude is removed
                coverage=min(confidence * frequency, 1.0),
                skew_deg=0.0,
                conf=confidence,
                ready=(most_frequent_class == 5 and frequency >= 0.8 and confidence > 0.9)
            )
        
        # Get most frequent class in the last 2 seconds
        most_frequent_class, frequency, action = self._get_most_frequent_class(current_time)
        
        # Calculate coverage and confidence
        coverage = min(confidence * frequency, 1.0)  # Combine confidence with frequency
        final_confidence = confidence
        
        # Calculate skew (fixed for classification model)
        skew_deg = 0.0
        
        # Determine if ready for capture (class 5 with high frequency and confidence)
        ready = (most_frequent_class == 5 and frequency >= 0.8 and confidence > 0.9)
        
        # Update last guidance time
        self.last_guidance_time = current_time
        
        log_debug("yolo_guidance", 
                 f"Temporal guidance: class {most_frequent_class} (freq: {frequency:.2f}, "
                 f"conf: {confidence:.2f}, ready: {ready})")
        
        return GuidanceResponse(
            type=MessageType.GUIDANCE,
            dir=action["direction"],  # Keep dir for internal compatibility
            magnitude=0.0,  # Set to 0.0 since magnitude is removed
            coverage=coverage,
            skew_deg=skew_deg,
            conf=final_confidence,
            ready=ready
        )
    
    def _update_frame_buffer(
        self, 
        frame_data: bytes, 
        guidance: GuidanceResponse, 
        metadata: Dict[str, Any]
    ):
        """Update frame buffer and track best frame."""
        frame_info = {
            "data": frame_data,
            "guidance": guidance,
            "metadata": metadata,
            "score": guidance.coverage * guidance.conf  # Combined score
        }
        
        self.frame_buffer.append(frame_info)
        
        # Update best frame if this one is better
        if frame_info["score"] > self.best_frame_score:
            self.best_frame = frame_info
            self.best_frame_jpeg = frame_data  # Store original JPEG bytes
            self.best_frame_score = frame_info["score"]
    
    def freeze_best_frame(self) -> Optional[Dict[str, Any]]:
        """Get the best frame from recent buffer."""
        if self.best_frame:
            log_info("yolo_capture", f"Best frame selected with score: {self.best_frame_score}")
            return self.best_frame
        elif self.frame_buffer:
            # Fallback to most recent frame
            recent_frame = self.frame_buffer[-1]
            log_info("yolo_capture", "Using most recent frame as fallback")
            return recent_frame
        else:
            log_error(ErrorCode.SERVER_ERROR.value, "No frames available for capture")
            return None
    
    async def is_ready(self) -> bool:
        """Check if YOLO engine is ready."""
        # Load model if not already loaded
        if not self.ready and not self._model_loading:
            self._model_loading = True
            try:
                await self._load_model()
            except Exception as e:
                self._model_loading = False
                log_error(ErrorCode.MODEL_LOAD_ERROR.value, f"Failed to load model in is_ready: {str(e)}")
                return False
        
        return self.ready and self.model is not None
    
    def get_gpu_info(self) -> dict:
        """Get GPU information and usage."""
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                
                return {
                    "available": True,
                    "name": gpu_name,
                    "memory_total_gb": round(gpu_memory_total, 2),
                    "memory_allocated_gb": round(gpu_memory_allocated, 2),
                    "memory_cached_gb": round(gpu_memory_cached, 2),
                    "memory_free_gb": round(gpu_memory_total - gpu_memory_allocated, 2)
                }
            except Exception as e:
                return {"available": False, "error": str(e)}
        else:
            return {"available": False, "reason": "CUDA not available"}
    
    def reset_session_state(self):
        """Reset session-specific state for new WebSocket connections."""
        self.finalized = False
        self.class_history.clear()
        self.history_timestamps.clear()
        self.frame_buffer.clear()
        self.best_frame = None
        self.best_frame_jpeg = None
        self.best_frame_score = 0.0
        self.last_guidance_time = 0.0
        log_debug("yolo_reset", "Session state reset for new connection")
