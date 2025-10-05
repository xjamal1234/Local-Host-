import json
import asyncio
import uuid
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from ...core.config import HEARTBEAT_SEC, TIMEOUT_SEC, MAX_JPEG_KB, CADENCE_MS, MAX_FPS, STOP_BEHAVIOR, ACK_TIMEOUT_MS
from ...core.logger import log_info, log_error, log_debug
from ...core.errors import ErrorCode, create_ws_error_response
from ...domain.enums import MessageType, LoadHint
from ...domain.dtos import FrameMetadata, CaptureRequest, HeartbeatResponse
from ...core.di import container


class WebSocketManager:
    """Manages WebSocket connections and message handling with rate limiting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept WebSocket connection and return session ID."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "last_seq": 0,
            "connected_at": time.time(),
            "frame_count": 0,
            "last_frame_time": 0,
            "frame_times": [],  # Track frame timing for rate limiting
            "last_guidance_time": 0,
            "processing": False,
            # New finalization fields
            "mode": "guidance",  # "guidance" or "processing_pending"
            "accepting_frames": True,
            "guidance_enabled": True,
            "frames_dropped_after_final": 0
        }
        log_info("ws_connect", f"WebSocket connected", session_id=session_id)
        return session_id
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection."""
        # Log final tally of dropped frames if session was finalized
        if session_id in self.session_data:
            session_data = self.session_data[session_id]
            if session_data.get("mode") == "processing_pending":
                dropped_count = session_data.get("frames_dropped_after_final", 0)
                if dropped_count > 0:
                    log_info("frames_dropped_after_final", f"Final tally: {dropped_count} frames dropped after finalization", session_id=session_id)
        
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        log_info("ws_disconnect", f"WebSocket disconnected", session_id=session_id)
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                log_error("WS_SEND_ERROR", f"Failed to send message: {str(e)}", session_id=session_id)
    
    def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limits."""
        if session_id not in self.session_data:
            return False
        
        session = self.session_data[session_id]
        current_time = time.time()
        
        # Clean old frame times (older than 1 second)
        session["frame_times"] = [
            t for t in session["frame_times"] 
            if current_time - t < 1.0
        ]
        
        # Check if exceeding MAX_FPS
        if len(session["frame_times"]) >= MAX_FPS:
            return False
        
        # Add current frame time
        session["frame_times"].append(current_time)
        session["last_frame_time"] = current_time
        
        return True
    
    def should_send_guidance(self, session_id: str) -> bool:
        """Check if it's time to send guidance (respect CADENCE_MS)."""
        if session_id not in self.session_data:
            return True
        
        session = self.session_data[session_id]
        current_time = time.time() * 1000  # Convert to milliseconds
        
        if current_time - session["last_guidance_time"] >= CADENCE_MS:
            session["last_guidance_time"] = current_time
            return True
        
        return False
    
    async def send_heartbeat(self, session_id: str):
        """Send heartbeat message."""
        session_data = self.session_data.get(session_id, {})
        heartbeat = HeartbeatResponse(
            type=MessageType.HEARTBEAT,
            rtt_ms=100,  # Mock RTT
            load_hint=LoadHint.NORMAL,
            last_seq=session_data.get("last_seq", 0)
        )
        
        message = {
            "type": heartbeat.type.value,
            "rtt_ms": heartbeat.rtt_ms,
            "load_hint": heartbeat.load_hint.value,
            "last_seq": heartbeat.last_seq
        }
        
        await self.send_message(session_id, message)
        log_debug("ws_heartbeat", f"Heartbeat sent", session_id=session_id)


# Global WebSocket manager
ws_manager = WebSocketManager()


async def guidance_websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for guidance communication with YOLO integration."""
    session_id = await ws_manager.connect(websocket)
    guidance_engine = container.get_guidance_engine()
    guidance_engine.reset_session_state()  # Reset for new session
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat_loop(session_id))
    
    # Start session timeout task
    timeout_task = asyncio.create_task(session_timeout_handler(session_id))
    
    try:
        while True:
            # Wait for message with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                error_msg = create_ws_error_response(
                    ErrorCode.WS_TIMEOUT,
                    "WebSocket connection timed out"
                )
                await ws_manager.send_message(session_id, error_msg)
                break
            
            # Reset timeout task since we received data
            timeout_task.cancel()
            timeout_task = asyncio.create_task(session_timeout_handler(session_id))
            
            # Handle different message types
            if data["type"] == "websocket.receive":
                if "text" in data:
                    # Handle JSON messages (frame_meta, capture, cancel)
                    await handle_json_message(session_id, data["text"], guidance_engine)
                elif "bytes" in data:
                    # Handle JPEG frame data
                    await handle_binary_message(session_id, data["bytes"], guidance_engine)
    
    except WebSocketDisconnect:
        log_info("ws_disconnect", "Client disconnected", session_id=session_id)
    except Exception as e:
        log_error("WS_ERROR", f"WebSocket error: {str(e)}", session_id=session_id)
        error_msg = create_ws_error_response(
            ErrorCode.SERVER_ERROR,
            "Internal server error"
        )
        await ws_manager.send_message(session_id, error_msg)
    finally:
        heartbeat_task.cancel()
        timeout_task.cancel()
        ws_manager.disconnect(session_id)


async def handle_json_message(session_id: str, message: str, guidance_engine):
    """Handle JSON WebSocket messages."""
    try:
        data = json.loads(message)
        msg_type = data.get("type")
        
        if msg_type == MessageType.FRAME_META.value:
            await handle_frame_meta(session_id, data)
        elif msg_type == MessageType.CAPTURE.value:
            await handle_capture_request(session_id, data, guidance_engine)
        elif msg_type == MessageType.CANCEL.value:
            await handle_cancel_request(session_id, data)
        else:
            error_msg = create_ws_error_response(
                ErrorCode.INVALID_MSG,
                f"Unknown message type: {msg_type}"
            )
            await ws_manager.send_message(session_id, error_msg)
    
    except json.JSONDecodeError:
        error_msg = create_ws_error_response(
            ErrorCode.INVALID_MSG,
            "Invalid JSON format"
        )
        await ws_manager.send_message(session_id, error_msg)


async def handle_final_capture(session_id: str, final_result: Dict[str, Any], guidance_engine):
    """Handle final frame capture and keep WebSocket open for processing."""
    try:
        # Log final candidate detection
        log_info("final_candidate", f"Class 5 dominance detected for final capture", session_id=session_id)
        
        # Send final_frame_captured message
        final_message = {
            "type": "final_frame_captured",
            "sessionId": session_id,
            "saved_path": final_result.get("save_path")  # Can be None if save failed
        }
        
        await ws_manager.send_message(session_id, final_message)
        log_info("final_signal_sent", f"Final capture signal sent", session_id=session_id)
        
        # Send loading message to notify client UI to show loading state
        loading_message = {
            "type": "loading",
            "message": "Processing the final frame"
        }
        await ws_manager.send_message(session_id, loading_message)
        
        # Update session state - keep WebSocket open but stop guidance and frame intake
        if session_id in ws_manager.session_data:
            session_data = ws_manager.session_data[session_id]
            session_data["mode"] = "processing_pending"
            session_data["accepting_frames"] = False
            session_data["guidance_enabled"] = False
            session_data["frames_dropped_after_final"] = 0
            
        log_info("final_selected", 
                f"Best frame selected and saved: path={final_result.get('save_path')}, "
                f"file_size={len(final_result.get('best_frame_jpeg', b''))}, "
                f"conf={final_result.get('confidence', 0.0)}", 
                session_id=session_id)
        
        # Log finalization complete - WebSocket stays open
        log_info("finalization_complete", 
                f"Finalization complete - WebSocket kept open for processing", 
                session_id=session_id)
                
    except Exception as e:
        log_error("final_capture_error", f"Final capture handling failed: {str(e)}", session_id=session_id)
        # Send error message to client but still finalize session
        error_msg = create_ws_error_response(
            ErrorCode.SERVER_ERROR,
            "final_save_failed"
        )
        await ws_manager.send_message(session_id, error_msg)
        
        # Still finalize the session even if save failed
        if session_id in ws_manager.session_data:
            session_data = ws_manager.session_data[session_id]
            session_data["mode"] = "processing_pending"
            session_data["accepting_frames"] = False
            session_data["guidance_enabled"] = False
            session_data["frames_dropped_after_final"] = 0


async def wait_for_ack(session_id: str):
    """Wait for ACK message from client."""
    # This would need to be implemented with a proper message queue
    # For now, we'll use a simple timeout approach
    await asyncio.sleep(ACK_TIMEOUT_MS / 1000.0)


async def handle_binary_message(session_id: str, frame_data: bytes, guidance_engine):
    """Handle binary JPEG frame data with YOLO processing."""
    try:
        # Check if session is still accepting frames
        session_data = ws_manager.session_data.get(session_id, {})
        if not session_data.get("accepting_frames", True):
            # Increment dropped frame counter
            if session_id in ws_manager.session_data:
                ws_manager.session_data[session_id]["frames_dropped_after_final"] += 1
                dropped_count = ws_manager.session_data[session_id]["frames_dropped_after_final"]
                
                # Log first drop, then sample every 10 drops or every 5 seconds
                if dropped_count == 1:
                    log_info("frame_ignored_after_final", "First frame dropped after finalization", session_id=session_id, reason="finalized")
                elif dropped_count % 10 == 0 or dropped_count == 5:
                    log_info("frame_ignored_after_final", f"Frame dropped after finalization (count: {dropped_count})", session_id=session_id, reason="finalized")
            return
        
        # Check rate limiting
        if not ws_manager.check_rate_limit(session_id):
            error_msg = create_ws_error_response(
                ErrorCode.RATE_LIMIT,
                f"Frame rate exceeded {MAX_FPS} FPS limit"
            )
            await ws_manager.send_message(session_id, error_msg)
            return
        
        # Check frame size limit
        if len(frame_data) > MAX_JPEG_KB * 1024:
            error_msg = create_ws_error_response(
                ErrorCode.FRAME_TOO_LARGE,
                f"Frame size {len(frame_data)} exceeds limit {MAX_JPEG_KB}KB"
            )
            await ws_manager.send_message(session_id, error_msg)
            return
        
        # Check if guidance is enabled for this session
        if not session_data.get("guidance_enabled", True):
            log_debug("ws_guidance_disabled", "Guidance disabled - session in processing mode", session_id=session_id)
            return
            
        # Check if we should send guidance (respect CADENCE_MS)
        if not ws_manager.should_send_guidance(session_id):
            return
        
        # Validate frame data
        if len(frame_data) < 100:  # Minimum reasonable JPEG size
            error_msg = create_ws_error_response(
                ErrorCode.INVALID_FORMAT,
                "Frame data too small to be valid JPEG"
            )
            await ws_manager.send_message(session_id, error_msg)
            return
        
        # JPEG sanity check - attempt to decode the frame
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(frame_data, np.uint8)
            decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded_image is None:
                # Invalid JPEG - send error but continue session
                error_msg = create_ws_error_response(
                    ErrorCode.INVALID_MSG,
                    "invalid JPEG frame"
                )
                await ws_manager.send_message(session_id, error_msg)
                log_info("frame_decode_failed", "Invalid JPEG frame received", session_id=session_id)
                return
        except Exception as e:
            # Decode error - send error but continue session
            error_msg = create_ws_error_response(
                ErrorCode.INVALID_MSG,
                "invalid JPEG frame"
            )
            await ws_manager.send_message(session_id, error_msg)
            log_info("frame_decode_failed", f"JPEG decode error: {str(e)}", session_id=session_id)
            return
        
        # Mark as processing
        if session_id in ws_manager.session_data:
            ws_manager.session_data[session_id]["processing"] = True
        
        # Process frame with YOLO guidance engine
        session_data = ws_manager.session_data.get(session_id, {})
        metadata = {
            "session_id": session_id, 
            "frame_size": len(frame_data),
            "frame_count": session_data.get("frame_count", 0)
        }
        
        # Get expected class from session data
        expected_class = session_data.get("expected_class")
        
        try:
            guidance_response = await guidance_engine.analyze_frame(frame_data, metadata, expected_class)
            
            # Send guidance response
            response_data = {
                "type": guidance_response.type.value,
                "class": guidance_response.dir.value,  # Changed from "dir" to "class"
                "coverage": guidance_response.coverage,
                "skew_deg": guidance_response.skew_deg,
                "conf": guidance_response.conf,
                "ready": guidance_response.ready
            }
            
            await ws_manager.send_message(session_id, response_data)
            
            # Check for final capture after guidance is sent
            final_result = guidance_engine.maybe_finalize(session_id)
            if final_result["triggered"]:
                await handle_final_capture(session_id, final_result, guidance_engine)
                return  # Exit early, session will be closed
            
            # Update session data
            if session_id in ws_manager.session_data:
                session_data = ws_manager.session_data[session_id]
                session_data["frame_count"] = session_data.get("frame_count", 0) + 1
                session_data["processing"] = False
            
            log_debug("ws_frame_processed", 
                     f"Frame processed: dir={guidance_response.dir.value}, ready={guidance_response.ready}", 
                     session_id=session_id)
        
        except Exception as e:
            log_error("YOLO_PROCESSING_ERROR", f"YOLO processing failed: {str(e)}", session_id=session_id)
            error_msg = create_ws_error_response(
                ErrorCode.INFERENCE_ERROR,
                "Frame processing failed"
            )
            await ws_manager.send_message(session_id, error_msg)
        
        finally:
            # Mark processing as complete
            if session_id in ws_manager.session_data:
                ws_manager.session_data[session_id]["processing"] = False
    
    except Exception as e:
        log_error("FRAME_HANDLING_ERROR", f"Frame handling failed: {str(e)}", session_id=session_id)
        error_msg = create_ws_error_response(
            ErrorCode.SERVER_ERROR,
            "Frame handling failed"
        )
        await ws_manager.send_message(session_id, error_msg)


async def handle_frame_meta(session_id: str, data: Dict[str, Any]):
    """Handle frame metadata message."""
    try:
        frame_meta = FrameMetadata(
            type=MessageType.FRAME_META,
            seq=data["seq"],
            ts=data["ts"],
            w=data["w"],
            h=data["h"],
            rotation_degrees=data["rotationDegrees"],
            jpeg_quality=data["jpegQuality"]
        )
        
        # Update session data with latest sequence and expected class
        if session_id in ws_manager.session_data:
            ws_manager.session_data[session_id]["last_seq"] = frame_meta.seq
            # Extract expected class from frame_meta (support both "expected" and "expectedClass")
            expected_class = data.get("expected") or data.get("expectedClass")
            if expected_class is not None:
                try:
                    expected_class = int(expected_class)
                    if 0 <= expected_class <= 6:
                        ws_manager.session_data[session_id]["expected_class"] = expected_class
                    else:
                        log_debug("ws_frame_meta", f"Invalid expected class: {expected_class} (must be 0-6)", session_id=session_id)
                except (ValueError, TypeError):
                    log_debug("ws_frame_meta", f"Invalid expected class format: {expected_class}", session_id=session_id)
        
        log_debug("ws_frame_meta", f"Frame meta received: seq={frame_meta.seq}", session_id=session_id)
    
    except KeyError as e:
        error_msg = create_ws_error_response(
            ErrorCode.INVALID_MSG,
            f"Missing required field: {str(e)}"
        )
        await ws_manager.send_message(session_id, error_msg)


async def handle_capture_request(session_id: str, data: Dict[str, Any], guidance_engine):
    """Handle capture request message with best frame selection."""
    try:
        capture_req = CaptureRequest(
            type=MessageType.CAPTURE,
            reason=data["reason"],
            best_seq=data.get("best_seq")
        )
        
        log_info("ws_capture", f"Capture requested: {capture_req.reason}", session_id=session_id)
        
        # Get best frame from YOLO engine
        best_frame_info = None
        if hasattr(guidance_engine, 'freeze_best_frame'):
            best_frame_info = guidance_engine.freeze_best_frame()
        
        # Send OCR progress simulation
        progress_stages = [
            ("text_detect", 25),
            ("text_recog", 50), 
            ("layout_analysis", 75),
            ("finalization", 100)
        ]
        
        for stage, pct in progress_stages:
            await asyncio.sleep(0.3)  # Simulate processing time
            progress_msg = {
                "type": MessageType.OCR_PROGRESS.value,
                "stage": stage,
                "pct": pct
            }
            await ws_manager.send_message(session_id, progress_msg)
        
        # Generate document ID and send OCR completion
        doc_id = str(uuid.uuid4())
        ocr_done_msg = {
            "type": MessageType.OCR_DONE.value,
            "docId": doc_id,
            "pages": 1,
            "lang": "en",
            "handoff": "internal",
            "meta": {
                "w": 800, 
                "h": 600,
                "best_frame_score": best_frame_info.get("score", 0.0) if best_frame_info else 0.0,
                "total_frames_processed": ws_manager.session_data.get(session_id, {}).get("frame_count", 0)
            }
        }
        await ws_manager.send_message(session_id, ocr_done_msg)
        
        log_info("ws_capture", f"Capture completed: docId={doc_id}", session_id=session_id)
    
    except KeyError as e:
        error_msg = create_ws_error_response(
            ErrorCode.INVALID_MSG,
            f"Missing required field: {str(e)}"
        )
        await ws_manager.send_message(session_id, error_msg)


async def handle_cancel_request(session_id: str, data: Dict[str, Any]):
    """Handle cancel request message."""
    log_info("ws_cancel", "Processing cancelled", session_id=session_id)
    
    # Mark processing as cancelled
    if session_id in ws_manager.session_data:
        ws_manager.session_data[session_id]["processing"] = False


async def heartbeat_loop(session_id: str):
    """Send periodic heartbeat messages."""
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_SEC)
            if session_id in ws_manager.active_connections:
                await ws_manager.send_heartbeat(session_id)
            else:
                break
    except asyncio.CancelledError:
        log_debug("heartbeat_cancelled", "Heartbeat loop cancelled", session_id=session_id)


async def session_timeout_handler(session_id: str):
    """Handle session timeout after inactivity."""
    try:
        await asyncio.sleep(TIMEOUT_SEC)
        
        # Check if session is still active and not processing
        if session_id in ws_manager.session_data:
            session = ws_manager.session_data[session_id]
            if not session.get("processing", False):
                log_info("ws_timeout", "Session timed out due to inactivity", session_id=session_id)
                error_msg = create_ws_error_response(
                    ErrorCode.WS_TIMEOUT,
                    "Session timeout due to inactivity"
                )
                await ws_manager.send_message(session_id, error_msg)
                
                # Close the connection
                if session_id in ws_manager.active_connections:
                    websocket = ws_manager.active_connections[session_id]
                    await websocket.close()
    
    except asyncio.CancelledError:
        log_debug("timeout_cancelled", "Timeout handler cancelled", session_id=session_id)