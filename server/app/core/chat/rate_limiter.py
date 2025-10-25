"""
Rate limiter for chat messages.

Implements a sliding window rate limiter to prevent abuse of the chat system.
Each session is tracked independently.
"""

import time
from collections import deque
from typing import Dict


class ChatRateLimiter:
    """Rate limiter for chat messages using sliding window."""
    
    def __init__(self, max_messages: int = 5, window_seconds: int = 10):
        """
        Initialize rate limiter.
        
        Args:
            max_messages: Maximum messages allowed per window
            window_seconds: Time window in seconds
        """
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        # session_id -> deque of timestamps
        self._sessions: Dict[str, deque] = {}
        # session_id -> bool (whether throttle message was sent in current window)
        self._throttle_sent: Dict[str, bool] = {}
    
    def check_rate_limit(self, session_id: str) -> bool:
        """
        Check if a session is within rate limits.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if within limits, False if rate limited
        """
        now = time.time()
        
        # Initialize session if new
        if session_id not in self._sessions:
            self._sessions[session_id] = deque()
            self._throttle_sent[session_id] = False
        
        # Remove old timestamps outside the window
        while self._sessions[session_id] and \
              (now - self._sessions[session_id][0]) > self.window_seconds:
            self._sessions[session_id].popleft()
            # Reset throttle flag when window resets
            if not self._sessions[session_id]:
                self._throttle_sent[session_id] = False
        
        # Check if under limit
        if len(self._sessions[session_id]) < self.max_messages:
            self._sessions[session_id].append(now)
            return True
        
        return False
    
    def should_send_throttle_message(self, session_id: str) -> bool:
        """
        Check if throttle message should be sent for this session.
        
        Only sends once per window to avoid spamming.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if throttle message should be sent, False otherwise
        """
        if session_id not in self._throttle_sent:
            return False
        
        if not self._throttle_sent[session_id]:
            self._throttle_sent[session_id] = True
            return True
        
        return False
    
    def cleanup_session(self, session_id: str):
        """
        Clean up data for a disconnected session.
        
        Args:
            session_id: Session to clean up
        """
        self._sessions.pop(session_id, None)
        self._throttle_sent.pop(session_id, None)

