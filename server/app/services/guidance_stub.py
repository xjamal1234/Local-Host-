import asyncio
from typing import Dict, Any
from ..domain.ports import IGuidanceEngine
from ..domain.dtos import GuidanceResponse
from ..domain.enums import GuidanceDirection, MessageType


class GuidanceEngineStub(IGuidanceEngine):
    """Stub implementation of guidance engine for testing."""
    
    def __init__(self):
        self.ready = True
    
    async def analyze_frame(self, frame_data: bytes, metadata: Dict[str, Any]) -> GuidanceResponse:
        """Return mock guidance response."""
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        return GuidanceResponse(
            type=MessageType.GUIDANCE,
            dir=GuidanceDirection.STEADY,
            magnitude=0.1,
            coverage=0.85,
            skew_deg=2.5,
            conf=0.9,
            ready=True
        )
    
    async def is_ready(self) -> bool:
        """Return readiness status."""
        return self.ready
