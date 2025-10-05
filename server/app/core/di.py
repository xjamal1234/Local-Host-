from typing import Dict, Any, TypeVar, Type
from ..domain.ports import IGuidanceEngine, IOcrEngine, ILayoutEngine, IChatEngine
from ..services.guidance_stub import GuidanceEngineStub
from ..services.ocr_stub import OcrEngineStub
from ..services.layout_chat_stub import LayoutEngineStub, ChatEngineStub
from ..engines.yolo import YoloGuidanceEngine
from .config import USE_STUBS
from .logger import log_info

T = TypeVar('T')


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._setup_services()
    
    def _setup_services(self):
        """Setup service registrations based on configuration."""
        if USE_STUBS:
            # Register stub implementations
            log_info("di_setup", "Using stub implementations")
            self.register("guidance_engine", GuidanceEngineStub())
            self.register("ocr_engine", OcrEngineStub())
            self.register("layout_engine", LayoutEngineStub())
            self.register("chat_engine", ChatEngineStub())
        else:
            # Register real implementations
            log_info("di_setup", "Using real implementations")
            self.register("guidance_engine", YoloGuidanceEngine())
            self.register("ocr_engine", OcrEngineStub())  # Still using stub for Phase 3
            self.register("layout_engine", LayoutEngineStub())  # Still using stub for Phase 3
            self.register("chat_engine", ChatEngineStub())  # Still using stub for Phase 3
    
    def register(self, service_name: str, implementation: Any):
        """Register a service implementation."""
        self._services[service_name] = implementation
    
    def get(self, service_name: str) -> Any:
        """Get a registered service."""
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")
        return self._services[service_name]
    
    def get_guidance_engine(self) -> IGuidanceEngine:
        """Get guidance engine implementation."""
        return self.get("guidance_engine")
    
    def get_ocr_engine(self) -> IOcrEngine:
        """Get OCR engine implementation."""
        return self.get("ocr_engine")
    
    def get_layout_engine(self) -> ILayoutEngine:
        """Get layout engine implementation."""
        return self.get("layout_engine")
    
    def get_chat_engine(self) -> IChatEngine:
        """Get chat engine implementation."""
        return self.get("chat_engine")


# Global DI container instance
container = DIContainer()
