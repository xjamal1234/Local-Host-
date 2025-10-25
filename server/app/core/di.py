from typing import Dict, Any, TypeVar, Type
from ..domain.ports import IGuidanceEngine, IOcrEngine, ILayoutEngine, IChatEngine
from ..services.guidance_stub import GuidanceEngineStub
from ..services.ocr_stub import OcrEngineStub
from ..services.layout_chat_stub import LayoutEngineStub, ChatEngineStub
from ..engines.yolo import YoloGuidanceEngine
from ..engines.ocr_model import OcrEngine
from ..use_cases.ocr_processing import OcrProcessingUseCase
from .config import USE_STUBS
from ..services.openai_responses_client import OpenAIResponsesClient
from ..use_cases.layout_understanding import RunLayoutUnderstandingUseCase
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
            
            # Setup OCR components
            from . import config
            langs = [s.strip() for s in config.OCR_LANGS.split(",") if s.strip()]
            ocr_engine = OcrEngine(langs=langs, gpu=None)  # Auto-detect GPU
            ocr_use_case = OcrProcessingUseCase(ocr_engine, config)
            self.register("ocr_engine", ocr_engine)
            self.register("ocr_uc", ocr_use_case)

            # Setup OpenAI client and layout understanding use case
            openai_client = OpenAIResponsesClient(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_API_BASE,
                responses_endpoint=config.OPENAI_RESPONSES_ENDPOINT,
                timeout_sec=config.OPENAI_TIMEOUT_SEC,
                max_retries=config.OPENAI_MAX_RETRIES,
            )
            layout_uc = RunLayoutUnderstandingUseCase(openai_client)
            self.register("openai_client", openai_client)
            self.register("layout_uc", layout_uc)
            
            # Setup chat intent extraction and JSON executor
            from ..services.openai_intents import OpenAIIntentsClient
            from ..core.chat.json_executor import JsonExecutor
            
            intents_client = OpenAIIntentsClient(
                api_key=config.OPENAI_API_KEY,
                model=config.CHAT_INTENT_MODEL,
                base_url=config.OPENAI_API_BASE,
                responses_endpoint=config.OPENAI_RESPONSES_ENDPOINT,
                timeout_sec=config.CHAT_INTENT_TIMEOUT_SEC,
                max_retries=config.OPENAI_MAX_RETRIES
            )
            json_executor = JsonExecutor(base_dir=config.FINAL_FRAME_DIR)
            
            self.register("intents_client", intents_client)
            self.register("json_executor", json_executor)
            
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
    
    def get_ocr_use_case(self) -> OcrProcessingUseCase:
        """Get OCR processing use case."""
        return self.get("ocr_uc")

    def get_openai_client(self) -> OpenAIResponsesClient:
        """Get OpenAI Responses API client."""
        return self.get("openai_client")

    def get_layout_uc(self) -> RunLayoutUnderstandingUseCase:
        """Get layout understanding use case."""
        return self.get("layout_uc")
    
    def get_intents_client(self):
        """Get OpenAI intents extraction client."""
        return self.get("intents_client")
    
    def get_json_executor(self):
        """Get JSON executor for chat."""
        return self.get("json_executor")


# Global DI container instance
container = DIContainer()
