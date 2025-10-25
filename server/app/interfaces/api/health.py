from fastapi import APIRouter, HTTPException
from ...core.config import APP_VERSION, BUILD_HASH
from ...domain.dtos import HealthResponse, ReadinessResponse, VersionResponse
from ...core.di import container
from ...core.logger import log_info

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    log_info("health_check", "Health check requested")
    return HealthResponse(status="ok")


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness check endpoint - validates system components."""
    try:
        # Check if core services are available
        guidance_engine = container.get_guidance_engine()
        ocr_engine = container.get_ocr_engine()
        layout_engine = container.get_layout_engine()
        chat_engine = container.get_chat_engine()
        
        # Check readiness of all engines
        guidance_ready = await guidance_engine.is_ready()
        ocr_ready = await ocr_engine.is_available()
        layout_ready = await layout_engine.is_ready()
        chat_ready = await chat_engine.is_available()
        
        ready = all([guidance_ready, ocr_ready, layout_ready, chat_ready])
        
        log_info("readiness_check", f"System readiness: {ready}")
        
        return ReadinessResponse(ready=ready)
        
    except Exception as e:
        log_info("readiness_check", f"Readiness check failed: {str(e)}")
        return ReadinessResponse(ready=False)


@router.get("/version", response_model=VersionResponse)
async def version_info():
    """Version information endpoint."""
    log_info("version_info", f"Version info requested: {APP_VERSION}")
    return VersionResponse(version=APP_VERSION, build=BUILD_HASH)


@router.get("/gpu")
async def gpu_info():
    """GPU and CUDA status endpoint."""
    log_info("gpu_info", "GPU status requested")
    
    try:
        # Try to get GPU info from guidance engine if available
        guidance_engine = container.get_guidance_engine()
        
        # Check if the engine has a get_gpu_info method
        if hasattr(guidance_engine, 'get_gpu_info'):
            gpu_info_data = guidance_engine.get_gpu_info()
            return gpu_info_data
        
        # Fallback: basic CUDA check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                
                return {
                    "cuda_available": True,
                    "device": gpu_name,
                    "memory_total_gb": round(gpu_memory_total, 2),
                    "memory_allocated_gb": round(gpu_memory_allocated, 2),
                    "fp16_enabled": False  # Default to False since we disabled it
                }
            else:
                return {
                    "cuda_available": False,
                    "reason": "CUDA not available"
                }
        except ImportError:
            return {
                "cuda_available": False,
                "reason": "PyTorch not available"
            }
        except Exception as e:
            return {
                "cuda_available": False,
                "reason": f"CUDA check failed: {str(e)}"
            }
            
    except Exception as e:
        log_info("gpu_info", f"GPU info request failed: {str(e)}")
        return {
            "cuda_available": False,
            "reason": f"Error getting GPU info: {str(e)}"
        }
