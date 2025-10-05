from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import PORT, WS_PATH, ALLOWED_ORIGINS, APP_VERSION
from app.core.logger import log_info, setup_logger
from app.interfaces.api.health import router as health_router
from app.interfaces.api.guidance_test import router as guidance_test_router
from app.interfaces.ws.guidance import guidance_websocket_endpoint

# Initialize logger
logger = setup_logger()

# Create FastAPI app
app = FastAPI(
    title="NOOR Assistive Reading System",
    description="Backend server for blind/visually-impaired reading assistance",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS.split(",") if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(guidance_test_router, prefix="/api/v1/guidance", tags=["guidance"])

# Mount static files for final captures
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# WebSocket endpoint
@app.websocket(WS_PATH)
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time guidance."""
    await guidance_websocket_endpoint(websocket)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    log_info("app_startup", f"NOOR server starting on port {PORT}")
    log_info("app_startup", f"WebSocket endpoint: {WS_PATH}")
    log_info("app_startup", f"API docs available at: http://10.7.0.250:{PORT}/docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    log_info("app_shutdown", "NOOR server shutting down")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "NOOR Assistive Reading System",
        "version": APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/api/v1/healthz",
            "readiness": "/api/v1/readiness", 
            "version": "/api/v1/version",
            "gpu": "/api/v1/gpu",
            "guidance": "/api/v1/guidance/analyze",
            "websocket": WS_PATH,
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )
