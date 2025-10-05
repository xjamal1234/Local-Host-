# NOOR - Assistive Reading System

NOOR is an assistive reading system for blind/visually-impaired users, featuring voice-guided document capture and AI-powered content interaction.

## Phase 3 - YOLO Integration Complete ✅

This phase integrates real YOLO-CLS model for live guidance with WebSocket communication.

### 🎯 YOLO Features
- **Real-time Guidance**: YOLO-CLS classification model for document framing
- **GPU Acceleration**: Automatic CUDA detection with FP16 optimization
- **Rate Limiting**: 24 FPS limit with intelligent frame dropping
- **Best Frame Selection**: Automatic selection of optimal frames for OCR
- **WebSocket Integration**: 250ms guidance cadence with 3-second heartbeats

### 🧠 YOLO Model Integration
- **Model**: Custom YOLO-CLS trained for document positioning
- **Classes**: 6 classes (corners + framing states)
- **Preprocessing**: 640x640 resize with RGB normalization
- **Inference**: Async processing to avoid blocking WebSocket communication
- **Error Handling**: Comprehensive YOLO-specific error codes and recovery

## Phase 2 - Core Setup Complete ✅

This phase establishes the foundational Core Layer with:

### 🏗️ Architecture
- **Clean Architecture**: Domain-driven design with ports and adapters
- **Dependency Injection**: Simple container-based DI system  
- **Configuration Management**: Environment-based config with CUDA detection
- **Structured Logging**: Development and production logging with loguru
- **Error Handling**: Standardized error codes and responses

### 🔧 Core Components
- **Configuration** (`core/config.py`): Environment variables, CUDA detection
- **Logging** (`core/logger.py`): JSON logs for prod, pretty logs for dev
- **Error Handling** (`core/errors.py`): Standardized error codes and formats
- **Dependency Injection** (`core/di.py`): Service container and registration

### 🌐 API Endpoints
- **Health Check**: `GET /api/v1/healthz` - Server health status
- **Readiness Check**: `GET /api/v1/readiness` - Component readiness
- **Version Info**: `GET /api/v1/version` - Application version
- **WebSocket**: `WS /ws/guidance` - Real-time guidance communication

### 🔌 WebSocket Features
- **Real-time Guidance**: Frame analysis and positioning guidance
- **Heartbeat System**: 3-second heartbeat with connection monitoring
- **Message Validation**: Frame metadata and capture request handling
- **Mock Responses**: OCR progress simulation and completion notifications

### 🧩 Stub Implementations
All engines use stub implementations for Phase 2:
- **Guidance Engine**: Mock camera positioning guidance
- **OCR Engine**: Simulated text extraction
- **Layout Engine**: Mock document structuring  
- **Chat Engine**: Simulated document Q&A

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- conda (optional but recommended)
- NVIDIA GPU (optional, auto-detected)

### Installation
```bash
# Clone and navigate to project
cd NOOR/server

# Install dependencies
pip install fastapi uvicorn loguru websockets

# Run the server
python -m app.main
```

### Verification
```bash
# Test imports
python -c "from app.main import app; print('✅ NOOR server ready!')"

# Check configuration
python -c "from app.core.config import *; print(f'PORT={PORT}, CUDA={USE_CUDA}')"
```

## 📊 System Status

**Current Configuration:**
- Port: 8080
- CUDA Support: Auto-detected (False - toolkit not installed)
- YOLO Mode: Enabled (USE_STUBS=False)
- Rate Limiting: 24 FPS max
- Guidance Cadence: 250ms
- Environment: Development

**API Endpoints:**
- Health: http://localhost:8080/api/v1/healthz
- Docs: http://localhost:8080/docs
- WebSocket: ws://localhost:8080/ws/guidance

## 🔄 Next Phases

- **Phase 4**: OpenCV image processing and enhancement
- **Phase 5**: OCR implementation (Tesseract/PaddleOCR)
- **Phase 6**: GPT layout structuring and chat
- **Phase 7**: Performance optimization and production hardening

## 📁 Project Structure

```
server/app/
├── core/           # Configuration, logging, DI, errors
├── domain/         # DTOs, enums, and port interfaces  
├── interfaces/     # API routes and WebSocket handlers
├── use_cases/      # Business logic orchestration
├── services/       # Stub implementations
├── engines/        # ML engine implementations (future)
├── data/           # Firebase integration (future)
└── static/         # Static file serving
```

---
**NOOR Phase 3 - YOLO Integration Complete** 🎯

**Key Achievements:**
- ✅ Real YOLO-CLS model integration
- ✅ WebSocket rate limiting and frame management  
- ✅ GPU acceleration with FP16 optimization
- ✅ Best frame selection for OCR preparation
- ✅ Comprehensive error handling and logging
- ✅ Production-ready guidance pipeline

**Ready for Phase 4: OpenCV Image Processing** 🚀
