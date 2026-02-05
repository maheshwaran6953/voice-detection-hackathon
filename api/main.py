from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import base64
import time
import json

app = FastAPI(
    title="Voice Detection API",
    version="1.0.0",
    description="HCL Guvi Hackathon - AI vs Human Voice Detection"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MAIN HACKATHON ENDPOINT
@app.post("/detect")
async def detect_voice(
    request_data: dict = Body(...),  # Accept raw JSON dict
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Hackathon Voice Detection Endpoint
    """
    # Validate API key
    if x_api_key != "hackathon-key-2024":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Extract fields from request data
    audio_base64_format = request_data.get("audio_base64_format")
    language = request_data.get("language", "en")
    audio_format = request_data.get("audio_format", "wav")
    
    # Validate required field
    if not audio_base64_format:
        raise HTTPException(status_code=400, detail="audio_base64_format is required")
    
    start_time = time.time()
    
    try:
        # Decode and validate base64
        audio_bytes = base64.b64decode(audio_base64_format)
        
        # Validate audio size (10MB max)
        if len(audio_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
        
        if len(audio_bytes) < 1024:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Your ML model would process here
        # For now, return mock response
        processing_time = int((time.time() - start_time) * 1000)
        
        # Mock detection (replace with your actual model)
        import random
        result = "HUMAN" if random.random() > 0.3 else "AI_GENERATED"
        confidence = random.uniform(0.75, 0.95)
        
        return {
            "status": "success",
            "result": result,
            "confidence": round(confidence, 4),
            "language": language,
            "processing_time_ms": processing_time,
            "features_extracted": 156,
            "message": f"Voice detection completed - {audio_format} format",
            "audio_duration_seconds": 2.5
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

# Test endpoint
@app.get("/detect/test")
async def test_detection():
    return {
        "status": "healthy",
        "message": "Detection endpoint is working",
        "endpoint": "POST /detect",
        "supported_formats": ["wav", "mp3"],
        "constraints": {
            "max_duration_seconds": 60,
            "max_file_size_mb": 10
        }
    }

@app.get("/")
def read_root():
    return {"message": "Voice Detection API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "voice-detection"}