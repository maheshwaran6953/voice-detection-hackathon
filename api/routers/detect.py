from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import time
import base64
import numpy as np
from ..models import DetectionRequest, DetectionResponse, Language
from core.audio_processor import AudioProcessor
from ml.voice_detector import VoiceDetector

router = APIRouter(prefix="/detect", tags=["Detection"])

# Valid API keys
VALID_API_KEYS = {
    "hackathon-key-2024": "team-hcl-guvi",
    "test-key": "test-team",
    "demo-key": "demo-team"
}

# Audio constraints (adjust based on hackathon requirements)
MAX_DURATION_SECONDS = 60  # Maximum audio duration
MIN_DURATION_SECONDS = 1   # Minimum audio duration  
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB maximum
MIN_SAMPLE_RATE = 8000     # Minimum sample rate
MAX_SAMPLE_RATE = 48000    # Maximum sample rate
MIN_AUDIO_RMS = 0.001       # Minimum audio volume (to detect silence)

# Initialize detector
detector = None

def get_detector():
    """Get or initialize the voice detector"""
    global detector
    if detector is None:
        detector = VoiceDetector(language="en")
    return detector

def validate_api_key(api_key: str):
    """Validate API key"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

def decode_base64_audio(base64_string: str) -> bytes:
    """
    Decode base64 audio string to bytes
    Handles both plain base64 and data URLs
    """
    try:
        # Check if it's a data URL
        if base64_string.startswith('data:audio/'):
            # Extract base64 part after comma
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64
        audio_bytes = base64.b64decode(base64_string)
        return audio_bytes
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 audio: {str(e)}"
        )

def validate_audio_size(audio_bytes: bytes):
    """Validate audio file size"""
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(audio_bytes) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large: {size_mb:.2f}MB. Maximum size is {max_mb:.0f}MB."
        )
    
    # Also check minimum size (at least 1KB)
    if len(audio_bytes) < 1024:
        raise HTTPException(
            status_code=400,
            detail="Audio file is too small (less than 1KB)."
        )

def validate_audio_constraints(audio_bytes: bytes, audio_data: np.ndarray, sample_rate: int):
    """Validate audio against constraints"""
    errors = []
    
    # 1. File size check (already done in validate_audio_size, but double-check)
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(audio_bytes) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        errors.append(f"File size {size_mb:.2f}MB exceeds maximum {max_mb:.0f}MB.")
    
    # 2. Duration check
    duration = len(audio_data) / sample_rate
    if duration > MAX_DURATION_SECONDS:
        errors.append(f"Audio duration {duration:.1f}s exceeds maximum {MAX_DURATION_SECONDS}s.")
    
    if duration < MIN_DURATION_SECONDS:
        errors.append(f"Audio duration {duration:.1f}s is too short. Minimum is {MIN_DURATION_SECONDS}s.")
    
    # 3. Sample rate check
    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        errors.append(f"Sample rate {sample_rate}Hz is outside acceptable range ({MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}Hz).")
    
    # 4. Check if audio is mostly silence
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < MIN_AUDIO_RMS:
        errors.append(f"Audio appears to be silent or has very low volume (RMS: {rms:.4f}).")
    
    # 5. Check for NaN or infinite values
    if np.any(np.isnan(audio_data)):
        errors.append("Audio contains NaN (not a number) values.")
    
    if np.any(np.isinf(audio_data)):
        errors.append("Audio contains infinite values.")
    
    # 6. Check audio range (should be between -1 and 1 after normalization)
    max_val = np.max(np.abs(audio_data))
    if max_val > 10:  # Allow some leeway for unnormalized audio
        errors.append(f"Audio has unusually high amplitude (max: {max_val:.2f}).")
    
    return errors, duration

@router.post("/", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Detect if voice is AI-generated or Human
    
    **Headers:**
    - X-API-Key: Your API key (required)
    
    **Request Body:**
    - audio: Base64-encoded MP3 audio string
    - language: Language code (ta, en, hi, ml, te) - default: en
    - test_description: Optional description for reference
    
    **Audio Constraints:**
    - Maximum duration: 60 seconds
    - Maximum file size: 10MB
    - Minimum duration: 1 second
    - Supported formats: MP3 (automatically converted)
    
    **Response:**
    - result: "AI_GENERATED" or "HUMAN"
    - confidence: Score between 0.0 and 1.0
    - language: Detected language
    - processing_time_ms: Time taken
    - features_extracted: Number of features analyzed
    
    **Example Request:**
    ```json
    {
        "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w==",
        "language": "en",
        "test_description": "Test audio"
    }
    ```
    """
    
    start_time = time.time()
    
    # Validate API key
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required in X-API-Key header"
        )
    validate_api_key(x_api_key)
    
    try:
        # Step 1: Decode base64 audio
        audio_bytes = decode_base64_audio(request.audio)
        
        # Step 2: Validate audio size
        validate_audio_size(audio_bytes)
        
        # Step 3: Load audio using processor
        processor = AudioProcessor(sample_rate=16000, max_duration=MAX_DURATION_SECONDS)
        
        # We assume MP3 format for hackathon (as per requirements)
        file_format = request.audio_format
        
        audio_data, sr = processor.load_audio_from_bytes(audio_bytes, file_format)
        
        # Step 4: Validate audio constraints
        constraint_errors, duration = validate_audio_constraints(audio_bytes, audio_data, sr)
        if constraint_errors:
            raise HTTPException(
                status_code=400,
                detail="; ".join(constraint_errors)
            )
        
        # Step 5: Validate audio quality
        is_valid, error_msg = processor.validate_audio(audio_data)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio: {error_msg}"
            )
        
        # Step 6: Normalize audio
        audio_data = processor.normalize_audio(audio_data)
        
        # Step 7: Get detector and extract features
        detector = get_detector()
        features = detector.extract_features(audio_data, sr)
        
        # Step 8: Make prediction
        result, confidence = detector.predict(features)
        
        # Step 9: Prepare response
        processing_time = int((time.time() - start_time) * 1000)
        
        return DetectionResponse(
            status="success",
            result=result,
            confidence=round(confidence, 4),
            language=request.language.value,
            processing_time_ms=processing_time,
            features_extracted=len(features),
            message=request.test_description or f"Voice classification completed. Audio: {duration:.2f}s"
        )
                
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio processing error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@router.get("/test")
async def test_detection():
    """Test endpoint to verify detection route is working"""
    return {
        "status": "healthy",
        "message": "Detection endpoint is working",
        "endpoints": {
            "POST /detect/": "Detect voice authenticity",
            "GET /detect/test": "This test endpoint"
        },
        "supported_languages": ["en", "ta", "hi", "ml", "te"],
        "supported_formats": ["mp3"],
        "input_format": "base64-encoded MP3 audio",
        "constraints": {
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
            "sample_rate_range": f"{MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz"
        }
    }

@router.get("/constraints")
async def get_constraints():
    """Get audio constraints information"""
    return {
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "min_duration_seconds": MIN_DURATION_SECONDS,
        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
        "min_sample_rate_hz": MIN_SAMPLE_RATE,
        "max_sample_rate_hz": MAX_SAMPLE_RATE,
        "min_audio_rms": MIN_AUDIO_RMS,
        "supported_languages": ["ta", "en", "hi", "ml", "te"]
    }