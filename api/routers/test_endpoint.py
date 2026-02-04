from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import time
import base64
from ..models import DetectionResponse
from core.audio_processor import AudioProcessor

router = APIRouter(prefix="/test", tags=["Test"])

VALID_API_KEYS = {"hackathon-key-2024": "team-hcl-guvi"}

def validate_api_key(api_key: str):
    """Validate API key"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

@router.post("/detect-base64", response_model=DetectionResponse)
async def detect_voice_base64(
    audio_base64: str,
    language: str = "en",
    test_description: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Detect voice from base64 encoded audio (for testing without internet)
    
    **Headers:**
    - X-API-Key: Your API key
    
    **Form Data:**
    - audio_base64: Base64 encoded audio string
    - language: Language code (ta, en, hi, ml, te)
    - test_description: Optional description
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
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Process audio
        processor = AudioProcessor()
        features = processor.process_audio_bytes(audio_bytes)
        
        # Mock prediction
        zero_crossing = features.get('zero_crossing_rate', 0)
        energy = features.get('energy', 0)
        
        # Normalize features
        zero_crossing_norm = min(zero_crossing * 100, 1.0)
        energy_norm = min(energy / 10000, 1.0)
        
        # Weighted score
        human_score = (zero_crossing_norm * 0.6) + (energy_norm * 0.4)
        ai_score = 1.0 - human_score
        
        if human_score > ai_score:
            result = "HUMAN"
            confidence = round(human_score, 4)
        else:
            result = "AI_GENERATED"
            confidence = round(ai_score, 4)
        
        # Prepare response
        processing_time = int((time.time() - start_time) * 1000)
        
        return DetectionResponse(
            status="success",
            result=result,
            confidence=confidence,
            language=language,
            processing_time_ms=processing_time,
            features_extracted=len(features),
            message=test_description or "Detection completed from base64 audio"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )