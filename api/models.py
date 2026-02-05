from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Language(str, Enum):
    en = "en"
    ta = "ta"
    hi = "hi"
    ml = "ml"
    te = "te"

class DetectionRequest(BaseModel):
    # For hackathon - make this REQUIRED
    language: Language = Language.en
    audio_format: str = "wav"
    audio_base64_format: str  # ⚠️ REMOVE Optional[str] = None
    audio: Optional[str] = None  # Keep original format optional
    test_description: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True

class DetectionResponse(BaseModel):
    status: str
    result: str
    confidence: float
    language: str
    processing_time_ms: int
    features_extracted: int
    message: Optional[str] = None
    audio_duration_seconds: Optional[float] = None