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
    language: Language = Language.en
    audio_format: str = "wav"  # Add this
    audio_base64_format: str = Field(..., alias="audio")  # Accept both
    test_description: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
        fields = {
            'audio_base64_format': {'alias': 'audio'}
        }

class DetectionResponse(BaseModel):
    """Response model for voice detection"""
    status: str
    result: str  # "AI_GENERATED" or "HUMAN"
    confidence: float
    language: str
    processing_time_ms: int
    features_extracted: int
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    status_code: int