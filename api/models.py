from pydantic import BaseModel, Field
from typing import Optional, Union
from enum import Enum

class Language(str, Enum):
    en = "en"
    ta = "ta"
    hi = "hi"
    ml = "ml"
    te = "te"

class DetectionRequest(BaseModel):
    # Accept both field names for hackathon compatibility
    language: Language = Language.en
    audio_format: str = "wav"
    audio_base64_format: Optional[str] = None  # Hackathon format
    audio: Optional[str] = None  # Original format
    test_description: Optional[str] = None
    
    # Custom validator to ensure we have at least one audio field
    def get_audio_data(self):
        """Get audio data from either field"""
        if self.audio_base64_format:
            return self.audio_base64_format
        elif self.audio:
            return self.audio
        else:
            raise ValueError("Either 'audio' or 'audio_base64_format' field is required")
    
    class Config:
        allow_population_by_field_name = True
        extra = "allow"  # Allow extra fields for flexibility
    
    class DetectionResponse(BaseModel):
        status: str
        result: str  # "AI_GENERATED" or "HUMAN"
        confidence: float
        language: str
        processing_time_ms: int
        features_extracted: int
        message: Optional[str] = None
        audio_duration_seconds: Optional[float] = None