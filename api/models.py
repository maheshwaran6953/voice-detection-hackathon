from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Language(str, Enum):
    """Supported languages for voice detection"""
    TAMIL = "ta"
    ENGLISH = "en"
    HINDI = "hi"
    MALAYALAM = "ml"
    TELUGU = "te"

class DetectionRequest(BaseModel):
    """Request model for voice detection - CORRECTED FOR HACKATHON"""
    audio: str = Field(..., description="Base64-encoded MP3 audio string")
    language: Language = Language.ENGLISH
    test_description: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w==",
                "language": "en",
                "test_description": "Test audio sample"
            }
        }
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