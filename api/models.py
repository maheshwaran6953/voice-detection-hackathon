from pydantic import BaseModel 
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
    """Request model for voice detection""" 
    audio_url: str 
    language: Language = Language.ENGLISH 
    test_description: Optional[str] = None 
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
 
