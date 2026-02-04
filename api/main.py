from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
from dotenv import load_dotenv 
import os 
from .routers import detect 
 
load_dotenv() 
 
app = FastAPI( 
    title=os.getenv("API_TITLE", "Voice Detection API"), 
    version=os.getenv("API_VERSION", "1.0.0"), 
    description="HCL Guvi Hackathon - AI vs Human Voice Detection" 
) 
 
# Allow CORS for frontend testing 
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
) 
 
# Include routers 
app.include_router(detect.router) 
 
@app.get("/") 
def read_root(): 
    return {"message": "Voice Detection API", "status": "running"} 
 
@app.get("/health") 
def health_check(): 
    return {"status": "healthy", "service": "voice-detection"} 

# In api/main.py, add after app initialization
@app.on_event("startup")
async def startup_event():
    """Initialize and save model on startup"""
    from ml.voice_detector import VoiceDetector
    
    print("Initializing voice detector...")
    detector = VoiceDetector()
    
    # Save the trained model
    import os
    model_dir = "ml/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "voice_detector_model.joblib")
    detector.save_model(model_path)
    print(f"Model saved to {model_path}")