from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
from dotenv import load_dotenv 
import os 
 
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
 
@app.get("/") 
def read_root(): 
    return {"message": "Voice Detection API", "status": "running"} 
 
@app.get("/health") 
def health_check(): 
    return {"status": "healthy", "service": "voice-detection"} 
