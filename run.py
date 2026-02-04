import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # For development with auto-reload (watches api, core, ml directories):
    # uvicorn.run(
    #     "api.main:app",
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=True,
    #     reload_dirs=["api", "core", "ml"],
    #     log_level="info"
    # )
    
    # For production or testing (no reload):
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
