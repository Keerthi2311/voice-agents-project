from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Agents Project - Day 2", 
    version="2.0.0",
    description="30 Days of Voice Agents Challenge - Now with TTS Integration!"
)

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    voice_id: str = "en-US-davis"  # Default voice
    speed: float = 1.0
    
class TTSResponse(BaseModel):
    success: bool
    audio_url: str = None
    message: str
    text: str
    voice_id: str
    day: int = 2

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the home page"""
    with open("templates/home.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/tts", response_class=HTMLResponse)
async def tts_page():
    """Serve the TTS Generator page"""
    with open("templates/tts.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/echo", response_class=HTMLResponse)
async def echo_page():
    """Serve the Echo Bot page"""
    with open("templates/echo.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/api/hello")
async def hello_api():
    """Sample API endpoint"""
    return {"message": "Hello from Voice Agents Backend!", "day": 2, "status": "success"}

@app.get("/api/voice-status")
async def voice_status():
    """API endpoint for voice agent status"""
    return {
        "voice_agent": "initialized",
        "features": ["text-to-speech", "speech-to-text", "voice-commands"],
        "day": 2,
        "challenge": "30 Days of Voice Agents",
        "new_features": ["REST TTS API", "Murf AI Integration", "Audio Generation"]
    }

@app.post("/api/tts/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using Murf AI TTS API
    
    - **text**: The text to convert to speech
    - **voice_id**: Voice identifier (default: en-US-davis)
    - **speed**: Speech speed (default: 1.0)
    """
    try:
        logger.info(f"TTS request received: {request.text[:50]}...")
        
        # Get API key from environment
        api_key = os.getenv("MURF_API_KEY")
        base_url = os.getenv("MURF_BASE_URL", "https://api.murf.ai/v1")
        
        if not api_key or api_key == "your_murf_api_key_here":
            # For demo purposes, return a mock response
            logger.warning("No valid API key found, returning mock response")
            return TTSResponse(
                success=True,
                audio_url="https://example.com/generated-audio-mock.mp3",
                message="Mock TTS generation successful! (Replace with real Murf API key)",
                text=request.text,
                voice_id=request.voice_id
            )
        
        # Prepare the request to Murf API
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": request.text,
            "voice_id": request.voice_id,
            "speed": request.speed,
            "format": "mp3"
        }
        
        # Make the API call to Murf
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/speech/generate",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("TTS generation successful")
                
                # Get audio URL from Murf's response format
                audio_url = result.get("audioFile", "")
                
                return TTSResponse(
                    success=True,
                    audio_url=audio_url,
                    message="TTS generation successful!",
                    text=request.text,
                    voice_id=request.voice_id
                )
            else:
                logger.error(f"Murf API error: {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Murf API error: {response.text}"
                )
                
    except httpx.TimeoutException:
        logger.error("TTS request timeout")
        raise HTTPException(status_code=408, detail="TTS request timeout")
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.get("/api/tts/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    return {
        "voices": [
            {"id": "en-US-cooper", "name": "Cooper (US Male)", "language": "en-US"},
            {"id": "en-US-samantha", "name": "Samantha (US Female)", "language": "en-US"},
            {"id": "en-UK-hazel", "name": "Hazel (UK Female)", "language": "en-UK"},
            {"id": "en-AU-ivy", "name": "Ivy (AU Female)", "language": "en-AU"},
            {"id": "en-US-wayne", "name": "Wayne (US Male)", "language": "en-US"}
        ],
        "default": "en-US-cooper",
        "day": 3
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
