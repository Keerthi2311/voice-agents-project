from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import aiofiles
from pathlib import Path
import uuid
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="30 Days Voice Agents API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    voice_id: str = "en-US-natalie"  # Default voice (Murf SDK format)
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "day": 1}

@app.post("/generate-speech")
async def generate_speech(request: TTSRequest):
    """Generate speech using Murf TTS API - mapped to generate-audio"""
    return await generate_audio(request)

@app.post("/generate-audio")
async def generate_audio(request: TTSRequest):
    """Generate audio using Murf TTS SDK"""
    
    murf_api_key = os.getenv("MURF_API_KEY")
    if not murf_api_key or murf_api_key == "your_murf_api_key_here":
        # For demo purposes, return a mock response
        logger.warning("No valid API key found, returning mock response")
        return {
            "success": True,
            "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
            "text": request.text,
            "message": "Mock TTS generation successful! (Replace with real Murf API key)"
        }
    
    try:
        # Import Murf SDK
        from murf import Murf
        
        # Initialize Murf client
        client = Murf(api_key=murf_api_key)
        
        # Generate speech using the SDK
        response = client.text_to_speech.generate(
            text=request.text,
            voice_id=request.voice_id
        )
        
        logger.info("TTS generation successful with Murf SDK")
        
        return {
            "success": True,
            "audio_url": response.audio_file,
            "text": request.text,
            "message": "TTS generation successful with Murf SDK!"
        }
        
    except ImportError:
        logger.error("Murf SDK not installed")
        return {
            "success": False,
            "audio_url": "",
            "text": request.text,
            "message": "Murf SDK not installed. Run: pip install murf"
        }
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts/echo")
async def echo_with_murf_voice(audio: UploadFile = File(...)):
    """
    Day 7: Echo Bot v2 - Transcribe audio and replay with Murf voice
    
    - **audio**: Audio file to transcribe and echo back with Murf voice
    """
    try:
        logger.info("Day 7: Echo Bot v2 request received")
        
        # Validate file type
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Read audio content
        audio_content = await audio.read()
        logger.info(f"Audio file size: {len(audio_content)} bytes")
        
        # Step 1: Transcribe the audio using AssemblyAI
        logger.info("Step 1: Transcribing audio with AssemblyAI...")
        
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key or assemblyai_api_key == "your_assemblyai_api_key_here":
            raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")
        
        import assemblyai as aai
        aai.settings.api_key = assemblyai_api_key
        
        # Create transcriber and transcribe
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_content)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
        
        transcribed_text = transcript.text
        logger.info(f"Transcription successful: {transcribed_text[:100]}...")
        
        # Step 2: Generate Murf audio from transcription
        logger.info("Step 2: Generating Murf audio from transcription...")
        
        murf_api_key = os.getenv("MURF_API_KEY")
        if not murf_api_key or murf_api_key == "your_murf_api_key_here":
            raise HTTPException(status_code=500, detail="Murf API key not configured")
        
        from murf import Murf
        
        # Initialize Murf client
        client = Murf(api_key=murf_api_key)
        
        # Generate speech using Murf SDK with a nice voice
        response = client.text_to_speech.generate(
            text=transcribed_text,
            voice_id="en-US-cooper"  # Nice male voice
        )
        
        logger.info("Day 7: Echo Bot v2 completed successfully!")
        
        return {
            "success": True,
            "original_text": transcribed_text,
            "audio_url": response.audio_file,
            "voice_id": "en-US-cooper",
            "message": "Echo Bot v2: Your voice transcribed and spoken back by Murf AI!",
            "day": 7
        }
        
    except Exception as e:
        logger.error(f"Day 7 Echo Bot v2 failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Echo Bot v2 failed: {str(e)}")

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    """Upload and temporarily save audio file"""
    
    try:
        # Validate file type
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Generate unique filename
        file_extension = audio.filename.split('.')[-1] if '.' in audio.filename else 'webm'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = uploads_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # Get file stats
        file_stats = file_path.stat()
        
        return {
            "success": True,
            "filename": unique_filename,
            "original_filename": audio.filename,
            "content_type": audio.content_type,
            "size": file_stats.st_size,
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/transcribe/file")
async def transcribe_audio_file(audio: UploadFile = File(...)):
    """Transcribe audio file using AssemblyAI"""
    
    try:
        # Validate file type
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key or assemblyai_api_key == "your_assemblyai_api_key_here":
            # For demo purposes, return a mock response
            logger.warning("No valid AssemblyAI API key found, returning mock response")
            return {
                "success": True,
                "transcript": "This is a mock transcription. Please add your AssemblyAI API key to enable real transcription.",
                "confidence": 0.95,
                "status": "completed",
                "audio_duration": 3.5,
                "words_count": 15,
                "message": "Mock transcription completed (Replace with real AssemblyAI API key)"
            }
        
        # Read audio file content
        audio_content = await audio.read()
        
        try:
            import assemblyai as aai
            # Configure AssemblyAI
            aai.settings.api_key = assemblyai_api_key
            
            # Initialize AssemblyAI transcriber
            transcriber = aai.Transcriber()
            
            # Transcribe the audio data directly
            transcript = transcriber.transcribe(audio_content)
            
            # Check if transcription was successful
            if transcript.status == aai.TranscriptStatus.error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {transcript.error}"
                )
            
            # Return transcription results
            return {
                "success": True,
                "transcript": transcript.text,
                "confidence": transcript.confidence,
                "status": transcript.status.value,
                "audio_duration": transcript.audio_duration,
                "words_count": len(transcript.text.split()) if transcript.text else 0,
                "message": "Transcription completed successfully"
            }
        except ImportError:
            logger.warning("AssemblyAI not installed, using mock response")
            return {
                "success": True,
                "transcript": "AssemblyAI module not installed. Run: pip install assemblyai",
                "confidence": 0.95,
                "status": "completed",
                "audio_duration": 3.5,
                "words_count": 10,
                "message": "Mock transcription (AssemblyAI not installed)"
            }
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/api/tts/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    return {
        "voices": [
            {"id": "en-US-natalie", "name": "Natalie (US Female)", "language": "en-US"},
            {"id": "en-US-cooper", "name": "Cooper (US Male)", "language": "en-US"},
            {"id": "en-US-samantha", "name": "Samantha (US Female)", "language": "en-US"},
            {"id": "en-UK-hazel", "name": "Hazel (UK Female)", "language": "en-UK"},
            {"id": "en-AU-ivy", "name": "Ivy (AU Female)", "language": "en-AU"},
        ],
        "default": "en-US-natalie",
        "sdk": "murf-python-sdk",
        "day": 6
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
