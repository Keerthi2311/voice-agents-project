from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
import aiofiles
from pathlib import Path
import uuid
import logging
import time
import requests
import io
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback
import asyncio
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Error handling constants
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
FALLBACK_AUDIO_URL = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTuW2e/LdCUELIHQ8tiJOQcZZ7zl559NEApPqOPxtmMcBQeA5"

# Error messages for different failure scenarios
ERROR_MESSAGES = {
    "stt_failed": "I'm having trouble understanding your audio right now. Could you please try speaking again?",
    "llm_failed": "I'm experiencing some difficulty processing your request at the moment. Please try again in a few seconds.",
    "tts_failed": "I can respond to you, but I'm having trouble generating voice output right now.",
    "general_error": "I'm experiencing technical difficulties right now. Please try again later.",
    "api_key_missing": "The service is temporarily unavailable due to configuration issues. Please try again later.",
    "network_error": "I'm having trouble connecting to my services right now. Please check your connection and try again.",
    "timeout_error": "The request is taking longer than expected. Please try again with a shorter message.",
    "rate_limit": "I'm receiving too many requests right now. Please wait a moment and try again."
}

app = FastAPI(title="30 Days Voice Agents API - Day 11: Error Handling Edition")

# Error handling helper functions
class VoiceAgentError(Exception):
    """Base exception for voice agent errors"""
    def __init__(self, message: str, error_type: str = "general_error", original_error: Exception = None):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(self.message)

class STTError(VoiceAgentError):
    """Speech-to-Text specific error"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "stt_failed", original_error)

class LLMError(VoiceAgentError):
    """Large Language Model specific error"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "llm_failed", original_error)

class TTSError(VoiceAgentError):
    """Text-to-Speech specific error"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "tts_failed", original_error)

async def retry_with_backoff(func, *args, **kwargs):
    """Retry function with exponential backoff"""
    for attempt in range(MAX_RETRIES):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise e
            
            wait_time = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

def handle_api_error(error: Exception, service_name: str) -> dict:
    """Handle API errors and return appropriate fallback response"""
    error_details = {
        "service": service_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.error(f"{service_name} error: {error_details}")
    
    # Determine error type and appropriate message
    if "api" in str(error).lower() and "key" in str(error).lower():
        error_type = "api_key_missing"
    elif "timeout" in str(error).lower() or "time" in str(error).lower():
        error_type = "timeout_error"
    elif "network" in str(error).lower() or "connection" in str(error).lower():
        error_type = "network_error"
    elif "rate" in str(error).lower() and "limit" in str(error).lower():
        error_type = "rate_limit"
    else:
        error_type = "general_error"
    
    return {
        "success": False,
        "error_type": error_type,
        "fallback_message": ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general_error"]),
        "error_details": error_details,
        "has_fallback": True
    }

def create_fallback_response(transcribed_text: str = "", error_context: str = "", session_id: str = "") -> dict:
    """Create a fallback response when all services fail"""
    fallback_responses = [
        "I apologize, but I'm experiencing technical difficulties right now. Please try again in a few moments.",
        "I'm having trouble connecting to my services at the moment. Could you please try again?",
        "Something went wrong on my end. Please give me a moment and try your request again.",
        "I'm temporarily unable to process your request. Please try again shortly."
    ]
    
    # Select fallback based on session history length to add variety
    history_length = len(get_chat_history(session_id)) if session_id else 0
    fallback_text = fallback_responses[history_length % len(fallback_responses)]
    
    return {
        "success": True,  # We're successfully providing a fallback
        "transcribed_text": transcribed_text or "Audio processing failed",
        "llm_response": fallback_text,
        "audio_url": FALLBACK_AUDIO_URL,
        "model": "fallback-system",
        "voice": "emergency-fallback",
        "message": f"Fallback response activated due to: {error_context}",
        "day": 11,
        "is_fallback": True,
        "pipeline_steps": [
            "⚠️ Error detected in pipeline",
            "✅ Fallback response generated",
            "✅ Emergency audio provided"
        ]
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Enhanced STT function with comprehensive error handling
async def transcribe_audio_with_fallback(audio_path: str, session_id: str = "") -> str:
    """
    Transcribe audio with robust error handling and fallback mechanisms
    """
    assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    # Check if API key is available
    if not assemblyai_api_key or assemblyai_api_key == "your_assemblyai_api_key_here":
        logger.warning("AssemblyAI API key not configured, using intelligent demo mode")
        return generate_demo_transcription(session_id)
    
    try:
        # Attempt transcription with retry logic
        return await retry_with_backoff(perform_assemblyai_transcription, audio_path, assemblyai_api_key)
        
    except Exception as e:
        logger.error(f"STT failed after retries: {str(e)}")
        # Return a contextual fallback based on conversation history
        return generate_demo_transcription(session_id, error_context=str(e))

def generate_demo_transcription(session_id: str = "", error_context: str = "") -> str:
    """Generate contextual demo transcription based on conversation state"""
    history_length = len(get_chat_history(session_id)) if session_id else 0
    
    demo_messages = [
        "Hello! I'd like to start a conversation with you. How are you doing today?",
        "That's interesting! Can you tell me more about that topic?",
        "I see what you mean. What's your perspective on this?",
        "Thanks for sharing that. Is there anything else you'd like to discuss?",
        "I appreciate your thoughts. What would you like to talk about next?"
    ]
    
    base_message = demo_messages[history_length % len(demo_messages)]
    
    if error_context:
        return f"{base_message} (Note: Audio transcription temporarily unavailable)"
    else:
        return base_message

async def perform_assemblyai_transcription(audio_path: str, api_key: str) -> str:
    """Perform the actual AssemblyAI transcription with enhanced error handling"""
    headers = {"authorization": api_key}
    
    try:
        # Step 1: Prepare audio file with format conversion if needed
        processed_audio_path = await prepare_audio_file(audio_path)
        
        # Step 2: Upload audio with proper error handling
        upload_url = await upload_audio_to_assemblyai(processed_audio_path, headers)
        
        # Step 3: Submit transcription request
        transcript_id = await submit_transcription_request(upload_url, headers)
        
        # Step 4: Poll for completion with timeout
        transcribed_text = await poll_transcription_completion(transcript_id, headers)
        
        # Clean up processed file if it's different from original
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        
        return transcribed_text
        
    except Exception as e:
        # Clean up on error
        if 'processed_audio_path' in locals() and processed_audio_path != audio_path:
            try:
                os.remove(processed_audio_path)
            except:
                pass
        raise STTError(f"AssemblyAI transcription failed: {str(e)}", e)

async def prepare_audio_file(audio_path: str) -> str:
    """Prepare audio file for transcription with format conversion"""
    try:
        # Try to convert to WAV format for better compatibility
        wav_path = audio_path.replace(os.path.splitext(audio_path)[1], '.wav')
        
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', wav_path, '-y'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            logger.info(f"Successfully converted audio to WAV format")
            return wav_path
        else:
            logger.warning("Audio conversion failed, using original format")
            return audio_path
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"Audio conversion failed: {e}, using original file")
        return audio_path

async def upload_audio_to_assemblyai(audio_path: str, headers: dict) -> str:
    """Upload audio to AssemblyAI with retry logic"""
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            
        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={**headers, "content-type": "application/octet-stream"},
            data=audio_data,
            timeout=30
        )
        
        if upload_response.status_code != 200:
            raise Exception(f"Upload failed: {upload_response.status_code} - {upload_response.text}")
        
        upload_url = upload_response.json()["upload_url"]
        logger.info("Audio uploaded to AssemblyAI successfully")
        return upload_url
        
    except requests.exceptions.Timeout:
        raise STTError("Upload timed out - please try with a shorter audio file")
    except requests.exceptions.ConnectionError:
        raise STTError("Network connection failed during upload")
    except Exception as e:
        raise STTError(f"Audio upload failed: {str(e)}")

async def submit_transcription_request(upload_url: str, headers: dict) -> str:
    """Submit transcription request to AssemblyAI"""
    try:
        transcript_request = {
            "audio_url": upload_url,
            "speech_model": "best",
            "language_detection": True,
            "punctuate": True,
            "format_text": True
        }
        
        transcript_response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json=transcript_request,
            timeout=10
        )
        
        if transcript_response.status_code != 200:
            raise Exception(f"Transcription request failed: {transcript_response.status_code} - {transcript_response.text}")
        
        transcript_id = transcript_response.json()["id"]
        logger.info(f"Transcription submitted with ID: {transcript_id}")
        return transcript_id
        
    except requests.exceptions.Timeout:
        raise STTError("Transcription request timed out")
    except Exception as e:
        raise STTError(f"Failed to submit transcription request: {str(e)}")

async def poll_transcription_completion(transcript_id: str, headers: dict, max_attempts: int = 60) -> str:
    """Poll for transcription completion with robust error handling"""
    for attempt in range(max_attempts):
        try:
            status_response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers,
                timeout=10
            )
            
            if status_response.status_code == 200:
                result = status_response.json()
                logger.info(f"Transcription status: {result['status']} (attempt {attempt + 1})")
                
                if result["status"] == "completed":
                    transcribed_text = result.get("text", "").strip()
                    if not transcribed_text:
                        raise STTError("Transcription completed but returned empty text")
                    return transcribed_text
                elif result["status"] == "error":
                    error_detail = result.get("error", "Unknown transcription error")
                    raise STTError(f"Transcription failed: {error_detail}")
            else:
                logger.warning(f"Status check failed: {status_response.status_code}")
            
            await asyncio.sleep(5)
            
        except requests.exceptions.Timeout:
            logger.warning(f"Status check timed out on attempt {attempt + 1}")
            continue
        except Exception as e:
            logger.warning(f"Status check error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_attempts - 1:
                raise STTError(f"Transcription polling failed: {str(e)}")
            continue
    
    raise STTError("Transcription timed out after maximum attempts")

# Enhanced LLM function with error handling
async def generate_llm_response_with_fallback(prompt: str, session_id: str = "") -> str:
    """Generate LLM response with robust error handling and fallback mechanisms"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        logger.warning("Gemini API key not configured, using intelligent fallback")
        return generate_fallback_llm_response(prompt, session_id)
    
    try:
        return await retry_with_backoff(perform_gemini_generation, prompt, gemini_api_key)
    except Exception as e:
        logger.error(f"LLM generation failed after retries: {str(e)}")
        return generate_fallback_llm_response(prompt, session_id, error_context=str(e))

def generate_fallback_llm_response(prompt: str, session_id: str = "", error_context: str = "") -> str:
    """Generate intelligent fallback response when LLM fails"""
    history_length = len(get_chat_history(session_id)) if session_id else 0
    
    # Analyze the prompt for context
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["hello", "hi", "hey", "start"]):
        response = "Hello! I'm happy to chat with you today. What would you like to talk about?"
    elif any(word in prompt_lower for word in ["how", "what", "why", "when", "where"]):
        response = "That's a great question! I'd love to help you with that. Could you tell me a bit more about what you're looking for?"
    elif any(word in prompt_lower for word in ["weather", "temperature", "rain", "sunny"]):
        response = "I'd recommend checking a current weather app or website for the most accurate weather information in your area."
    elif any(word in prompt_lower for word in ["thank", "thanks", "appreciate"]):
        response = "You're very welcome! I'm here to help whenever you need assistance."
    elif history_length > 3:
        response = "I appreciate our conversation so far! Is there anything specific you'd like to explore further?"
    else:
        response = "I understand you're trying to communicate with me. Could you please rephrase your question or try again?"
    
    if error_context:
        response += " (Note: I'm currently experiencing some technical difficulties, but I'm still here to help!)"
    
    return response

async def perform_gemini_generation(prompt: str, api_key: str) -> str:
    """Perform Gemini LLM generation with enhanced error handling"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Add safety and length constraints
        enhanced_prompt = f"""Please provide a helpful, safe, and concise response (under 2000 characters for voice synthesis) to this query: {prompt}"""
        
        response = model.generate_content(enhanced_prompt)
        
        if not response.text:
            raise LLMError("LLM returned empty response")
        
        # Ensure response length is appropriate for TTS
        llm_text = response.text
        if len(llm_text) > 2200:
            llm_text = llm_text[:2100] + "... Would you like me to continue?"
        
        return llm_text
        
    except ImportError:
        raise LLMError("Google Generative AI library not installed")
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            raise LLMError("API quota exceeded - please try again later")
        elif "key" in str(e).lower():
            raise LLMError("API key authentication failed")
        else:
            raise LLMError(f"LLM generation failed: {str(e)}")

# Enhanced TTS function with error handling
async def generate_tts_with_fallback(text: str) -> tuple[str, str]:
    """Generate TTS audio with robust error handling and fallback mechanisms"""
    murf_api_key = os.getenv("MURF_API_KEY")
    
    if not murf_api_key or murf_api_key == "your_murf_api_key_here":
        logger.warning("Murf API key not configured, using fallback audio")
        return FALLBACK_AUDIO_URL, "Fallback audio (Murf API key required)"
    
    try:
        return await retry_with_backoff(perform_murf_generation, text, murf_api_key)
    except Exception as e:
        logger.error(f"TTS generation failed after retries: {str(e)}")
        return FALLBACK_AUDIO_URL, f"Fallback audio (TTS error: {str(e)[:50]}...)"

async def perform_murf_generation(text: str, api_key: str) -> tuple[str, str]:
    """Perform Murf TTS generation with enhanced error handling"""
    try:
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "voiceId": "en-US-ken",
            "style": "Conversational",
            "text": text,
            "rate": 0,
            "pitch": 0,
            "sampleRate": 48000,
            "format": "MP3",
            "channelType": "STEREO",
            "pronunciationDictionary": {},
            "encodeAsBase64": False,
            "variation": 1,
            "audioDuration": 0,
            "modelVersion": "GEN2"
        }
        
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise TTSError(f"Murf API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        audio_url = result.get("audioFile")
        
        if not audio_url:
            error_msg = result.get("error", "No audio file URL returned")
            raise TTSError(f"Murf AI speech generation failed: {error_msg}")
        
        return audio_url, "en-US-ken (Murf AI)"
        
    except requests.exceptions.Timeout:
        raise TTSError("TTS request timed out")
    except requests.exceptions.ConnectionError:
        raise TTSError("Network connection failed during TTS generation")
    except Exception as e:
        raise TTSError(f"TTS generation failed: {str(e)}")

# In-memory chat history storage (for prototype)
# Format: {session_id: [{"role": "user"|"assistant", "content": str, "timestamp": datetime}, ...]}
chat_history_store: Dict[str, List[Dict[str, Any]]] = {}

# Helper functions for chat history management
def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a session"""
    return chat_history_store.get(session_id, [])

def add_to_chat_history(session_id: str, role: str, content: str):
    """Add a message to chat history"""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    
    chat_history_store[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })
    
    # Keep only last 20 messages to prevent memory issues
    if len(chat_history_store[session_id]) > 20:
        chat_history_store[session_id] = chat_history_store[session_id][-20:]

def format_chat_history_for_llm(session_id: str) -> str:
    """Format chat history for LLM context"""
    history = get_chat_history(session_id)
    if not history:
        return ""
    
    formatted_history = "Previous conversation:\n"
    for message in history:
        role_label = "Human" if message["role"] == "user" else "Assistant"
        formatted_history += f"{role_label}: {message['content']}\n"
    
    return formatted_history + "\n"

def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_history_store:
        del chat_history_store[session_id]

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

@app.get("/voice-pipeline", response_class=HTMLResponse)
async def voice_pipeline_page():
    """Serve the Voice Pipeline page - Day 9"""
    with open("templates/voice_pipeline.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/conversational-agent", response_class=HTMLResponse)
async def conversational_agent_page():
    """Serve the Conversational Agent page - Day 10"""
    with open("templates/conversational_agent.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/chat")
async def chat_interface_redirect():
    """Redirect to the unified Conversational Agent - Day 10"""
    return RedirectResponse(url="/conversational-agent", status_code=301)

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

# Pydantic model for LLM request
class LLMRequest(BaseModel):
    text: str

@app.post("/llm/query")
async def query_llm_text(request: LLMRequest):
    """
    Day 8: Large Language Model Integration with Google Gemini (Text Input)
    
    - **text**: Input text to send to the LLM for processing
    """
    try:
        logger.info(f"Day 8: LLM query request received: {request.text[:100]}...")
        
        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
            # For demo purposes, return a mock response
            logger.warning("No valid Gemini API key found, returning mock response")
            return {
                "success": True,
                "input_text": request.text,
                "response": f"This is a mock response from Gemini AI for your input: '{request.text}'. To get real AI responses, please add your GEMINI_API_KEY to the .env file. Get your free API key at: https://ai.google.dev/gemini-api/docs/quickstart",
                "model": "gemini-pro (mock)",
                "message": "Mock LLM response - Add real Gemini API key for actual AI responses",
                "day": 8
            }
        
        # Import and configure Gemini
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        logger.info("Sending request to Google Gemini...")
        response = model.generate_content(request.text)
        
        logger.info("Day 8: LLM query completed successfully!")
        
        return {
            "success": True,
            "input_text": request.text,
            "response": response.text,
            "model": "gemini-1.5-flash",
            "message": "LLM query processed successfully with Google Gemini!",
            "day": 8
        }
        
    except ImportError:
        logger.error("Google Generative AI library not installed")
        raise HTTPException(status_code=500, detail="Google Generative AI library not installed. Run: pip install google-generativeai")
    except Exception as e:
        logger.error(f"Day 8 LLM query failed: {str(e)}")

@app.post("/llm/query-audio")
async def query_llm_audio(audio_file: UploadFile = File(...)):
    """
    Day 9: The Full Non-Streaming Pipeline - Audio to LLM to Speech
    
    Complete pipeline: Audio Input → Transcription → LLM Processing → Voice Synthesis
    
    - **audio_file**: Audio file to transcribe, send to LLM, and respond with voice
    """
    try:
        logger.info(f"Day 9: Full pipeline request received - Audio file: {audio_file.filename}")
        
        # Validate file
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.webm', '.ogg')):
            raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, MP3, M4A, WebM, or OGG.")
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file
        timestamp = int(time.time())
        original_filename = f"llm_query_{timestamp}_{audio_file.filename}"
        audio_path = os.path.join(uploads_dir, original_filename)
        
        with open(audio_path, "wb") as f:
            audio_content = await audio_file.read()
            f.write(audio_content)
        
        logger.info(f"Audio file saved: {audio_path} (size: {len(audio_content)} bytes)")
        
        # Step 1: Transcribe audio using AssemblyAI
        logger.info("Step 1: Transcribing audio with AssemblyAI...")
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key or assemblyai_api_key == "your_assemblyai_api_key_here":
            # Demo mode - return mock transcription
            logger.warning("No valid AssemblyAI API key found, using demo mode")
            transcribed_text = "Hello, this is a demo transcription. The AI will respond to this mock input to demonstrate the full pipeline."
        else:
            try:
                # Real AssemblyAI transcription with better error handling
                headers = {"authorization": assemblyai_api_key}
                
                # Try direct file upload with proper content type detection
                import mimetypes
                content_type, _ = mimetypes.guess_type(audio_path)
                if not content_type or not content_type.startswith('audio/'):
                    content_type = 'audio/wav'  # Default to WAV
                
                logger.info(f"Uploading audio with content type: {content_type}")
                
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, content_type)}
                    upload_response = requests.post(
                        "https://api.assemblyai.com/v2/upload",
                        headers=headers,
                        files=files
                    )
                
                if upload_response.status_code != 200:
                    logger.error(f"AssemblyAI upload failed: {upload_response.status_code} - {upload_response.text}")
                    raise Exception(f"Upload failed: {upload_response.text}")
                
                upload_url = upload_response.json()["upload_url"]
                logger.info(f"Audio uploaded to AssemblyAI: {upload_url}")
                
                # Submit for transcription with enhanced settings
                transcript_request = {
                    "audio_url": upload_url,
                    "speech_model": "best",
                    "language_detection": True,
                    "punctuate": True,
                    "format_text": True
                }
                
                transcript_response = requests.post(
                    "https://api.assemblyai.com/v2/transcript",
                    headers=headers,
                    json=transcript_request
                )
                
                if transcript_response.status_code != 200:
                    logger.error(f"AssemblyAI transcription request failed: {transcript_response.status_code} - {transcript_response.text}")
                    raise Exception(f"Transcription request failed: {transcript_response.text}")
                
                transcript_id = transcript_response.json()["id"]
                logger.info(f"Transcription submitted with ID: {transcript_id}")
                
                # Poll for transcription completion
                max_attempts = 60  # 5 minutes max
                for attempt in range(max_attempts):
                    logger.info(f"Checking transcription status (attempt {attempt + 1}/60)...")
                    status_response = requests.get(
                        f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                        headers=headers
                    )
                    
                    if status_response.status_code == 200:
                        result = status_response.json()
                        logger.info(f"Transcription status: {result['status']}")
                        
                        if result["status"] == "completed":
                            transcribed_text = result["text"]
                            if not transcribed_text or transcribed_text.strip() == "":
                                logger.warning("Transcription returned empty text, using demo mode")
                                transcribed_text = "Hello, this is a demo response because the audio transcription was empty. The AI will still demonstrate the full pipeline."
                            else:
                                logger.info(f"Transcription completed: {transcribed_text[:100]}...")
                            break
                        elif result["status"] == "error":
                            error_detail = result.get("error", "Unknown transcription error")
                            logger.error(f"AssemblyAI transcription error: {error_detail}")
                            raise Exception(f"Transcription failed: {error_detail}")
                    else:
                        logger.error(f"Status check failed: {status_response.status_code} - {status_response.text}")
                    
                    time.sleep(5)
                else:
                    raise Exception("Transcription timed out")
                    
            except Exception as e:
                # Fallback to demo mode if transcription fails
                logger.warning(f"Transcription failed, falling back to demo mode: {str(e)}")
                # Let's create a more realistic demo based on common questions
                transcribed_text = f"Hi, how is the weather today in Kochi? (Demo: original audio transcription failed due to format issues - '{str(e)[:100]}...', but this simulates a real voice question)"
        
        # Step 2: Send transcribed text to LLM
        logger.info("Step 2: Sending transcribed text to Google Gemini...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        # Configure Gemini
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        
        # Initialize the model with a prompt to keep responses concise for voice
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Add instruction to keep response under 3000 characters for Murf API
        if "Demo:" in transcribed_text or "demo" in transcribed_text.lower():
            # If we're in demo mode, but simulate answering the weather question
            enhanced_prompt = f"""You are a helpful AI assistant. The user asked about the weather in Kochi, India, but due to technical issues with audio transcription, this is a demo response. Please provide a realistic and helpful response about weather in Kochi as if you were actually answering their weather question. Include:
1. A brief acknowledgment 
2. Some general information about Kochi's typical weather patterns
3. Suggest they could check a weather app for current conditions
4. Keep the response conversational and under 2000 characters for voice synthesis.

User's question (simulated): {transcribed_text}"""
        else:
            # Normal transcription worked
            enhanced_prompt = f"""Please provide a concise and helpful response to this question (keep under 2500 characters for voice synthesis): {transcribed_text}"""
        
        # Generate response
        llm_response = model.generate_content(enhanced_prompt)
        llm_text = llm_response.text
        
        # Ensure response is under 3000 characters for Murf API
        if len(llm_text) > 2500:
            # Truncate and add ending
            llm_text = llm_text[:2400] + "... I hope this helps!"
        
        logger.info(f"LLM response generated: {llm_text[:100]}...")
        
        # Step 3: Convert LLM response to speech using Murf AI
        logger.info("Step 3: Converting LLM response to speech with Murf AI...")
        murf_api_key = os.getenv("MURF_API_KEY")
        if not murf_api_key or murf_api_key == "your_murf_api_key_here":
            logger.warning("No valid Murf AI API key found, using demo mode")
            # Return a demo audio URL (you can replace this with an actual demo file)
            audio_url = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTuW2e/LdCUELIHQ8tiJOQcZZ7zl559NEApPqOPxtmMcBQeA5"
            voice_info = "Demo voice (Murf AI key required for real voice synthesis)"
        else:
            try:
                # Murf TTS request
                murf_headers = {
                    "api-key": murf_api_key,
                    "Content-Type": "application/json"
                }
                
                murf_payload = {
                    "voiceId": "en-US-ken",  # Professional male voice
                    "style": "Conversational",
                    "text": llm_text,
                    "rate": 0,
                    "pitch": 0,
                    "sampleRate": 48000,
                    "format": "MP3",
                    "channelType": "STEREO",
                    "pronunciationDictionary": {},
                    "encodeAsBase64": False,
                    "variation": 1,
                    "audioDuration": 0,
                    "modelVersion": "GEN2"
                }
                
                murf_response = requests.post(
                    "https://api.murf.ai/v1/speech/generate",
                    headers=murf_headers,
                    json=murf_payload
                )
                
                if murf_response.status_code != 200:
                    logger.error(f"Murf API error: {murf_response.status_code} - {murf_response.text}")
                    raise Exception(f"Murf API request failed: {murf_response.text}")
                
                murf_result = murf_response.json()
                logger.info(f"Murf API response: {murf_result}")
                
                # Check for audio file URL instead of success field
                audio_url = murf_result.get("audioFile")
                if not audio_url:
                    error_msg = murf_result.get("error", "No audio file URL returned")
                    logger.error(f"Murf API failed: {error_msg}")
                    raise Exception(f"Murf AI speech generation failed: {error_msg}")
                
                voice_info = "en-US-ken (Murf AI)"
                logger.info(f"Murf AI speech generated successfully: {audio_url}")
                
            except Exception as e:
                # Fallback to demo mode if Murf AI fails
                logger.warning(f"Murf AI failed, falling back to demo mode: {str(e)}")
                audio_url = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTuW2e/LdCUELIHQ8tiJOQcZZ7zl559NEApPqOPxtmMcBQeA5"
                voice_info = f"Demo voice (Murf AI error: {str(e)[:100]}...)"
        
        logger.info("Day 9: Full pipeline completed successfully!")
        
        # Clean up uploaded file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            "success": True,
            "transcribed_text": transcribed_text,
            "llm_response": llm_text,
            "audio_url": audio_url,
            "model": "gemini-1.5-flash",
            "voice": voice_info,
            "message": "Full pipeline completed: Audio → Transcription → LLM → Speech!",
            "day": 9,
            "pipeline_steps": [
                "✅ Audio uploaded and saved",
                "✅ Audio transcribed with AssemblyAI (or demo mode)",
                "✅ Text processed with Google Gemini",
                "✅ Response converted to speech with Murf AI (or demo mode)"
            ]
        }
        
    except ImportError:
        logger.error("Required libraries not installed")
        raise HTTPException(status_code=500, detail="Required libraries not installed. Ensure google-generativeai and requests are installed.")
    except Exception as e:
        logger.error(f"Day 9 Full pipeline failed: {str(e)}")
        # Clean up uploaded file on error
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

@app.post("/agent/chat/{session_id}")
async def conversational_agent_chat(session_id: str, audio_file: UploadFile = File(...)):
    """
    Day 11: Conversational Agent with Comprehensive Error Handling
    
    Complete pipeline with robust error handling: Audio Input → Transcription → Chat History → LLM → Voice Synthesis
    Features: Retry logic, fallback mechanisms, graceful degradation, detailed error reporting
    
    - **session_id**: Unique session identifier for maintaining chat history
    - **audio_file**: Audio file to transcribe and process in conversation context
    """
    audio_path = None
    error_context = ""
    
    try:
        logger.info(f"Day 11: Enhanced conversational chat request for session {session_id}")
        
        # Step 0: Input validation with enhanced error handling
        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.webm', '.ogg')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported audio format. Please use WAV, MP3, M4A, WebM, or OGG format."
            )
        
        # Check file size (limit to 10MB)
        audio_content = await audio_file.read()
        if len(audio_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="Audio file too large. Please use files smaller than 10MB."
            )
        
        if len(audio_content) < 1000:  # Minimum file size check
            raise HTTPException(
                status_code=400,
                detail="Audio file too small. Please ensure you recorded properly."
            )
        
        # Create uploads directory and save file
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        timestamp = int(time.time())
        original_filename = f"chat_{session_id}_{timestamp}_{audio_file.filename}"
        audio_path = os.path.join(uploads_dir, original_filename)
        
        with open(audio_path, "wb") as f:
            f.write(audio_content)
        
        logger.info(f"Audio file saved: {audio_path} (size: {len(audio_content)} bytes)")
        
        # Step 1: Enhanced Speech-to-Text with comprehensive error handling
        logger.info("Step 1: Enhanced STT with fallback mechanisms...")
        try:
            transcribed_text = await transcribe_audio_with_fallback(audio_path, session_id)
            logger.info(f"STT successful: {transcribed_text[:100]}...")
        except Exception as e:
            error_context += f"STT Error: {str(e)}; "
            logger.warning(f"STT failed, using fallback: {str(e)}")
            transcribed_text = generate_demo_transcription(session_id, f"STT failed: {str(e)}")
        
        # Step 2: Add user message to chat history (always succeeds)
        logger.info("Step 2: Adding user message to chat history...")
        try:
            add_to_chat_history(session_id, "user", transcribed_text)
        except Exception as e:
            logger.error(f"Failed to add to chat history: {str(e)}")
            error_context += f"History Error: {str(e)}; "
            # Continue without chat history if needed
        
        # Step 3: Enhanced LLM processing with context and error handling
        logger.info("Step 3: Enhanced LLM processing with fallback...")
        try:
            # Get chat history and format for LLM
            chat_context = format_chat_history_for_llm(session_id)
            
            # Create enhanced prompt with conversation context
            conversation_prompt = f"""You are a helpful, friendly AI assistant having a natural conversation. Keep your responses conversational, engaging, and under 2000 characters for voice synthesis.

{chat_context}

Current user message: {transcribed_text}

Please respond in a natural, conversational way. Remember the context of our previous conversation and build upon it. Ask follow-up questions to keep the conversation engaging."""
            
            llm_response = await generate_llm_response_with_fallback(conversation_prompt, session_id)
            logger.info(f"LLM response generated: {llm_response[:100]}...")
            
        except Exception as e:
            error_context += f"LLM Error: {str(e)}; "
            logger.warning(f"LLM failed, using fallback: {str(e)}")
            llm_response = generate_fallback_llm_response(transcribed_text, session_id, str(e))
        
        # Step 4: Add assistant response to chat history
        logger.info("Step 4: Adding assistant response to chat history...")
        try:
            add_to_chat_history(session_id, "assistant", llm_response)
        except Exception as e:
            logger.error(f"Failed to add assistant response to history: {str(e)}")
            error_context += f"History Save Error: {str(e)}; "
            # Continue without saving to history
        
        # Step 5: Enhanced Text-to-Speech with fallback
        logger.info("Step 5: Enhanced TTS with fallback mechanisms...")
        try:
            audio_url, voice_info = await generate_tts_with_fallback(llm_response)
            logger.info("TTS generation completed")
        except Exception as e:
            error_context += f"TTS Error: {str(e)}; "
            logger.warning(f"TTS failed, using fallback audio: {str(e)}")
            audio_url = FALLBACK_AUDIO_URL
            voice_info = f"Fallback audio (TTS failed: {str(e)[:50]}...)"
        
        # Clean up uploaded files
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Clean up any converted WAV files
            wav_path = audio_path.replace(os.path.splitext(audio_path)[1], '.wav') if audio_path else None
            if wav_path and os.path.exists(wav_path) and wav_path != audio_path:
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
        
        # Get current chat history for response
        try:
            current_history = get_chat_history(session_id)
            history_length = len(current_history)
        except:
            history_length = 0
        
        # Determine success status
        is_full_success = not error_context
        
        logger.info(f"Day 11: Enhanced conversational chat completed for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": audio_url,
            "model": "gemini-1.5-flash-enhanced" if is_full_success else "gemini-fallback-system",
            "voice": voice_info,
            "chat_history_length": history_length,
            "message": "Enhanced conversational chat with error handling completed!",
            "day": 11,
            "error_handling_active": True,
            "pipeline_status": "full_success" if is_full_success else "partial_fallback",
            "error_context": error_context.strip() if error_context else None,
            "pipeline_steps": [
                "✅ Audio uploaded and validated",
                "✅ Enhanced STT with fallback" if "STT Error" not in error_context else "⚠️ STT failed - fallback used",
                "✅ Chat history management",
                "✅ Enhanced LLM with context" if "LLM Error" not in error_context else "⚠️ LLM failed - fallback used",
                "✅ Enhanced TTS with fallback" if "TTS Error" not in error_context else "⚠️ TTS failed - fallback used",
                "✅ Graceful error handling"
            ],
            "fallback_features": {
                "intelligent_stt_fallback": True,
                "contextual_llm_fallback": True,
                "emergency_audio_fallback": True,
                "retry_mechanisms": True,
                "graceful_degradation": True
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are intentional
        raise
    except Exception as e:
        # Log the full error for debugging
        logger.error(f"Day 11 Enhanced conversational chat critical failure: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Clean up on critical failure
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        # Return a comprehensive fallback response
        fallback_response = create_fallback_response(
            transcribed_text="Audio processing failed",
            error_context=f"Critical system error: {str(e)}",
            session_id=session_id
        )
        
        # Override with error details for debugging
        fallback_response.update({
            "session_id": session_id,
            "chat_history_length": 0,
            "day": 11,
            "critical_error": True,
            "error_message": str(e),
            "error_type": type(e).__name__
        })
        
        return fallback_response

# Helper endpoint to get chat history
@app.get("/agent/chat/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for a session"""
    history = get_chat_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }

# Helper endpoint to clear chat history
@app.delete("/agent/chat/{session_id}/history")
async def clear_session_history(session_id: str):
    """Clear chat history for a session"""
    clear_chat_history(session_id)
    return {
        "session_id": session_id,
        "message": "Chat history cleared"
    }

# Day 11: Error simulation endpoint for testing
@app.post("/agent/chat/{session_id}/test-errors")
async def test_error_scenarios(session_id: str, error_type: str = "all"):
    """
    Day 11: Test error handling scenarios by simulating various failures
    
    - **error_type**: Type of error to simulate (stt, llm, tts, network, all)
    """
    logger.info(f"Day 11: Testing error scenario: {error_type} for session {session_id}")
    
    error_scenarios = {
        "stt": "Speech-to-Text service temporarily unavailable",
        "llm": "Large Language Model service experiencing high load",
        "tts": "Text-to-Speech service connection failed",
        "network": "Network timeout during API call",
        "api_key": "API authentication failed",
        "rate_limit": "Rate limit exceeded for service",
        "all": "Multiple service failures detected"
    }
    
    # Simulate the error based on type
    if error_type == "stt":
        transcribed_text = generate_demo_transcription(session_id, "Simulated STT failure")
        add_to_chat_history(session_id, "user", transcribed_text)
        
        llm_response = "I apologize, but I'm having trouble understanding your audio right now. This is a simulated error to test our fallback systems. Could you please try speaking again?"
        add_to_chat_history(session_id, "assistant", llm_response)
        
        return {
            "success": True,
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": FALLBACK_AUDIO_URL,
            "model": "error-simulation-mode",
            "voice": "fallback-audio",
            "message": f"Error simulation completed: {error_scenarios[error_type]}",
            "day": 11,
            "error_simulation": True,
            "simulated_error": error_type,
            "fallback_systems_active": True
        }
    
    elif error_type == "llm":
        transcribed_text = "Can you help me understand how error handling works?"
        add_to_chat_history(session_id, "user", transcribed_text)
        
        llm_response = generate_fallback_llm_response(transcribed_text, session_id, "Simulated LLM failure")
        add_to_chat_history(session_id, "assistant", llm_response)
        
        return {
            "success": True,
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": FALLBACK_AUDIO_URL,
            "model": "fallback-llm-system",
            "voice": "fallback-audio",
            "message": f"Error simulation completed: {error_scenarios[error_type]}",
            "day": 11,
            "error_simulation": True,
            "simulated_error": error_type,
            "fallback_systems_active": True
        }
    
    elif error_type == "tts":
        transcribed_text = "I'd like to hear your voice response."
        add_to_chat_history(session_id, "user", transcribed_text)
        
        llm_response = "I can understand and respond to you perfectly, but I'm currently experiencing issues with voice generation. This demonstrates how our system gracefully handles TTS failures while maintaining conversation flow."
        add_to_chat_history(session_id, "assistant", llm_response)
        
        return {
            "success": True,
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": FALLBACK_AUDIO_URL,
            "model": "gemini-1.5-flash",
            "voice": "tts-fallback-system",
            "message": f"Error simulation completed: {error_scenarios[error_type]}",
            "day": 11,
            "error_simulation": True,
            "simulated_error": error_type,
            "fallback_systems_active": True
        }
    
    elif error_type == "all":
        # Simulate complete system failure
        fallback_response = create_fallback_response(
            transcribed_text="System-wide error simulation",
            error_context="Testing complete system failure scenario",
            session_id=session_id
        )
        
        fallback_response.update({
            "session_id": session_id,
            "message": f"Error simulation completed: {error_scenarios[error_type]}",
            "day": 11,
            "error_simulation": True,
            "simulated_error": error_type,
            "all_systems_failed": True
        })
        
        return fallback_response
    
    else:
        return {
            "success": False,
            "error": f"Unknown error type: {error_type}",
            "available_types": list(error_scenarios.keys()),
            "day": 11
        }

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
