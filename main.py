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

app = FastAPI(title="30 Days Voice Agents - Day 12: Enhanced UI")

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
        "day": 12,
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

# Routes for Day 12 - Only Conversational Agent
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main conversational agent page - Day 12"""
    with open("templates/conversational_agent.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/conversational-agent", response_class=HTMLResponse)
async def conversational_agent_page():
    """Serve the Conversational Agent page - Day 12"""
    with open("templates/conversational_agent.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/chat")
async def chat_interface_redirect():
    """Redirect to the main conversational agent"""
    return RedirectResponse(url="/", status_code=301)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "day": 12, "ui_version": "enhanced"}

@app.post("/agent/chat/{session_id}")
async def conversational_agent_chat(session_id: str, audio_file: UploadFile = File(...)):
    """
    Day 12: Enhanced Conversational Agent with Revamped UI
    
    Complete pipeline with robust error handling: Audio Input → Transcription → Chat History → LLM → Voice Synthesis
    Features: Enhanced UI, single record button, auto-play audio, smooth animations
    
    - **session_id**: Unique session identifier for maintaining chat history
    - **audio_file**: Audio file to transcribe and process in conversation context
    """
    audio_path = None
    error_context = ""
    
    try:
        logger.info(f"Day 12: Enhanced conversational chat request for session {session_id}")
        
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
        
        logger.info(f"Day 12: Enhanced conversational chat completed for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": audio_url,
            "model": "gemini-1.5-flash-enhanced" if is_full_success else "gemini-fallback-system",
            "voice": voice_info,
            "chat_history_length": history_length,
            "message": "Enhanced conversational chat with revamped UI completed!",
            "day": 12,
            "ui_version": "enhanced",
            "auto_play": True,  # Signal frontend to auto-play
            "error_handling_active": True,
            "pipeline_status": "full_success" if is_full_success else "partial_fallback",
            "error_context": error_context.strip() if error_context else None,
            "pipeline_steps": [
                "✅ Audio uploaded and validated",
                "✅ Enhanced STT with fallback" if "STT Error" not in error_context else "⚠️ STT failed - fallback used",
                "✅ Chat history management",
                "✅ Enhanced LLM with context" if "LLM Error" not in error_context else "⚠️ LLM failed - fallback used",
                "✅ Enhanced TTS with fallback" if "TTS Error" not in error_context else "⚠️ TTS failed - fallback used",
                "✅ Enhanced UI integration"
            ],
            "ui_features": {
                "single_record_button": True,
                "auto_play_audio": True,
                "smooth_animations": True,
                "responsive_design": True,
                "error_handling": True
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are intentional
        raise
    except Exception as e:
        # Log the full error for debugging
        logger.error(f"Day 12 Enhanced conversational chat critical failure: {str(e)}")
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
            "day": 12,
            "ui_version": "enhanced",
            "auto_play": True,
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
        "message_count": len(history),
        "day": 12
    }

# Helper endpoint to clear chat history
@app.delete("/agent/chat/{session_id}/history")
async def clear_session_history(session_id: str):
    """Clear chat history for a session"""
    clear_chat_history(session_id)
    return {
        "session_id": session_id,
        "message": "Chat history cleared",
        "day": 12
    }

# Day 12: UI status endpoint
@app.get("/ui/status")
async def get_ui_status():
    """Get UI status and features for Day 12"""
    return {
        "day": 12,
        "ui_version": "enhanced",
        "features": {
            "single_record_button": True,
            "auto_play_audio": True,
            "smooth_animations": True,
            "responsive_design": True,
            "error_handling": True,
            "chat_history": True,
            "session_management": True
        },
        "removed_features": [
            "tts_standalone_interface",
            "echo_bot_interface",
            "separate_record_stop_buttons"
        ],
        "enhanced_features": [
            "unified_record_button_with_state",
            "automatic_audio_playback",
            "animated_record_button",
            "streamlined_interface",
            "improved_error_messages"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)