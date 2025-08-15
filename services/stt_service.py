import os
import requests
import logging
import asyncio
from datetime import datetime
from services.utils import STTError, retry_with_backoff

# Import get_chat_history for demo transcription fallback (avoid circular import issues)
try:
    from main import get_chat_history
except ImportError:
    get_chat_history = None

logger = logging.getLogger(__name__)

async def transcribe_audio_with_fallback(audio_path: str, session_id: str = "") -> str:
    """
    Transcribe audio with robust error handling and fallback mechanisms
    """
    assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not assemblyai_api_key or assemblyai_api_key == "your_assemblyai_api_key_here":
        logger.warning("AssemblyAI API key not configured, using intelligent demo mode")
        return generate_demo_transcription(session_id)
    try:
        return await retry_with_backoff(perform_assemblyai_transcription, audio_path, assemblyai_api_key)
    except Exception as e:
        logger.error(f"STT failed after retries: {str(e)}")
        return generate_demo_transcription(session_id, error_context=str(e))


async def perform_assemblyai_transcription(audio_path: str, api_key: str) -> str:
    """Perform the actual AssemblyAI transcription with enhanced error handling"""
    headers = {"authorization": api_key}
    try:
        processed_audio_path = await prepare_audio_file(audio_path)
        upload_url = await upload_audio_to_assemblyai(processed_audio_path, headers)
        transcript_id = await submit_transcription_request(upload_url, headers)
        transcribed_text = await poll_transcription_completion(transcript_id, headers)
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        return transcribed_text
    except Exception as e:
        if 'processed_audio_path' in locals() and processed_audio_path != audio_path:
            try:
                os.remove(processed_audio_path)
            except:
                pass
        raise STTError(f"AssemblyAI transcription failed: {str(e)}", e)

async def prepare_audio_file(audio_path: str) -> str:
    try:
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

def generate_demo_transcription(session_id: str = "", error_context: str = "") -> str:
    history_length = 0
    if get_chat_history:
        try:
            history_length = len(get_chat_history(session_id)) if session_id else 0
        except Exception:
            pass
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
