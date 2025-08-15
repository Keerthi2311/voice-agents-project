import os
import requests
import logging
from services.utils import TTSError, retry_with_backoff, FALLBACK_AUDIO_URL

logger = logging.getLogger(__name__)

async def generate_tts_with_fallback(text: str) -> tuple[str, str]:
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
