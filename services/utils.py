import asyncio
import logging
from datetime import datetime

# Error handling constants
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
FALLBACK_AUDIO_URL = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTuW2e/LdCUELIHQ8tiJOQcZZ7zl559NEApPqOPxtmMcBQeA5"

class VoiceAgentError(Exception):
    def __init__(self, message: str, error_type: str = "general_error", original_error: Exception = None):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(self.message)

class STTError(VoiceAgentError):
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "stt_failed", original_error)

class LLMError(VoiceAgentError):
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "llm_failed", original_error)

class TTSError(VoiceAgentError):
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "tts_failed", original_error)

async def retry_with_backoff(func, *args, **kwargs):
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
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
