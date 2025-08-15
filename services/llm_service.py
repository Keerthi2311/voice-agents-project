import os
import logging
from services.utils import LLMError, retry_with_backoff

logger = logging.getLogger(__name__)

# Import get_chat_history for fallback LLM response (avoid circular import issues)
try:
    from main import get_chat_history
except ImportError:
    get_chat_history = None

async def generate_llm_response_with_fallback(prompt: str, session_id: str = "") -> str:
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
    history_length = 0
    if get_chat_history:
        try:
            history_length = len(get_chat_history(session_id)) if session_id else 0
        except Exception:
            pass
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
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        enhanced_prompt = f"""Please provide a helpful, safe, and concise response (under 2000 characters for voice synthesis) to this query: {prompt}"""
        response = model.generate_content(enhanced_prompt)
        if not response.text:
            raise Exception("LLM returned empty response")
        llm_text = response.text
        if len(llm_text) > 2200:
            llm_text = llm_text[:2100] + "... Would you like me to continue?"
        return llm_text
    except ImportError:
        raise Exception("Google Generative AI library not installed")
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            raise Exception("API quota exceeded - please try again later")
        elif "key" in str(e).lower():
            raise Exception("API key authentication failed")
        else:
            raise Exception(f"LLM generation failed: {str(e)}")
