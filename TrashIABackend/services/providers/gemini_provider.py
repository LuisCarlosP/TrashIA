import logging
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class GeminiChatProvider:
    def __init__(self, api_key: str = None, model_name: str = None):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY must be configured")
        
        genai.configure(api_key=key)
        model = model_name or GEMINI_MODEL
        self._model = genai.GenerativeModel(model, safety_settings=SAFETY_SETTINGS)
        self._generation_config = genai.GenerationConfig(max_output_tokens=1500)
        logger.info(f"GeminiChatProvider initialized with model: {model}")

    def start_chat(self, system_context: str = None) -> Any:
        return self._model.start_chat(history=[])

    def send_message(self, chat_session: Any, message: str) -> Optional[str]:
        response = chat_session.send_message(
            message,
            generation_config=self._generation_config,
            safety_settings=SAFETY_SETTINGS
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            return None
        
        return response.text
