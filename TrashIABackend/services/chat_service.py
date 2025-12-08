import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config.settings import GEMINI_API_KEY, CHAT_PROMPTS, MATERIAL_TRANSLATIONS
import os
from exceptions.validation_exceptions import ValidationError

logger = logging.getLogger(__name__)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

@dataclass
class MaterialContext:
    material_type: str
    is_recyclable: bool
    material_info: str
    
class ChatService:
    
    def __init__(self):
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not configured")
            raise ValueError("GEMINI_API_KEY must be configured in environment variables")

        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = os.getenv("GEMINI_MODEL")
        self.model = genai.GenerativeModel(gemini_model, safety_settings=SAFETY_SETTINGS)
        self.chat_sessions: Dict[str, Any] = {}
        logger.info(f"ChatService initialized with model: {gemini_model}")
    
    def _translate_material(self, material_type: str, language: str) -> str:
        if language in MATERIAL_TRANSLATIONS:
            return MATERIAL_TRANSLATIONS[language].get(material_type, material_type)
        return material_type
    
    def _build_system_context(self, material_context: Optional[MaterialContext], language: str = "en") -> str:
        lang_key = language if language in ["en", "es"] else "en"
        
        identity = CHAT_PROMPTS['identity'][f'description_{lang_key}']
        off_topic = CHAT_PROMPTS['off_topic_rule'][lang_key]
        
        material_info = ""
        if material_context:
            material_info = CHAT_PROMPTS['material_context'][lang_key].format(
                material_type=material_context.material_type
            )
        
        return f"{identity} {off_topic} {material_info}".strip()
    
    def create_chat_session(
        self, 
        session_id: str, 
        material_context: Optional[MaterialContext] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        try:
            if language not in ["en", "es"]:
                language = "en"
            
            system_context = self._build_system_context(material_context, language)
            chat = self.model.start_chat(history=[])
            
            self.chat_sessions[session_id] = {
                "chat": chat,
                "material_context": material_context,
                "language": language,
                "system_context": system_context,
                "history": []
            }
            
            if material_context:
                translated_material = self._translate_material(material_context.material_type, language)
                welcome = CHAT_PROMPTS['messages']['welcome_message'][language].format(
                    material_type=translated_material
                )
            else:
                welcome = CHAT_PROMPTS['messages']['no_material'][language]
            
            logger.info(f"Chat session created: {session_id}")
            
            return {
                "session_id": session_id,
                "message": welcome,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise ValidationError(f"Error starting chat: {str(e)}")
    
    def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        try:
            if session_id not in self.chat_sessions:
                raise ValidationError("Chat session not found. Please create a session first.")
            
            session = self.chat_sessions[session_id]
            chat = session["chat"]
            language = session["language"]
            system_context = session["system_context"]
            
            if len(session["history"]) == 0:
                full_message = f"{system_context}\n\nUser: {message}"
            else:
                full_message = message
            
            generation_config = genai.GenerationConfig(max_output_tokens=1500)
            
            response = chat.send_message(
                full_message, 
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning(f"Response blocked for session {session_id}")
                response_text = CHAT_PROMPTS['messages']['off_topic'][language]
            else:
                response_text = response.text
            
            session["history"].append({"role": "user", "content": message})
            session["history"].append({"role": "assistant", "content": response_text})
            
            logger.info(f"Message processed: {session_id}")
            
            return {
                "response": response_text,
                "session_id": session_id,
                "on_topic": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            language = self.chat_sessions.get(session_id, {}).get("language", "en")
            error_msg = CHAT_PROMPTS['messages']['error'][language]
            
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": str(e)
            }
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self.chat_sessions:
            raise ValidationError("Chat session not found")
        return self.chat_sessions[session_id]["history"]
    
    def delete_chat_session(self, session_id: str) -> bool:
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    
    def update_material_context(self, session_id: str, material_context: MaterialContext) -> Dict[str, Any]:
        if session_id not in self.chat_sessions:
            raise ValidationError("Chat session not found")
        
        session = self.chat_sessions[session_id]
        language = session["language"]
        
        session["material_context"] = material_context
        session["system_context"] = self._build_system_context(material_context, language)
        session["chat"] = self.model.start_chat(history=[])
        
        logger.info(f"Material context updated: {session_id}")
        
        translated_material = self._translate_material(material_context.material_type, language)
        welcome = CHAT_PROMPTS['messages']['welcome_message'][language].format(
            material_type=translated_material
        )
        
        return {
            "session_id": session_id,
            "message": welcome,
            "updated": True
        }
