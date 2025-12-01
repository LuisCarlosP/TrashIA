import logging
import google.generativeai as genai
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config.settings import GEMINI_API_KEY, CHAT_PROMPTS
from exceptions.validation_exceptions import ValidationError

logger = logging.getLogger(__name__)

@dataclass
class MaterialContext:
    """Contexto del material identificado"""
    material_type: str
    is_recyclable: bool
    material_info: str
    
class ChatService:
    """Servicio para manejar conversaciones con Gemini AI sobre reciclaje"""
    
    def __init__(self):
        """Inicializa el servicio de chat con Gemini"""
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY no configurada")
            raise ValueError("GEMINI_API_KEY debe estar configurada en las variables de entorno")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.chat_sessions: Dict[str, Any] = {}
        logger.info("ChatService inicializado correctamente")
    
    def _is_on_topic(self, question: str) -> bool:
        """
        Verifica si la pregunta está relacionada con reciclaje y sostenibilidad
        usando palabras clave
        """
        question_lower = question.lower()
        keywords = CHAT_PROMPTS['topic_keywords']['on_topic']
        
        return any(keyword.lower() in question_lower for keyword in keywords)
    
    def _build_system_context(self, material_context: Optional[MaterialContext], language: str = "en") -> str:
        """
        Construye el contexto del sistema con los prompts desde el JSON
        """
        system_prompt = CHAT_PROMPTS['system_prompt'][language]
        moderation_prompt = CHAT_PROMPTS['moderation_prompt'][language]
        
        context = f"{system_prompt}\n\n{moderation_prompt}\n\n"
        
        if material_context:
            material_prompt = CHAT_PROMPTS['material_context_prompt'][language]
            recyclable_text = "Yes" if material_context.is_recyclable else "No"
            if language == "es":
                recyclable_text = "Sí" if material_context.is_recyclable else "No"
            
            context += material_prompt.format(
                material_type=material_context.material_type,
                is_recyclable=recyclable_text,
                material_info=material_context.material_info
            )
        
        return context
    
    def create_chat_session(
        self, 
        session_id: str, 
        material_context: Optional[MaterialContext] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Crea una nueva sesión de chat para un usuario
        
        Args:
            session_id: ID único de la sesión
            material_context: Contexto del material identificado
            language: Idioma (en/es)
        
        Returns:
            Mensaje de bienvenida
        """
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
                welcome = CHAT_PROMPTS['welcome_message'][language].format(
                    material_type=material_context.material_type
                )
            else:
                welcome = CHAT_PROMPTS['no_material_context'][language]
            
            logger.info(f"Nueva sesión de chat creada: {session_id}")
            
            return {
                "session_id": session_id,
                "message": welcome,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Error al crear sesión de chat: {e}")
            raise ValidationError(f"Error al iniciar chat: {str(e)}")
    
    def send_message(
        self, 
        session_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """
        Envía un mensaje al chat y obtiene respuesta
        
        Args:
            session_id: ID de la sesión
            message: Mensaje del usuario
        
        Returns:
            Respuesta del asistente
        """
        try:
            if session_id not in self.chat_sessions:
                raise ValidationError("Sesión de chat no encontrada. Por favor, crea una sesión primero.")
            
            session = self.chat_sessions[session_id]
            chat = session["chat"]
            language = session["language"]
            system_context = session["system_context"]
            
            if not self._is_on_topic(message):
                off_topic = CHAT_PROMPTS['off_topic_response'][language]
                session["history"].append({
                    "role": "user",
                    "content": message
                })
                session["history"].append({
                    "role": "assistant",
                    "content": off_topic
                })
                
                return {
                    "response": off_topic,
                    "session_id": session_id,
                    "on_topic": False
                }
            
            full_message = f"{system_context}\n\nUser question: {message}"
            
            response = chat.send_message(full_message)
            response_text = response.text
            
            session["history"].append({
                "role": "user",
                "content": message
            })
            session["history"].append({
                "role": "assistant",
                "content": response_text
            })
            
            logger.info(f"Mensaje procesado para sesión: {session_id}")
            
            return {
                "response": response_text,
                "session_id": session_id,
                "on_topic": True
            }
            
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}")
            language = self.chat_sessions.get(session_id, {}).get("language", "en")
            error_msg = CHAT_PROMPTS['error_message'][language]
            
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": str(e)
            }
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Obtiene el historial de conversación
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Lista de mensajes
        """
        if session_id not in self.chat_sessions:
            raise ValidationError("Sesión de chat no encontrada")
        
        return self.chat_sessions[session_id]["history"]
    
    def delete_chat_session(self, session_id: str) -> bool:
        """
        Elimina una sesión de chat
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            True si se eliminó correctamente
        """
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            logger.info(f"Sesión eliminada: {session_id}")
            return True
        return False
    
    def update_material_context(
        self, 
        session_id: str, 
        material_context: MaterialContext
    ) -> Dict[str, Any]:
        """
        Actualiza el contexto del material en una sesión existente
        
        Args:
            session_id: ID de la sesión
            material_context: Nuevo contexto del material
        
        Returns:
            Confirmación de actualización
        """
        if session_id not in self.chat_sessions:
            raise ValidationError("Sesión de chat no encontrada")
        
        session = self.chat_sessions[session_id]
        language = session["language"]
        
        session["material_context"] = material_context
        session["system_context"] = self._build_system_context(material_context, language)
        
        session["chat"] = self.model.start_chat(history=[])
        
        logger.info(f"Contexto de material actualizado para sesión: {session_id}")
        
        welcome = CHAT_PROMPTS['welcome_message'][language].format(
            material_type=material_context.material_type
        )
        
        return {
            "session_id": session_id,
            "message": welcome,
            "updated": True
        }
