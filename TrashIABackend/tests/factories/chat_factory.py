import uuid
from typing import Dict, Any, List, Optional


class ChatDataFactory:
    
    LANGUAGES = ["en", "es"]
    MATERIALS = ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'trash']
    
    @staticmethod
    def create_session_id() -> str:
        return str(uuid.uuid4())
    
    @staticmethod
    def create_session_request(
        material_type: Optional[str] = None,
        is_recyclable: Optional[bool] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        return {
            "material_type": material_type,
            "is_recyclable": is_recyclable,
            "material_info": f"Test {material_type or 'material'} item",
            "language": language
        }
    
    @staticmethod
    def create_session_response(
        session_id: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id or ChatDataFactory.create_session_id(),
            "message": "Hello! I'm TrashIA, your recycling assistant.",
            "language": language
        }
    
    @staticmethod
    def create_message_request(
        session_id: Optional[str] = None,
        message: str = "How can I recycle plastic bottles?"
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id or ChatDataFactory.create_session_id(),
            "message": message
        }
    
    @staticmethod
    def create_message_response(
        session_id: Optional[str] = None,
        response: str = "Plastic bottles should be rinsed and placed in the recycling bin."
    ) -> Dict[str, Any]:
        return {
            "response": response,
            "session_id": session_id or ChatDataFactory.create_session_id(),
            "on_topic": True
        }
    
    @staticmethod
    def create_chat_history(message_count: int = 4) -> List[Dict[str, str]]:
        history = []
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"{'Question' if role == 'user' else 'Answer'} {i // 2 + 1}"
            history.append({"role": role, "content": content})
        return history
