import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.chat_service import ChatService, MaterialContext
from exceptions.validation_exceptions import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])
limiter = Limiter(key_func=get_remote_address)

# Instancia global del servicio de chat
chat_service: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """Dependency para obtener el servicio de chat"""
    global chat_service
    if chat_service is None:
        try:
            chat_service = ChatService()
        except Exception as e:
            logger.error(f"Error al inicializar ChatService: {e}")
            raise HTTPException(
                status_code=503,
                detail="Servicio de chat no disponible. Verifica que GEMINI_API_KEY esté configurada."
            )
    return chat_service

# Modelos de request/response
class CreateSessionRequest(BaseModel):
    """Request para crear una sesión de chat"""
    material_type: Optional[str] = Field(None, description="Tipo de material identificado")
    is_recyclable: Optional[bool] = Field(None, description="Si el material es reciclable")
    material_info: Optional[str] = Field(None, description="Información adicional del material")
    language: str = Field("en", description="Idioma del chat (en/es)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "material_type": "plastic",
                "is_recyclable": True,
                "material_info": "PET plastic bottle",
                "language": "en"
            }
        }

class SendMessageRequest(BaseModel):
    """Request para enviar un mensaje"""
    session_id: str = Field(..., description="ID de la sesión de chat")
    message: str = Field(..., min_length=1, max_length=1000, description="Mensaje del usuario")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456",
                "message": "How can I recycle this plastic bottle?"
            }
        }

class UpdateMaterialRequest(BaseModel):
    """Request para actualizar el contexto del material"""
    session_id: str = Field(..., description="ID de la sesión de chat")
    material_type: str = Field(..., description="Tipo de material identificado")
    is_recyclable: bool = Field(..., description="Si el material es reciclable")
    material_info: str = Field(..., description="Información adicional del material")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456",
                "material_type": "cardboard",
                "is_recyclable": True,
                "material_info": "Corrugated cardboard box"
            }
        }

@router.post("/session")
@limiter.limit("20/minute")
async def create_chat_session(
    request: Request,
    session_request: CreateSessionRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Crea una nueva sesión de chat.
    
    Si se proporciona información del material, se inicializa el chat con ese contexto.
    """
    try:
        session_id = str(uuid.uuid4())
        
        material_context = None
        if session_request.material_type:
            material_context = MaterialContext(
                material_type=session_request.material_type,
                is_recyclable=session_request.is_recyclable or False,
                material_info=session_request.material_info or ""
            )
        
        result = chat_svc.create_chat_session(
            session_id=session_id,
            material_context=material_context,
            language=session_request.language
        )
        
        logger.info(f"Sesión de chat creada: {session_id}")
        return JSONResponse(content=result, status_code=201)
        
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear sesión: {e}")
        raise HTTPException(status_code=500, detail="Error interno al crear sesión de chat")

@router.post("/message")
@limiter.limit("30/minute")
async def send_message(
    request: Request,
    message_request: SendMessageRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Envía un mensaje al chat y obtiene una respuesta.
    
    El chat solo responderá preguntas relacionadas con reciclaje y sostenibilidad.
    """
    try:
        result = chat_svc.send_message(
            session_id=message_request.session_id,
            message=message_request.message
        )
        
        return JSONResponse(content=result)
        
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al enviar mensaje: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar mensaje")

@router.get("/history/{session_id}")
@limiter.limit("20/minute")
async def get_chat_history(
    request: Request,
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Obtiene el historial de conversación de una sesión.
    """
    try:
        history = chat_svc.get_chat_history(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "history": history
        })
        
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener historial")

@router.delete("/session/{session_id}")
@limiter.limit("10/minute")
async def delete_chat_session(
    request: Request,
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Elimina una sesión de chat.
    """
    try:
        deleted = chat_svc.delete_chat_session(session_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        return JSONResponse(content={
            "message": "Sesión eliminada correctamente",
            "session_id": session_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar sesión: {e}")
        raise HTTPException(status_code=500, detail="Error interno al eliminar sesión")

@router.put("/material")
@limiter.limit("20/minute")
async def update_material_context(
    request: Request,
    update_request: UpdateMaterialRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Actualiza el contexto del material en una sesión existente.
    
    Útil cuando el usuario identifica un nuevo material.
    """
    try:
        material_context = MaterialContext(
            material_type=update_request.material_type,
            is_recyclable=update_request.is_recyclable,
            material_info=update_request.material_info
        )
        
        result = chat_svc.update_material_context(
            session_id=update_request.session_id,
            material_context=material_context
        )
        
        return JSONResponse(content=result)
        
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al actualizar contexto: {e}")
        raise HTTPException(status_code=500, detail="Error interno al actualizar contexto")
