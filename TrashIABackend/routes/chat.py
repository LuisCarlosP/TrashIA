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

# Global chat service instance
chat_service: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """Dependency to get the chat service"""
    global chat_service
    if chat_service is None:
        try:
            chat_service = ChatService()
        except Exception as e:
            logger.error(f"Error initializing ChatService: {e}")
            raise HTTPException(
                status_code=503,
                detail="Chat service unavailable. Verify that GEMINI_API_KEY is configured."
            )
    return chat_service

# Request/response models
class CreateSessionRequest(BaseModel):
    """Request to create a chat session"""
    material_type: Optional[str] = Field(None, description="Identified material type")
    is_recyclable: Optional[bool] = Field(None, description="Whether the material is recyclable")
    material_info: Optional[str] = Field(None, description="Additional material information")
    language: str = Field("en", description="Chat language (en/es)")
    
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
    """Request to send a message"""
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456",
                "message": "How can I recycle this plastic bottle?"
            }
        }

class UpdateMaterialRequest(BaseModel):
    """Request to update material context"""
    session_id: str = Field(..., description="Chat session ID")
    material_type: str = Field(..., description="Identified material type")
    is_recyclable: bool = Field(..., description="Whether the material is recyclable")
    material_info: str = Field(..., description="Additional material information")
    
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
    Create a new chat session.
    
    If material information is provided, the chat is initialized with that context.
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
        
        logger.info(f"Chat session created: {session_id}")
        return JSONResponse(content=result, status_code=201)
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Internal error creating chat session")

@router.post("/message")
@limiter.limit("30/minute")
async def send_message(
    request: Request,
    message_request: SendMessageRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Send a message to the chat and get a response.
    
    The chat will only respond to questions related to recycling and sustainability.
    """
    try:
        result = chat_svc.send_message(
            session_id=message_request.session_id,
            message=message_request.message
        )
        
        return JSONResponse(content=result)
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing message")

@router.get("/history/{session_id}")
@limiter.limit("20/minute")
async def get_chat_history(
    request: Request,
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Get the conversation history of a session.
    """
    try:
        history = chat_svc.get_chat_history(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "history": history
        })
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail="Internal error getting history")

@router.delete("/session/{session_id}")
@limiter.limit("10/minute")
async def delete_chat_session(
    request: Request,
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Delete a chat session.
    """
    try:
        deleted = chat_svc.delete_chat_session(session_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return JSONResponse(content={
            "message": "Session deleted successfully",
            "session_id": session_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="Internal error deleting session")

@router.put("/material")
@limiter.limit("20/minute")
async def update_material_context(
    request: Request,
    update_request: UpdateMaterialRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """
    Update the material context in an existing session.
    
    Useful when the user identifies a new material.
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
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating context: {e}")
        raise HTTPException(status_code=500, detail="Internal error updating context")
