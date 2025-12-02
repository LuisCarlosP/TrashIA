"""
Modelos Pydantic para funcionalidades de ubicación y puntos de reciclaje.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Coordenadas geográficas."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitud")
    longitude: float = Field(..., ge=-180, le=180, description="Longitud")


class RecyclingPoint(BaseModel):
    """Punto de reciclaje con información detallada."""
    id: str = Field(..., description="Identificador único del punto")
    name: str = Field(..., description="Nombre del punto de reciclaje")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    address: Optional[str] = Field(None, description="Dirección física")
    types: List[str] = Field(
        default_factory=list,
        description="Tipos de materiales aceptados (plastic, glass, paper, metal, cardboard, electronics, batteries)"
    )
    opening_hours: Optional[str] = Field(None, description="Horario de atención")
    phone: Optional[str] = Field(None, description="Teléfono de contacto")
    website: Optional[str] = Field(None, description="Sitio web")
    distance: Optional[float] = Field(None, description="Distancia en metros desde el usuario")
    operator: Optional[str] = Field(None, description="Operador o empresa")


class RecyclingPointsRequest(BaseModel):
    """Solicitud para buscar puntos de reciclaje cercanos."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitud del usuario")
    longitude: float = Field(..., ge=-180, le=180, description="Longitud del usuario")
    radius: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Radio de búsqueda en metros (100-50000)"
    )
    types: Optional[List[str]] = Field(
        None,
        description="Filtrar por tipos de materiales específicos"
    )


class RecyclingPointsResponse(BaseModel):
    """Respuesta con lista de puntos de reciclaje."""
    success: bool = Field(True)
    count: int = Field(..., description="Número de puntos encontrados")
    radius: int = Field(..., description="Radio de búsqueda utilizado")
    center: Coordinates = Field(..., description="Centro de búsqueda")
    points: List[RecyclingPoint] = Field(..., description="Lista de puntos de reciclaje")


class LocationErrorResponse(BaseModel):
    """Respuesta de error para operaciones de ubicación."""
    success: bool = Field(False)
    error: str = Field(..., description="Mensaje de error")
    code: int = Field(..., description="Código de error HTTP")
