"""
Pydantic models for location functionality and recycling points.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Geographic coordinates."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")


class RecyclingPoint(BaseModel):
    """Recycling point with detailed information."""
    id: str = Field(..., description="Unique point identifier")
    name: str = Field(..., description="Recycling point name")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    address: Optional[str] = Field(None, description="Physical address")
    types: List[str] = Field(
        default_factory=list,
        description="Accepted material types (plastic, glass, paper, metal, cardboard, electronics, batteries)"
    )
    opening_hours: Optional[str] = Field(None, description="Opening hours")
    phone: Optional[str] = Field(None, description="Contact phone")
    website: Optional[str] = Field(None, description="Website")
    distance: Optional[float] = Field(None, description="Distance in meters from user")
    operator: Optional[str] = Field(None, description="Operator or company")


class RecyclingPointsRequest(BaseModel):
    """Request to search for nearby recycling points."""
    latitude: float = Field(..., ge=-90, le=90, description="User latitude")
    longitude: float = Field(..., ge=-180, le=180, description="User longitude")
    radius: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Search radius in meters (100-50000)"
    )
    types: Optional[List[str]] = Field(
        None,
        description="Filter by specific material types"
    )


class RecyclingPointsResponse(BaseModel):
    """Response with list of recycling points."""
    success: bool = Field(True)
    count: int = Field(..., description="Number of points found")
    radius: int = Field(..., description="Search radius used")
    center: Coordinates = Field(..., description="Search center")
    points: List[RecyclingPoint] = Field(..., description="List of recycling points")


class LocationErrorResponse(BaseModel):
    """Error response for location operations."""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    code: int = Field(..., description="HTTP error code")
