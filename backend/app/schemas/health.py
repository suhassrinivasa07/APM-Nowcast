from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health status indicator")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the health check")
    environment: str = Field(..., description="Active runtime environment")
    device_target: Optional[str] = Field(default=None, description="Configured compute target (cuda/cpu)")
