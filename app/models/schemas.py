from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CBCResult(BaseModel):
    hemoglobin: Optional[float] = Field(default=None, description="Hemoglobin value in g/dL")
    rbc: Optional[float] = Field(default=None, description="RBC count in mill/cumm")
    wbc: Optional[int] = Field(default=None, description="WBC count in /cumm")
    platelets: Optional[int] = Field(default=None, description="Platelet count in /cumm")


class ExtractionResponse(BaseModel):
    success: bool = Field(..., description="Indicates if extraction was successful")
    data: Optional[CBCResult] = Field(default=None, description="Extracted patient result values")
    error: Optional[str] = Field(default=None, description="Error message in case of failure")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional processing metadata")


class AnalysisResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the full analysis pipeline succeeded")
    data: Optional[CBCResult] = Field(default=None, description="Extracted CBC biomarkers")
    ml_predictions: Optional[Dict[str, float]] = Field(
        default=None, description="ML disease risk probabilities (percent)"
    )
    summary: Optional[str] = Field(default=None, description="LLM-generated patient-friendly summary")
    error: Optional[str] = Field(default=None, description="Error message if pipeline failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Processing metadata")
