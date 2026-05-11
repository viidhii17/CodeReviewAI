"""
Pydantic schemas for FastAPI request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ReviewRequest(BaseModel):
    code: str = Field(..., description="Source code to review", min_length=10)
    language: str = Field(default="python", description="Programming language: python or javascript")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "def divide(a, b):\n    return a / b",
                "language": "python"
            }
        }


class Finding(BaseModel):
    function_name: str
    start_line: int
    end_line: int
    num_lines: int
    complexity: int
    bug_type: str
    confidence: float
    severity: str          # high | medium | none
    suggestion: str
    probabilities: Dict[str, float]


class ReviewResponse(BaseModel):
    language: str
    total_functions: int
    bugs_found: int
    overall_risk: str      # high | medium | low | clean
    findings: List[Finding]
    summary: str