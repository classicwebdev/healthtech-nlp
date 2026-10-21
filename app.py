"""
HealthTech Clinical NER REST API
--------------------------------
FastAPI application serving clinical named entity recognition endpoints.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
from nlp_engine import ClinicalNEREngine

# 1. DEFINE PYDANTIC VALIDATION SCHEMAS
class ClinicalTextRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3,
        example="Patient presents with severe headache and nausea. Prescribed 50mg Sumatriptan."
    )

class EntityOutput(BaseModel):
    entity: str
    category: str
    score: float

class AnalysisResponse(BaseModel):
    status: str
    total_entities: int
    entities: List[EntityOutput]

# 2. INITIALIZE FASTAPI APP
app = FastAPI(
    title="HealthTech Clinical NER API",
    description="Microservice for extracting clinical entities (problems, treatments, tests) from unstructured text.",
    version="1.0.0"
)

# 3. INSTANTIATE AI ENGINE AT STARTUP
# Model weights load into memory once when the server starts
engine = ClinicalNEREngine()

@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify API operational status."""
    return {"status": "online", "service": "Clinical NER API"}

@app.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
def analyze_notes(payload: ClinicalTextRequest):
    """
    Extracts medical entities from raw clinical text.
    """
    try:
        extracted = engine.extract_entities(payload.text)
        return AnalysisResponse(
            status="success",
            total_entities=len(extracted),
            entities=extracted
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )