from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from enum import Enum

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AssessmentData(BaseModel):
    parent_observations: str = Field(description="Summary of key observations reported by parents/caregivers")
    ai_analysis: str = Field(description="AI's professional analysis of trauma indicators and behavioral patterns")
    severity_score: int = Field(description="Trauma severity score from 1-10", ge=1, le=10)
    risk_indicators: List[str] = Field(description="Array of specific trauma/behavioral risk flags identified")
    cultural_context: str = Field(description="Cultural considerations and context-specific interpretations")

class TraumaAssessmentDataset(BaseModel):
    """Generate realistic trauma assessment conversation and corresponding professional report"""
    conversation: List[ConversationTurn] = Field(description="Complete conversation between parent/caregiver and AI assistant. The conversation should always start with the parent/caregiver and end with the AI assistant.")
    assessment_data: AssessmentData = Field(description="Professional assessment report based on the conversation")
    
class DatasetBatch(BaseModel):
    """Batch of multiple trauma assessment cases for fine-tuning"""
    cases: List[TraumaAssessmentDataset] = Field(description="List of trauma assessment cases")