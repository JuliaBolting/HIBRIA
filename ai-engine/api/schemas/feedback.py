from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    analysis_id: UUID
    evaluator_id: UUID
    rating: int = Field(ge=1, le=5)


class FeedbackSummary(BaseModel):
    total: int
    average: float | None
    positive: int
    neutral: int
    negative: int
    positive_percentage: float
    neutral_percentage: float
    negative_percentage: float


class FeedbackResponse(BaseModel):
    success: bool
    rating: int
    category: str
    already_submitted: bool = False
    summary: FeedbackSummary
