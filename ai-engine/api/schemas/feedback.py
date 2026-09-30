from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    analysis_id: UUID
    evaluator_id: UUID
    rating: float = Field(ge=0.5, le=5, multiple_of=0.5, allow_inf_nan=False)


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
    rating: float
    category: str
    already_submitted: bool = False
    summary: FeedbackSummary
