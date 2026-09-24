from fastapi import APIRouter, HTTPException

from api.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummary,
)
from pipeline.persistence.feedback_repository import FeedbackRepository


router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


@router.post("", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    repository = FeedbackRepository()

    try:
        record = repository.save(
            analysis_id=str(request.analysis_id),
            evaluator_id=str(request.evaluator_id),
            rating=request.rating,
        )
        summary = FeedbackSummary(**repository.summary())
        return FeedbackResponse(
            success=True,
            rating=record.rating,
            category=record.category,
            already_submitted=record.already_submitted,
            summary=summary,
        )
    except Exception as exc:
        message = str(exc)
        if "foreign key" in message.casefold():
            raise HTTPException(
                status_code=404,
                detail="A análise informada não foi encontrada.",
            ) from exc
        raise HTTPException(
            status_code=503,
            detail="Não foi possível registrar a avaliação agora.",
        ) from exc


@router.get("/summary", response_model=FeedbackSummary)
def feedback_summary():
    try:
        return FeedbackSummary(**FeedbackRepository().summary())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível consultar o resumo das avaliações.",
        ) from exc
