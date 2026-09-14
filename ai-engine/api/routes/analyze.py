from fastapi import APIRouter

from api.schemas.request import AnalyzeRequest
from api.schemas.response import AnalyzeResponse

from pipeline.pipeline import HibriaPipeline
from pipeline.preprocessing.extractor import ExtractionError


router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)


@router.post("", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        result = HibriaPipeline.run(
            url=str(request.url),
            title=request.title,
            content=request.content,
        )

        return AnalyzeResponse(
            success=True,
            data=result.to_dict(),
            error=None,
        )

    except ExtractionError as e:
        return AnalyzeResponse(
            success=False,
            data=None,
            error=str(e),
        )

    except Exception as e:
        return AnalyzeResponse(
            success=False,
            data=None,
            error=f"Erro inesperado: {str(e)}",
        )