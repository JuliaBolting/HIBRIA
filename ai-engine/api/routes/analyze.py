from copy import deepcopy
import os
import threading

from fastapi import APIRouter

from api.schemas.request import AnalyzeRequest
from api.schemas.response import AnalyzeResponse

from pipeline.pipeline import HibriaPipeline
from pipeline.persistence.analysis_repository import AnalysisRepository
from pipeline.persistence.rag_memory_service import RagMemoryService
from pipeline.preprocessing.extractor import ExtractionError


router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "sim", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(
    _env_int("HIBRIA_MAX_CONCURRENT_ANALYSES", 1)
)


def _mark_cache(
    data: dict,
    *,
    hit: bool,
    analysis_id: str | None,
    analyzed_at=None,
    request_count: int = 1,
) -> dict:
    payload = deepcopy(data)
    metadata = payload.setdefault("metadata", {})
    metadata["cache"] = {
        "hit": hit,
        "analysis_id": analysis_id,
        "analyzed_at": analyzed_at.isoformat() if analyzed_at else None,
        "request_count": request_count,
        "pipeline_version": AnalysisRepository.pipeline_version(),
    }
    return payload


def _cached_response(
    repository: AnalysisRepository,
    request: AnalyzeRequest,
) -> AnalyzeResponse | None:
    if not _env_flag("HIBRIA_ANALYSIS_CACHE_ENABLED", True):
        return None

    cached = repository.get(
        url=str(request.url),
        content=request.content,
    )
    if cached is None:
        return None

    return AnalyzeResponse(
        success=True,
        data=_mark_cache(
            cached.data,
            hit=True,
            analysis_id=cached.analysis_id,
            analyzed_at=cached.analyzed_at,
            request_count=cached.request_count,
        ),
        error=None,
    )


@router.post("", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    repository = AnalysisRepository()

    try:
        cached_response = _cached_response(repository, request)
        if cached_response is not None:
            return cached_response

        # Na instância de 8 GB, o padrão é executar uma análise pesada por vez.
        # As demais requisições aguardam e verificam o cache novamente.
        with ANALYSIS_SEMAPHORE:
            cached_response = _cached_response(repository, request)
            if cached_response is not None:
                return cached_response

            result = HibriaPipeline.run(
                url=str(request.url),
                title=request.title,
                content=request.content,
            )

            data = result.response or result.to_dict()
            processing_time = sum(
                float(value)
                for value in getattr(result, "_processing_time", {}).values()
            )

            analysis_id = repository.save(
                url=str(request.url),
                title=result.title or request.title,
                content=request.content,
                score=result.score_final,
                classification=result.label_final,
                explanation=result.explanation,
                processing_time=processing_time,
                data=data,
            )

            rag_memory = RagMemoryService().persist_and_index(
                analysis_id,
                result,
                HibriaPipeline._get_vector_store(),
            )
            metadata = data.setdefault("metadata", {})
            metadata["rag_memory"] = rag_memory
            if analysis_id:
                repository.update_result(analysis_id, data)

            return AnalyzeResponse(
                success=True,
                data=_mark_cache(
                    data,
                    hit=False,
                    analysis_id=analysis_id,
                ),
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
