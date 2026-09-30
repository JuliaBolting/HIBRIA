from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import os
import logging
import traceback
import threading
import time
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, HTTPException

from api.schemas.request import AnalyzeRequest
from api.schemas.response import AnalyzeResponse
from pipeline.pipeline import HibriaPipeline
from pipeline.persistence.analysis_repository import AnalysisRepository, PersistenceError
from pipeline.persistence.rag_memory_service import RagMemoryService
from pipeline.preprocessing.extractor import ExtractionError, TextExtractor
from pipeline.security.redaction import redact, error_summary
from pipeline.cancellation import AnalysisCancelled as AnalysisJobCancelled, cancellable
from pipeline.output.explanation_cache import fingerprint, refresh


router = APIRouter(prefix="/analyze", tags=["Analysis"])


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
ANALYSIS_QUEUE_TIMEOUT = _env_int("HIBRIA_ANALYSIS_QUEUE_TIMEOUT_SECONDS", 600)
ANALYSIS_JOB_TTL_SECONDS = _env_int("HIBRIA_ANALYSIS_JOB_TTL_SECONDS", 21_600)
ANALYSIS_JOB_WORKERS = _env_int("HIBRIA_ANALYSIS_JOB_WORKERS", 4)
MAX_PENDING_JOBS = _env_int("HIBRIA_MAX_PENDING_JOBS", 8)
MAX_RETAINED_JOBS = _env_int("HIBRIA_MAX_RETAINED_JOBS", 200)
_job_admission = threading.BoundedSemaphore(MAX_PENDING_JOBS)
CACHE_POLICY = "server-verified-security-1"


logger = logging.getLogger(__name__)


def _log_failure(exc):
    # Registra classe e locais da falha, nunca argumentos/credenciais da exceção.
    frames = traceback.extract_tb(exc.__traceback__)
    locations = " > ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in frames)
    logger.error("[analysis] %s at %s", error_summary(exc), locations)


@dataclass
class AnalysisJob:
    job_id: str
    status: str = "queued"
    current_step: int = 0
    data: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    deadline: float = field(default_factory=lambda: time.monotonic() + ANALYSIS_QUEUE_TIMEOUT)
    future: Any = None


_jobs_lock = threading.Lock()
_jobs: dict[str, AnalysisJob] = {}
_job_executor = ThreadPoolExecutor(
    max_workers=ANALYSIS_JOB_WORKERS,
    thread_name_prefix="hibria-analysis",
)


PIPELINE_UI_STEPS = {
    "extractor": 0,
    "cleaner": 1,
    "normalizer": 1,
    "segmentation": 1,
    "claim_detector": 1,
    "retriever": 2,
    "similarity": 2,
    "stance": 2,
    "bertimbau": 2,
    "text_features": 2,
    "reputation": 2,
    "aggregator": 2,
    "explanation": 3,
    "formatter": 3,
}


def _mark_cache(
    data: dict,
    *,
    hit: bool,
    analysis_id: str | None,
    analyzed_at=None,
    request_count: int = 1,
) -> dict:
    payload = redact(deepcopy(data))
    metadata = payload.setdefault("metadata", {})
    metadata["cache"] = {
        "hit": hit,
        "analysis_id": analysis_id,
        "analyzed_at": analyzed_at.isoformat() if analyzed_at else None,
        "request_count": request_count,
        "pipeline_version": AnalysisRepository.pipeline_version(),
    }
    return payload


def _is_complete_analysis(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return False
    score = analysis.get("score")
    label = analysis.get("label")
    explanation = data.get("explanation")
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0 <= float(score) <= 100
        and isinstance(label, str)
        and bool(label.strip())
        and isinstance(explanation, str)
        and bool(explanation.strip())
    )


def _cached_data(
    repository: AnalysisRepository,
    request: AnalyzeRequest,
    *, refresh_explanation: bool = False,
) -> dict[str, Any] | None:
    if not _env_flag("HIBRIA_ANALYSIS_CACHE_ENABLED", True):
        return None

    cached = repository.get(url=str(request.url), content=request.content)
    if cached is None or not _is_complete_analysis(cached.data):
        return None

    analysis_id = cached.analysis_id
    data = deepcopy(cached.data)
    metadata = data.get("metadata") or {}
    if metadata.get("input_policy") != CACHE_POLICY:
        return None
    if metadata.get("explanation_fingerprint") != fingerprint():
        if not refresh_explanation:
            return None
        data = refresh(data)
        analysis_id = repository.revise_explanation(cached.analysis_id, data)

    return _mark_cache(
        data,
        hit=True,
        analysis_id=analysis_id,
        analyzed_at=cached.analyzed_at,
        request_count=cached.request_count,
    )


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisJobCancelled("Análise cancelada pelo usuário.")


def _acquire_analysis_slot(cancel_event: threading.Event | None, deadline: float | None = None) -> None:
    deadline = deadline if deadline is not None else time.monotonic() + ANALYSIS_QUEUE_TIMEOUT
    while True:
        _raise_if_cancelled(cancel_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(
                status_code=503,
                detail="Servidor ocupado. Tente novamente em alguns minutos.",
            )
        if ANALYSIS_SEMAPHORE.acquire(timeout=min(0.5, remaining)):
            return


@cancellable
def _perform_analysis(
    request: AnalyzeRequest,
    *,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    repository = AnalysisRepository()
    repository.ensure_ready()
    _raise_if_cancelled(cancel_event)

    cached = _cached_data(repository, request)
    if cached is not None:
        return cached

    _acquire_analysis_slot(cancel_event, deadline)
    try:
        _raise_if_cancelled(cancel_event)
        cached = _cached_data(repository, request, refresh_explanation=True)
        if cached is not None:
            return cached

        def report_progress(step_name: str) -> None:
            _raise_if_cancelled(cancel_event)
            if progress_callback is not None:
                progress_callback(step_name)

        report_progress("extractor")
        extracted = TextExtractor.extract(str(request.url))
        verified_content = extracted.get("content", "").strip()[:200_000]
        if len(verified_content) < 100:
            raise ExtractionError("Não foi possível conferir o conteúdo desta página no servidor. Tente outra notícia.")
        _raise_if_cancelled(cancel_event)
        result = HibriaPipeline.run(
            url=str(request.url),
            title=extracted.get("title", "")[:500],
            content=verified_content,
            progress_callback=report_progress,
        )
        _raise_if_cancelled(cancel_event)

        data = result.response or result.to_dict()
        data.setdefault("metadata", {}).update(input_policy=CACHE_POLICY,
                                               explanation_fingerprint=fingerprint(),
                                               render_method=extracted.get("render_method", "static"),
                                               paywall_detected=bool(extracted.get("paywall_detected", False)))
        if not _is_complete_analysis(data):
            raise RuntimeError(
                "A análise terminou sem score ou classificação final; "
                "o resultado incompleto não foi salvo no cache."
            )

        processing_time = float(
            getattr(result, "_processing_time", {}).get("total", 0.0)
        )
        analysis_id = repository.save(
            url=str(request.url),
            title=result.title or "Notícia",
            content=verified_content,
            score=result.score_final,
            classification=result.label_final,
            explanation=result.explanation,
            processing_time=processing_time,
            data=data,
        )

        if not analysis_id:
            raise RuntimeError("Não foi possível salvar a análise no banco de dados.")
        _raise_if_cancelled(cancel_event)
        rag_memory = RagMemoryService().persist_and_index(
            analysis_id,
            result,
            HibriaPipeline._get_vector_store(),
        )
        metadata = data.setdefault("metadata", {})
        metadata["rag_memory"] = rag_memory
        if analysis_id:
            if not repository.update_result(analysis_id, data):
                logger.warning("[analysis] metadados RAG não atualizados; resultado principal preservado")

        return _mark_cache(data, hit=False, analysis_id=analysis_id)
    finally:
        ANALYSIS_SEMAPHORE.release()


def _cleanup_jobs() -> None:
    threshold = datetime.now(timezone.utc) - timedelta(
        seconds=ANALYSIS_JOB_TTL_SECONDS
    )
    with _jobs_lock:
        expired = [
            job_id
            for job_id, job in _jobs.items()
            if job.status in {"completed", "failed", "cancelled"}
            and job.updated_at < threshold
        ]
        for job_id in expired:
            _jobs.pop(job_id, None)
        for job in _jobs.values():
            if job.status == "queued" and time.monotonic() >= job.deadline:
                job.cancel_event.set()
                job.status, job.error = "failed", "A espera excedeu o limite. Tente novamente."
                if job.future is not None:
                    job.future.cancel()
        terminal = sorted((job for job in _jobs.values() if job.status in {"completed", "failed", "cancelled"}),
                          key=lambda job: job.updated_at)
        for job in terminal[:max(0, len(_jobs) - MAX_RETAINED_JOBS)]:
            _jobs.pop(job.job_id, None)


def _job_payload(job: AnalysisJob) -> dict[str, Any]:
    return {
        "success": job.status != "failed",
        "job_id": job.job_id,
        "status": job.status,
        "current_step": job.current_step,
        "data": deepcopy(job.data),
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def _compact_job_data(data: dict[str, Any]) -> dict[str, Any]:
    """Mantém no trabalho apenas os campos usados pela extensão."""
    evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return {
        "analysis": deepcopy(data.get("analysis")),
        "explanation": data.get("explanation"),
        "details": deepcopy(data.get("details") or []),
        "evidence": {
            "score": evidence.get("score"),
            "coverage": evidence.get("coverage"),
            "claim_count": evidence.get("claim_count"),
            "evidence_count": evidence.get("evidence_count"),
        },
        "metadata": {
            "cache": deepcopy(metadata.get("cache")),
            "explanation_source": metadata.get("explanation_source"),
        },
    }


def _update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)


def _run_job(job_id: str, request: AnalyzeRequest) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job.cancel_event.is_set():
            return
        cancel_event = job.cancel_event
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)

    def progress(step_name: str) -> None:
        _raise_if_cancelled(cancel_event)
        _update_job(
            job_id,
            current_step=PIPELINE_UI_STEPS.get(step_name, 1),
            status="running",
        )

    try:
        data = _perform_analysis(
            request,
            progress_callback=progress,
            cancel_event=cancel_event,
            deadline=job.deadline,
        )
        _raise_if_cancelled(cancel_event)
        _update_job(
            job_id,
            status="completed",
            current_step=3,
            data=_compact_job_data(data),
            error=None,
        )
    except AnalysisJobCancelled:
        _update_job(
            job_id,
            status="cancelled",
            data=None,
            error="Análise cancelada pelo usuário.",
        )
    except PersistenceError as exc:
        _log_failure(exc)
        _update_job(job_id, status="failed", data=None, error="Não foi possível salvar ou recuperar a análise. Verifique a conexão com o banco de dados.")
    except ExtractionError as exc:
        _update_job(job_id, status="failed", data=None, error=redact(str(exc)))
    except HTTPException as exc:
        _update_job(job_id, status="failed", data=None, error=str(exc.detail))
    except Exception as exc:
        _log_failure(exc)
        _update_job(
            job_id,
            status="failed",
            data=None,
            error="Não foi possível concluir a análise. Tente novamente mais tarde.",
        )


def _get_job(job_id: str) -> AnalysisJob:
    _cleanup_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Análise em andamento não encontrada.",
            )
        return job


@router.post("", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """Endpoint síncrono mantido para compatibilidade."""
    if not _job_admission.acquire(blocking=False):
        raise HTTPException(429, "Servidor ocupado. Tente novamente em instantes.", headers={"Retry-After": "30"})
    try:
        return AnalyzeResponse(
            success=True,
            data=_perform_analysis(request),
            error=None,
        )
    except HTTPException:
        raise
    except PersistenceError as exc:
        _log_failure(exc)
        raise HTTPException(503, "Não foi possível salvar ou recuperar a análise. Verifique a conexão com o banco de dados.") from None
    except ExtractionError as exc:
        return AnalyzeResponse(success=False, data=None, error=redact(str(exc)))
    except Exception as exc:
        _log_failure(exc)
        return AnalyzeResponse(
            success=False,
            data=None,
            error="Não foi possível concluir a análise. Tente novamente mais tarde.",
        )

    finally:
        _job_admission.release()

@router.put("/jobs/{job_id}")
def start_analysis_job(job_id: UUID, request: AnalyzeRequest):
    """Cria uma análise idempotente que continua após o popup ser fechado."""
    job_key = str(job_id)
    _cleanup_jobs()

    with _jobs_lock:
        existing = _jobs.get(job_key)
        if existing is not None:
            return _job_payload(existing)

        if not _job_admission.acquire(blocking=False):
            raise HTTPException(429, "Servidor ocupado. Tente novamente em instantes.", headers={"Retry-After": "30"})
        job = AnalysisJob(job_id=job_key)
        _jobs[job_key] = job
        payload = _job_payload(job)

    try:
        job.future = _job_executor.submit(_run_job, job_key, request.model_copy(deep=True))
        job.future.add_done_callback(lambda _future: _job_admission.release())
    except Exception:
        _job_admission.release()
        with _jobs_lock:
            _jobs.pop(job_key, None)
        raise HTTPException(503, "Não foi possível iniciar a análise.") from None
    return payload


@router.get("/jobs/{job_id}")
def get_analysis_job(job_id: UUID):
    return _job_payload(_get_job(str(job_id)))


@router.delete("/jobs/{job_id}")
def cancel_analysis_job(job_id: UUID):
    job = _get_job(str(job_id))
    with _jobs_lock:
        job.cancel_event.set()
        if job.future is not None:
            job.future.cancel()
        if job.status not in {"completed", "failed"}:
            job.status = "cancelled"
            job.error = "Análise cancelada pelo usuário."
            job.updated_at = datetime.now(timezone.utc)
        return _job_payload(job)
