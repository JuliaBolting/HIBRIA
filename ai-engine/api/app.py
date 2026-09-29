import os
import secrets
import threading
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from api.routes.analyze import router as analyze_router  # noqa: E402
from api.routes.feedback import router as feedback_router  # noqa: E402


app = FastAPI(
    title="HIBRIA API",
    version="1.3.0",
    description="API de análise de desinformação do HIBRIA",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


HIBRIA_API_KEY = os.getenv("HIBRIA_API_KEY", "").strip()
HIBRIA_PUBLIC_API = os.getenv("HIBRIA_PUBLIC_API", "false").strip().lower() in {
    "1", "true", "sim", "yes", "on"
}

try:
    RATE_LIMIT_PER_MINUTE = max(
        1, int(os.getenv("HIBRIA_RATE_LIMIT_PER_MINUTE", "6"))
    )
except ValueError:
    RATE_LIMIT_PER_MINUTE = 6

_rate_lock = threading.Lock()
_rate_windows: dict[str, deque[float]] = defaultdict(deque)


def _within_public_rate_limit(client: str) -> bool:
    now = time.monotonic()
    with _rate_lock:
        if len(_rate_windows) > 2000:
            for key in list(_rate_windows):
                if not _rate_windows[key] or now - _rate_windows[key][-1] >= 60:
                    _rate_windows.pop(key, None)
            if len(_rate_windows) >= 10000 and client not in _rate_windows:
                return False
        window = _rate_windows[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return False
        window.append(now)
        return True


@app.middleware("http")
async def proteger_api(request: Request, call_next):
    path = request.url.path
    protected_path = path.startswith("/analyze") or path.startswith("/feedback")

    if protected_path and request.method != "OPTIONS":

        if not HIBRIA_API_KEY and not HIBRIA_PUBLIC_API:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Chave da API HÍBRIA não configurada.",
                },
            )

        chave_recebida = request.headers.get("X-Hibria-Key", "")
        authenticated = bool(
            HIBRIA_API_KEY
            and secrets.compare_digest(chave_recebida, HIBRIA_API_KEY)
        )

        if not authenticated and not HIBRIA_PUBLIC_API:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Não autorizado.",
                },
            )

        starts_analysis = (
            request.method == "POST" and path == "/analyze"
        ) or (
            request.method == "PUT" and path.startswith("/analyze/jobs/")
        )

        # Consultas de andamento são frequentes e não consomem as APIs de
        # busca. O limite público é aplicado apenas quando uma análise começa.
        sends_feedback = request.method == "POST" and path == "/feedback"
        if not authenticated and (starts_analysis or sends_feedback):
            client = (request.client.host if request.client else "unknown") + (":feedback" if sends_feedback else ":analysis")
            if not _within_public_rate_limit(client):
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "error": "Limite temporário de solicitações atingido. Tente novamente em um minuto.",
                    },
                )

    if protected_path and request.method in {"POST", "PUT"}:
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1_500_000:
                return JSONResponse(status_code=413, content={"error": "Conteúdo excede o limite permitido."})
            chunks.append(chunk)
        request._body = b"".join(chunks)
    return await call_next(request)


app.include_router(analyze_router)
app.include_router(feedback_router)


@app.get("/")
def root():
    return {
        "message": "HIBRIA API online",
    }
