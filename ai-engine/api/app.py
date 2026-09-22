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


app = FastAPI(
    title="HIBRIA API",
    version="1.2.0",
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
        window = _rate_windows[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return False
        window.append(now)
        return True


@app.middleware("http")
async def proteger_api(request: Request, call_next):
    if request.url.path.startswith("/analyze") and request.method != "OPTIONS":

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

        if not authenticated:
            client = request.client.host if request.client else "unknown"
            if not _within_public_rate_limit(client):
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "error": "Limite temporário de análises atingido. Tente novamente em um minuto.",
                    },
                )

    return await call_next(request)


app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "message": "HIBRIA API online",
    }
