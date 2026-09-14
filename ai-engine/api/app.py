import os
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.analyze import router as analyze_router


app = FastAPI(
    title="HIBRIA API",
    version="1.0.0",
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


@app.middleware("http")
async def proteger_api(request: Request, call_next):
    if request.url.path.startswith("/analyze") and request.method != "OPTIONS":

        if not HIBRIA_API_KEY:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Chave da API HÍBRIA não configurada.",
                },
            )

        chave_recebida = request.headers.get("X-Hibria-Key", "")

        if not secrets.compare_digest(chave_recebida, HIBRIA_API_KEY):
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Não autorizado.",
                },
            )

    return await call_next(request)


app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "message": "HIBRIA API online",
    }
