import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ollama import ResponseError

from app.config import Settings, sanitize
from app.rag_engine import LocalRAGPipeline
from app.schemas import (
    DirectGenerateRequest,
    DirectGenerateResponse,
    IngestRequest,
    IngestResponse,
    ModelListResponse,
    RAGRequest,
    RAGResponse,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, pipeline_factory=LocalRAGPipeline) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        pipeline = pipeline_factory(settings)
        app.state.pipeline = pipeline
        try:
            pipeline.client.heartbeat()
            try:
                pipeline.models()
            except (httpx.HTTPError, ConnectionError) as error:
                if settings.REQUIRE_OLLAMA_ON_STARTUP:
                    raise
                logger.warning("Ollama unavailable at startup: %s", error)
            yield
        finally:
            pipeline.close()

    app = FastAPI(title="Local RAG", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def invalid_input(request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def missing_resource(request: Request, error: FileNotFoundError):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(httpx.HTTPError)
    async def upstream_http(request: Request, error: httpx.HTTPError):
        status = 504 if isinstance(error, httpx.TimeoutException) else 503
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @app.exception_handler(ConnectionError)
    async def upstream_connection(request: Request, error: ConnectionError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(ResponseError)
    async def ollama_error(request: Request, error: ResponseError):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, error: Exception):
        logger.exception("Unhandled service error", exc_info=error)
        return JSONResponse(status_code=500, content={"detail": "Internal service error; see logs"})

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest, request: Request):
        if "embedding_model" not in body.model_fields_set:
            body = IngestRequest.model_validate(
                body.model_dump() | {"embedding_model": settings.DEFAULT_EMBED_MODEL}
            )
        return request.app.state.pipeline.ingest(body)

    @app.post("/rag", response_model=RAGResponse)
    def rag(body: RAGRequest, request: Request):
        defaults = {
            "llm_model": settings.DEFAULT_LLM_MODEL,
            "embedding_model": settings.DEFAULT_EMBED_MODEL,
            "top_k": settings.DEFAULT_TOP_K,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }
        body = RAGRequest.model_validate(
            body.model_dump()
            | {key: value for key, value in defaults.items() if key not in body.model_fields_set}
        )
        return request.app.state.pipeline.query(body)

    @app.post("/generate", response_model=DirectGenerateResponse)
    def generate(body: DirectGenerateRequest, request: Request):
        defaults = {
            "model": settings.DEFAULT_LLM_MODEL,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }
        body = DirectGenerateRequest.model_validate(
            body.model_dump()
            | {key: value for key, value in defaults.items() if key not in body.model_fields_set}
        )
        return request.app.state.pipeline.generate(body)

    @app.get("/models", response_model=ModelListResponse)
    def models(request: Request):
        return request.app.state.pipeline.models()

    @app.get("/health")
    def health(request: Request):
        import torch

        pipeline = request.app.state.pipeline
        errors = []
        installed = []
        try:
            installed = pipeline.models().ollama_models
        except (httpx.HTTPError, ConnectionError) as error:
            errors.append(f"Ollama: {error}")
        if settings.DEFAULT_LLM_MODEL not in installed:
            errors.append(f"Ollama model missing: {settings.DEFAULT_LLM_MODEL}")
        for name in {
            settings.DEFAULT_EMBED_MODEL,
            settings.DEFAULT_RERANKER_MODEL,
            "intfloat/multilingual-e5-large",
        }:
            if not (settings.MODEL_DIR / sanitize(name) / "config.json").is_file():
                errors.append(f"Local model missing: {name}")
        gpu: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                gpu.append(
                    {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "free_bytes": free,
                        "total_bytes": total,
                    }
                )
        body = {
            "status": "ready" if not errors else "degraded",
            "errors": errors,
            "storage_heartbeat": pipeline.client.heartbeat(),
            "cpu_count": psutil.cpu_count(),
            "ram_available_bytes": psutil.virtual_memory().available,
            "device": settings.DEVICE,
            "cuda": gpu,
            "mps_available": torch.backends.mps.is_available(),
        }
        return JSONResponse(status_code=200 if not errors else 503, content=body)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = Settings()
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, workers=1)
