from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.schemas import DirectGenerateResponse, IngestRequest, ModelListResponse


class StubPipeline:
    def __init__(self, settings):
        self.client = SimpleNamespace(heartbeat=lambda: 123)
        self.closed = False

    def models(self):
        return ModelListResponse(ollama_models=["qwen3.5:4b"], embedding_models=[])

    def generate(self, request):
        if request.prompt == "timeout":
            raise httpx.ReadTimeout("Ollama timed out")
        return DirectGenerateResponse(response="Ответ", latency_s=0.2)

    def query(self, request):
        raise FileNotFoundError("Index missing; call /ingest first")

    def close(self):
        self.closed = True


def test_api_validation_errors_health_and_lifespan(settings):
    app = create_app(settings, StubPipeline)
    with TestClient(app) as client:
        assert client.post("/rag", json={"question": " "}).status_code == 422
        assert (
            client.post("/ingest", json={"chunk_size": 64, "chunk_overlap": 64}).status_code == 422
        )
        assert client.post("/rag", json={"question": "Test"}).status_code == 409
        assert client.post("/generate", json={"prompt": "timeout"}).status_code == 504
        assert client.post("/generate", json={"prompt": "Test"}).json()["response"] == "Ответ"
        assert client.get("/models").json()["ollama_models"] == ["qwen3.5:4b"]
        assert client.get("/health").status_code == 503
    assert app.state.pipeline.closed


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://8.8.8.8", "http://localhost/x"])
def test_remote_ollama_is_rejected(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, OLLAMA_BASE_URL=url)


def test_overlap_validation():
    with pytest.raises(ValidationError):
        IngestRequest(chunk_size=128, chunk_overlap=120)


def test_request_defaults_preserve_explicit_values_and_validate():
    from app.schemas import RAGRequest

    request = RAGRequest(question="Question", temperature=0, top_k=3)
    updated = request.with_defaults(temperature=0.7, top_k=5, llm_model="gemma4:12b")
    assert updated.temperature == 0
    assert updated.top_k == 3
    assert updated.llm_model == "gemma4:12b"
    assert request.llm_model == "qwen3.5:9b"
    with pytest.raises(ValidationError):
        RAGRequest(question="Question").with_defaults(top_k=0)


def test_pipeline_closed_when_required_ollama_startup_fails(settings):
    class UnavailablePipeline(StubPipeline):
        def models(self):
            raise httpx.ConnectError("Offline")

    settings.REQUIRE_OLLAMA_ON_STARTUP = True
    app = create_app(settings, UnavailablePipeline)
    with pytest.raises(httpx.ConnectError), TestClient(app):
        pytest.fail("Startup should fail")
    assert app.state.pipeline.closed
