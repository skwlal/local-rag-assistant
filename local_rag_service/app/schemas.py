from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EmbeddingModel = Literal["BAAI/bge-m3", "intfloat/multilingual-e5-large"]
ChunkStrategy = Literal["naive", "structural"]
RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank"]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class IngestRequest(RequestModel):
    chunk_strategy: ChunkStrategy = "structural"
    chunk_size: int = Field(default=512, ge=64, le=512)
    chunk_overlap: int = Field(default=64, ge=0)
    embedding_model: EmbeddingModel = "BAAI/bge-m3"

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size - 16:
            raise ValueError("chunk_overlap must be less than chunk_size - 16")
        return self


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    total_nodes: int
    elapsed_s: float
    collection_name: str
    corpus_hash: str


class SourceNodeInfo(BaseModel):
    doc_name: str
    chunk_id: str
    text: str
    score: float


class RAGMetrics(BaseModel):
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    reranking_latency_s: float = 0.0


class RAGRequest(IngestRequest):
    question: str = Field(min_length=1, max_length=8000)
    llm_model: str = "qwen3.5:9b"
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = Field(default=3, ge=1, le=15)
    temperature: float = Field(default=0.1, ge=0, le=2)


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceNodeInfo]
    metrics: RAGMetrics
    index_version: str
    citation_warnings: list[str] = Field(default_factory=list)


class DirectGenerateRequest(RequestModel):
    prompt: str = Field(min_length=1, max_length=32000)
    model: str = "qwen3.5:9b"
    temperature: float = Field(default=0.1, ge=0, le=2)


class DirectGenerateResponse(BaseModel):
    response: str
    latency_s: float


class ModelListResponse(BaseModel):
    ollama_models: list[str]
    embedding_models: list[str]
