import httpx
import pytest
from llama_index.core.llms import MockLLM

from app.config import sanitize
from app.rag_engine import LocalRAGPipeline
from app.schemas import IngestRequest, RAGRequest


def test_real_embedding_and_reranker_interfaces_without_downloads(settings, monkeypatch):
    """Exercise neural interfaces using tiny random local weights."""
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "query",
        "passage",
        ":",
        "matrix",
        "vector",
        "probability",
        "graph",
        "method",
    ]
    settings.MODEL_DIR.mkdir(parents=True)
    vocab_path = settings.MODEL_DIR / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
    )
    for name in ("BAAI/bge-m3", "intfloat/multilingual-e5-large", "BAAI/bge-reranker-v2-m3"):
        path = settings.MODEL_DIR / sanitize(name)
        model = BertForSequenceClassification(config) if "reranker" in name else BertModel(config)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    for i, text in enumerate(["matrix vector", "probability method", "graph method"]):
        (settings.RAW_DATA_DIR / f"doc{i}.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(LocalRAGPipeline, "llm", lambda *args: MockLLM())
    pipeline = LocalRAGPipeline(settings)
    try:
        for name in ("BAAI/bge-m3", "intfloat/multilingual-e5-large"):
            assert len(pipeline.embedding(name).get_query_embedding("matrix")) == 16
            pipeline.ingest(IngestRequest(embedding_model=name))
            result = pipeline.query(
                RAGRequest(
                    question="matrix",
                    embedding_model=name,
                    retrieval_mode="hybrid_rerank",
                    top_k=2,
                )
            )
            assert len(result.sources) == 2
            assert all(source.text for source in result.sources)
    finally:
        pipeline.close()


def test_pipeline_rejects_remote_model(settings):
    pipeline = LocalRAGPipeline(settings)
    pipeline.http.close()
    pipeline.http = httpx.Client(
        base_url=settings.OLLAMA_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"remote_model": "remote-llm"}),
        ),
    )
    try:
        with pytest.raises(ValueError, match="Remote Ollama"):
            pipeline.llm("remote-llm", 0)
    finally:
        pipeline.close()
