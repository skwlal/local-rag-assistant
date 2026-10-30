import json

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from app.rag_engine import LocalRAGPipeline, citation_warnings, collection_name, strip_thinking
from app.schemas import IngestRequest, RAGRequest, SourceNodeInfo


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


@pytest.fixture
def pipeline(settings, monkeypatch):
    monkeypatch.setattr("app.rag_engine.local_tokenizer", lambda settings: WordTokenizer())
    monkeypatch.setattr(
        LocalRAGPipeline, "embedding", lambda self, model: MockEmbedding(embed_dim=8)
    )
    monkeypatch.setattr(LocalRAGPipeline, "llm", lambda *args: MockLLM())
    for i, text in enumerate(
        [
            "Градиент оптимизация минимум",
            "Вероятность событие вероятность",
            "Матрица вектор алгебра",
            "Алгоритм граф ребро",
            "Теорема доказательство",
        ]
    ):
        (settings.RAW_DATA_DIR / f"lecture{i}.md").write_text(text, encoding="utf-8")
    result = LocalRAGPipeline(settings)
    yield result
    result.close()


def test_index_survives_restart_and_bm25_rebuilds(pipeline):
    request = IngestRequest()
    built = pipeline.ingest(request)
    assert built.total_nodes == 5
    restored = LocalRAGPipeline(pipeline.settings)
    try:
        index, manifest = restored.load_index(request)
        assert manifest["collection_name"] == built.collection_name
        dense = restored.get_retriever(index, "dense", 3).retrieve("матрица")
        hybrid = restored.get_retriever(index, "hybrid", 5).retrieve("вероятность")
        assert len(dense) == 3
        assert len(hybrid) == 5
        assert {n.node.metadata["file_name"] for n in hybrid} == {
            f"lecture{i}.md" for i in range(5)
        }
    finally:
        restored.close()


def test_hybrid_search_needs_no_default_llm(pipeline, monkeypatch):
    pipeline.ingest(IngestRequest())
    index, _ = pipeline.load_index(IngestRequest())

    def unexpected(*args, **kwargs):
        raise AssertionError("Hybrid retrieval must not load an LLM")

    monkeypatch.setattr(pipeline, "llm", unexpected)
    monkeypatch.setattr(MockLLM, "complete", unexpected)
    found = pipeline.get_retriever(index, "hybrid", 3).retrieve("вероятность")
    # MockEmbedding gives every document the same vector: ranks can tie.
    assert len(found) == 3


def test_reingest_replaces_deleted_documents(pipeline):
    old = pipeline.ingest(IngestRequest())
    (pipeline.settings.RAW_DATA_DIR / "lecture0.md").unlink()
    new = pipeline.ingest(IngestRequest())
    assert new.total_nodes == 4
    assert old.collection_name != new.collection_name
    index, _ = pipeline.load_index(IngestRequest())
    assert {n.metadata["file_name"] for n in index.docstore.docs.values()} == {
        f"lecture{i}.md" for i in range(1, 5)
    }


def test_failed_rebuild_preserves_active_manifest(pipeline, monkeypatch):
    previous = pipeline.ingest(IngestRequest())

    def failed(*args, **kwargs):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(pipeline, "build_index", failed)
    with pytest.raises(RuntimeError):
        pipeline.ingest(IngestRequest())
    assert pipeline.manifest(IngestRequest())["collection_name"] == previous.collection_name


def test_query_logs_sources_and_timings(pipeline):
    pipeline.ingest(IngestRequest())
    result = pipeline.query(RAGRequest(question="Что такое матрица?", retrieval_mode="dense"))
    assert len(result.sources) == 3
    assert result.metrics.total_latency_s >= result.metrics.retrieval_latency_s
    log = json.loads((pipeline.settings.LOG_DIR / "queries.jsonl").read_text().splitlines()[0])
    assert log["request"]["question"] == "Что такое матрица?"
    assert len(log["response"]["sources"]) == 3


def test_collection_identity_includes_model_strategy_size_and_overlap():
    base = IngestRequest()
    configs = [
        base,
        base.model_copy(update={"chunk_overlap": 32}),
        base.model_copy(update={"embedding_model": "intfloat/multilingual-e5-large"}),
        base.model_copy(update={"chunk_strategy": "naive"}),
        base.model_copy(update={"chunk_size": 256}),
    ]
    assert len({collection_name(c) for c in configs}) == len(configs)


def test_reasoning_and_citations():
    assert strip_thinking("<think>secret\nreason</think>Ответ") == "Ответ"
    assert strip_thinking("<think>unfinished") == ""
    sources = [SourceNodeInfo(doc_name="a", chunk_id="1", text="Точная цитата", score=0.5)]
    assert not citation_warnings("«Точная цитата» [S1]", sources)
    assert len(citation_warnings("«Выдумка» [S9]", sources)) == 2
