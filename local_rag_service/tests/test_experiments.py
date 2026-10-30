import asyncio
import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from pydantic import BaseModel

from app.evaluation import LocalJudge, retrieval_metrics
from app.storage import write_json
from experiments import (
    ExperimentConfig,
    evaluate_runs,
    experiment_configs,
    export_reports,
    generate_runs,
    load_questions,
    plot_results,
)


def test_factorial_grid():
    configs = experiment_configs()
    assert len(configs) == 25
    assert [len(experiment_configs(f"exp{i}")) for i in range(1, 5)] == [4, 5, 12, 4]
    assert len({(c.experiment, c.name) for c in configs}) == 25
    rag = [c for c in configs if c.experiment == "exp3" and c.mode == "rag"]
    assert all(c.top_k == 3 and c.retrieval_mode == "hybrid_rerank" for c in rag)


def test_retrieval_metrics_are_document_level_and_respect_depth():
    sources = [{"doc_name": name} for name in ["a", "b", "target", "target", "e"]]
    assert retrieval_metrics(sources, "target", 5) == {
        "hit_rate_at_3": 1,
        "hit_rate_at_5": 1,
        "mrr": 1 / 3,
    }
    assert retrieval_metrics(sources[:3], "target", 3)["hit_rate_at_5"] is None
    assert retrieval_metrics(sources, "absent", 5)["mrr"] == 0
    assert retrieval_metrics([], "target", 3)["hit_rate_at_3"] is None


def test_questions_reject_missing_fields_and_duplicates(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text("question_id,question\n1,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_questions(path)
    path.write_text(
        "question_id,question,ground_truth_answer,expected_source_doc\n"
        "1,q,a,doc.md\n1,q2,a2,doc.md\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_questions(path)


def test_csv_aggregation_and_four_plots_preserve_na(tmp_path):
    for i, config in enumerate(experiment_configs()):
        write_json(
            tmp_path / "checkpoints" / f"{i}.json",
            {
                "experiment": config.experiment,
                "config": config.name,
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model,
                "mode": config.mode,
                "top_k": config.top_k,
                "status": "evaluated",
                "sources": [],
                "citation_warnings": [],
                "hit_rate_at_3": 1.0 if config.mode == "rag" else None,
                "hit_rate_at_5": 1.0 if config.top_k == 5 else None,
                "mrr": 0.5,
                "faithfulness": 0.8 if config.mode == "rag" else None,
                "semantic_similarity": 0.75,
                "retrieval_latency_s": 0.1,
                "generation_latency_s": 0.2,
            },
        )
    frame = export_reports(tmp_path)
    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    direct = summary[summary["mode"] == "direct"]
    assert direct.faithfulness_mean.isna().all()
    assert direct.faithfulness_count.eq(0).all()
    plot_results(frame, tmp_path / "plots")
    paths = list((tmp_path / "plots").glob("*.png"))
    assert len(paths) == 4
    from PIL import Image

    for path in paths:
        with Image.open(path) as image:
            assert image.width >= 3000
            assert abs(image.info["dpi"][0] - 300) < 1


def test_native_ollama_judge_schema_and_retry(settings):
    class Verdict(BaseModel):
        value: int

    requests = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"family": "qwen3.5"}})
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.host == "localhost"
        assert payload["options"]["temperature"] == 0
        assert payload["model"] == settings.JUDGE_MODEL
        assert payload["think"] is False
        assert "value" in payload["format"]["properties"]
        text = "invalid" if len(requests) == 1 else '{"value": 1}'
        return httpx.Response(200, json={"message": {"content": text}})

    async def run():
        judge = LocalJudge(settings)
        await judge.client.aclose()
        judge.client = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            transport=httpx.MockTransport(handler),
        )
        try:
            result = await judge.agenerate("Evaluate", Verdict)
            assert result.value == 1
            assert len(requests) == 2
        finally:
            await judge.close()

    asyncio.run(run())


def test_resume_keeps_completed_answers_and_retries_failed_evaluation(
    settings, tmp_path, monkeypatch
):
    from app.schemas import DirectGenerateResponse, ModelListResponse

    calls = []

    class Pipeline:
        def __init__(self, settings):
            self.settings = settings

        def models(self):
            return ModelListResponse(ollama_models=["qwen3.5:9b"], embedding_models=[])

        def generate(self, request):
            calls.append(request.prompt)
            return DirectGenerateResponse(response="Ответ", latency_s=0.1)

        def close(self):
            return None

    class Scorer:
        def __init__(self, settings):
            self.settings = settings

        async def evaluate(self, question, answer, reference, contexts):
            if len(evaluations) == 0:
                evaluations.append("failed")
                raise httpx.ReadTimeout("Temporary judge failure")
            evaluations.append("success")
            return {"faithfulness": None, "semantic_similarity": 0.9, "answer_relevance": 0.8}

        async def close(self):
            return None

    monkeypatch.setattr("experiments.LocalRAGPipeline", Pipeline)
    monkeypatch.setattr("app.evaluation.Evaluator", Scorer)
    (settings.RAW_DATA_DIR / "doc.md").write_text("Источник", encoding="utf-8")
    args = SimpleNamespace(
        output_dir=tmp_path / "results", repeats=1, warmup=False, temperature=0.1, phase="generate"
    )
    questions = [
        {
            "question_id": "q1",
            "question": "Вопрос",
            "ground_truth_answer": "Ответ",
            "expected_source_doc": "doc.md",
        }
    ]
    configs = [ExperimentConfig("exp3", "direct_test", mode="direct")]
    generate_runs(args, configs, questions, settings)
    generate_runs(args, configs, questions, settings)
    assert len(calls) == 1
    evaluations = []
    asyncio.run(evaluate_runs(args, settings))
    failed = export_reports(args.output_dir)
    assert failed.status.tolist() == ["evaluation_error"]
    asyncio.run(evaluate_runs(args, settings))
    asyncio.run(evaluate_runs(args, settings))
    complete = export_reports(args.output_dir)
    assert complete.status.tolist() == ["evaluated"]
    assert evaluations == ["failed", "success"]
    assert complete.faithfulness.isna().all()


def test_judge_refuses_remote_model_before_sending_prompt(settings):
    class Verdict(BaseModel):
        score: int

    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"remote_host": "https://ollama.com"})

    async def run():
        judge = LocalJudge(settings)
        await judge.client.aclose()
        judge.client = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(ValueError, match="local Ollama"):
                await judge.agenerate("Private question", Verdict)
        finally:
            await judge.close()

    asyncio.run(run())
    assert requests == ["/api/show"]


def test_run_manifest_rejects_changed_inputs_on_resume(settings, tmp_path, monkeypatch):
    from app.storage import digest
    from experiments import main

    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,ground_truth_answer,expected_source_doc\n1,Question,Answer,doc.md\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("experiments.Settings", lambda: settings)
    monkeypatch.setattr(
        "experiments.run_manifest",
        lambda args, *rest: {
            "dataset": digest(args.data_path.read_bytes()),
        },
    )
    monkeypatch.setattr("experiments.generate_runs", lambda *args: None)
    args = [
        "--data-path",
        str(path),
        "--output-dir",
        str(tmp_path / "results"),
        "--phase",
        "generate",
    ]
    main(args)
    main(args + ["--resume"])
    with pytest.raises(FileExistsError):
        main(args)
    path.write_text(path.read_text().replace("Question", "Changed"))
    with pytest.raises(ValueError, match="changed"):
        main(args + ["--resume"])


def test_model_subset_and_baseline():
    configs = experiment_configs("all", baseline="gemma4:12b", models=["qwen3.5:9b"])
    assert len(configs) == 15
    assert {c.llm_model for c in configs if c.experiment != "exp3"} == {"gemma4:12b"}
    assert {c.llm_model for c in configs if c.experiment == "exp3"} == {"qwen3.5:9b"}


def test_generation_and_evaluation_have_separate_model_requirements(
    settings, tmp_path, monkeypatch
):
    from experiments import evaluation_manifest, run_manifest

    installed = ["qwen3.5:9b"]
    real_client = httpx.Client

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": name, "digest": "hash-" + name, "details": {}}
                        for name in installed
                    ]
                },
            )
        return httpx.Response(200, json={"version": "test"})

    monkeypatch.setattr(
        "experiments.httpx.Client",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)),
    )
    dataset = tmp_path / "questions.csv"
    dataset.write_text("test")
    args = SimpleNamespace(
        phase="generate",
        data_path=dataset,
        output_dir=tmp_path,
        repeats=1,
        temperature=0.1,
        warmup=False,
    )
    configs = experiment_configs("exp3", models=["qwen3.5:9b"])
    manifest = run_manifest(args, configs, settings)
    assert set(manifest["ollama_models"]) == {"qwen3.5:9b"}
    write_json(tmp_path / "run_manifest.json", manifest)
    installed[:] = [settings.JUDGE_MODEL]
    args.phase = "evaluate"
    assert run_manifest(args, configs, settings) == manifest
    assert evaluation_manifest(settings)["judge"] == settings.JUDGE_MODEL
    args.phase = "generate"
    with pytest.raises(ValueError, match="models missing"):
        run_manifest(args, configs, settings)


def test_reference_review_status_survives_loading(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,ground_truth_answer,expected_source_doc,review_status\n"
        "1,Q,A,doc.pdf,ai_source_checked\n"
    )
    assert load_questions(path)[0]["review_status"] == "ai_source_checked"
