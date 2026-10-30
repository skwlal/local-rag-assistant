import argparse
import asyncio
import gc
import importlib.metadata
import json
import logging
import platform
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter

import httpx
import pandas as pd
import psutil
from filelock import FileLock
from ollama import ResponseError

import app  # noqa: F401 -- sets offline policy before ML libraries are imported
from app.config import BASE_DIR, LLM_MODELS, Settings
from app.ingestion import corpus_fingerprint
from app.rag_engine import LocalRAGPipeline, collection_name, strip_thinking
from app.schemas import DirectGenerateRequest, IngestRequest, RAGRequest
from app.storage import atomic_write, digest, read_json, write_json

logger = logging.getLogger(__name__)
METRICS = [
    "hit_rate_at_3",
    "hit_rate_at_5",
    "mrr",
    "semantic_similarity",
    "faithfulness",
    "answer_relevance",
    "retrieval_latency_s",
    "reranking_latency_s",
    "generation_latency_s",
    "total_latency_s",
    "evaluation_latency_s",
    "rss_bytes",
]
QUESTION_COLUMNS = ["question_id", "question", "ground_truth_answer", "expected_source_doc"]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    name: str
    chunk_strategy: str = "structural"
    chunk_size: int = 512
    embedding_model: str = "BAAI/bge-m3"
    retrieval_mode: str = "dense"
    top_k: int = 5
    llm_model: str = "qwen3.5:9b"
    mode: str = "rag"

    def request(self, question: str, temperature: float = 0.1) -> RAGRequest:
        return RAGRequest(
            question=question,
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            embedding_model=self.embedding_model,
            retrieval_mode=self.retrieval_mode,
            top_k=self.top_k,
            llm_model=self.llm_model,
            temperature=temperature,
        )

    def ingest_request(self) -> IngestRequest:
        return IngestRequest(
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            embedding_model=self.embedding_model,
        )


def experiment_configs(
    experiment: str = "all", baseline: str = "qwen3.5:9b", models: list[str] | None = None
) -> list[ExperimentConfig]:
    configs = []
    for strategy in ("naive", "structural"):
        for size in (256, 512):
            configs.append(ExperimentConfig("exp1", f"{strategy}_{size}", strategy, size))
    for mode, top_k, name in [
        ("dense", 3, "dense_top3"),
        ("dense", 5, "dense_top5"),
        ("hybrid", 5, "hybrid_top5"),
        ("hybrid_rerank", 3, "hybrid_rerank_15_to_3"),
        ("hybrid_rerank", 5, "hybrid_rerank_15_to_5"),
    ]:
        configs.append(ExperimentConfig("exp2", name, retrieval_mode=mode, top_k=top_k))
    for model in models or LLM_MODELS:
        for mode in ("rag", "direct"):
            configs.append(
                ExperimentConfig(
                    "exp3",
                    f"{model}_{mode}",
                    retrieval_mode="hybrid_rerank",
                    top_k=3,
                    llm_model=model,
                    mode=mode,
                )
            )
    for model in ("BAAI/bge-m3", "intfloat/multilingual-e5-large"):
        for top_k in (3, 5):
            configs.append(
                ExperimentConfig(
                    "exp4",
                    f"{model.split('/')[-1]}_top{top_k}",
                    embedding_model=model,
                    top_k=top_k,
                )
            )
    return [
        replace(config, llm_model=baseline) if config.experiment != "exp3" else config
        for config in configs
        if experiment in {"all", config.experiment}
    ]


def load_questions(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(QUESTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Question CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Question CSV contains no questions")
    for column in QUESTION_COLUMNS:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Question CSV contains empty {column}")
    if frame.question_id.duplicated().any():
        raise ValueError("question_id must be unique")
    for value in frame.expected_source_doc:
        source = PurePosixPath(value)
        if source.is_absolute() or ".." in source.parts or "\\" in value:
            raise ValueError("expected_source_doc must be a relative POSIX path inside data/raw")
    if "review_status" in frame:
        logger.warning(
            "Reference answers include review statuses: %s",
            frame.review_status.value_counts().to_dict(),
        )
    columns = QUESTION_COLUMNS + [
        c for c in ("review_status", "expert_reviewer", "expert_reviewed_at") if c in frame
    ]
    return frame[columns].to_dict("records")


def checkpoint_key(config: ExperimentConfig, question_id: str, repeat: int) -> str:
    return digest(json.dumps([asdict(config), question_id, repeat], sort_keys=True))


def export_reports(output: Path) -> pd.DataFrame:
    records = [read_json(path) for path in sorted((output / "checkpoints").glob("*.json"))]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    exported = frame.copy()
    for name in ("sources", "citation_warnings"):
        if name in exported:
            exported[name] = exported[name].map(lambda x: json.dumps(x, ensure_ascii=False))
    atomic_write(output / "evaluation_results_raw.csv", exported.to_csv(index=False))
    for name in METRICS:
        if name not in frame:
            frame[name] = float("nan")
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    keys = ["experiment", "config", "llm_model", "embedding_model", "mode", "top_k"]
    grouped = frame.groupby(keys, dropna=False)
    summary = grouped[METRICS].agg(["mean", "median", "count"])
    summary.columns = ["_".join(name) for name in summary.columns]
    summary["attempts"] = grouped.size()
    summary["completed"] = grouped["status"].agg(lambda x: int(x.eq("evaluated").sum()))
    atomic_write(output / "summary_metrics.csv", summary.reset_index().to_csv(index=False))
    return frame


def plot_results(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    def bars(ax, data, metrics, title):
        tidy = data.melt(
            id_vars=["config"], value_vars=metrics, var_name="Metric", value_name="Score"
        )
        finite = tidy.dropna(subset=["Score"])
        if finite.empty:
            ax.text(0.5, 0.5, "No applicable measurements", ha="center", transform=ax.transAxes)
        else:
            present = set(data["config"])
            order = [c.name for c in experiment_configs() if c.name in present]
            sns.barplot(
                data=finite, x="config", y="Score", hue="Metric", errorbar=None, ax=ax, order=order
            )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    specs = {
        "exp1": ("exp1_chunking_strategy.png", ["hit_rate_at_5", "mrr"], "Chunking comparison"),
        "exp4": (
            "exp4_embeddings_benchmark.png",
            ["hit_rate_at_3", "hit_rate_at_5", "mrr"],
            "Embeddings (Hit@5 is N/A for top-3)",
        ),
    }
    for experiment, (filename, metrics, title) in specs.items():
        data = frame[frame.experiment == experiment]
        if not data.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            bars(ax, data, metrics, title)
            fig.tight_layout()
            fig.savefig(output / filename, dpi=300)
            plt.close(fig)
    data = frame[frame.experiment == "exp2"]
    if not data.empty:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        bars(axes[0], data, ["hit_rate_at_3", "faithfulness"], "Retrieval quality (common k=3)")
        bars(axes[1], data, ["retrieval_latency_s", "generation_latency_s"], "Latency, seconds")
        fig.tight_layout()
        fig.savefig(output / "exp2_retrieval_comparison.png", dpi=300)
        plt.close(fig)
    data = frame[frame.experiment == "exp3"]
    if not data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, metric in zip(axes, ["faithfulness", "semantic_similarity"], strict=True):
            finite = data.dropna(subset=[metric])
            if not finite.empty:
                sns.barplot(
                    data=finite,
                    x="llm_model",
                    y=metric,
                    hue="mode",
                    order=list(LLM_MODELS),
                    hue_order=["rag", "direct"],
                    palette={"rag": "#4378a8", "direct": "#ce854c"},
                    errorbar=None,
                    ax=ax,
                )
            ax.tick_params(axis="x", rotation=25)
            ax.set_xlabel("")
            ax.set_title(metric)
            ax.set_ylim(0 if metric == "faithfulness" else -1, 1)
        axes[0].set_title("Faithfulness: direct mode N/A (no context)")
        fig.tight_layout()
        fig.savefig(output / "exp3_rag_vs_raw_llm.png", dpi=300)
        plt.close(fig)


def perform_query(pipeline, config, question, temperature):
    if config.mode == "direct":
        direct = pipeline.generate(
            DirectGenerateRequest(
                prompt=f"Ответь по-русски на учебный вопрос.\n{question}",
                model=config.llm_model,
                temperature=temperature,
            )
        )
        return {
            "answer_raw": direct.response,
            "sources": [],
            "citation_warnings": [],
            "retrieval_latency_s": None,
            "reranking_latency_s": None,
            "generation_latency_s": direct.latency_s,
            "total_latency_s": direct.latency_s,
            "index_version": None,
        }
    response = pipeline.query(config.request(question, temperature))
    return {
        "answer_raw": response.answer,
        "sources": [source.model_dump() for source in response.sources],
        "citation_warnings": response.citation_warnings,
        "index_version": response.index_version,
        **response.metrics.model_dump(),
    }


def generate_runs(args, configs, questions, settings):
    from app.evaluation import retrieval_metrics

    with closing(LocalRAGPipeline(settings)) as pipeline:
        installed = pipeline.models().ollama_models
        required = {config.llm_model for config in configs}
        if args.phase == "all":
            required.add(settings.JUDGE_MODEL)
        if missing := required - set(installed):
            raise ValueError(f"Ollama models missing: {sorted(missing)}")
        expected_corpus = corpus_fingerprint(settings)
        if any(not (settings.RAW_DATA_DIR / q["expected_source_doc"]).is_file() for q in questions):
            raise ValueError("expected_source_doc must name an existing file relative to data/raw")
        prepared = set()
        version_path = args.output_dir / "index_versions.json"
        pinned = read_json(version_path) if version_path.exists() else {}
        model_manifest_path = settings.MODEL_DIR / "manifest.json"
        model_hash = (
            digest(model_manifest_path.read_bytes()) if model_manifest_path.exists() else None
        )
        for config in configs:
            if corpus_fingerprint(settings) != expected_corpus:
                raise ValueError("Corpus changed during the experiment; restore it before resuming")
            ingest = config.ingest_request()
            logical_name = collection_name(ingest)
            if config.mode == "rag" and logical_name not in prepared:
                path = settings.INDEX_DIR / f"{logical_name}.json"
                active = read_json(path) if path.exists() else {}
                if logical_name in pinned and active.get("collection_name") != pinned[logical_name]:
                    raise ValueError(
                        "Active index changed during this run; use a new output directory"
                    )
                if (
                    active.get("corpus_hash") != expected_corpus
                    or active.get("models_manifest_hash") != model_hash
                ):
                    logger.info("Indexing %s", logical_name)
                    pipeline.ingest(ingest)
                active = read_json(path)
                pinned[logical_name] = active["collection_name"]
                if active["corpus_hash"] != expected_corpus:
                    raise ValueError("Corpus changed while preparing an experiment index")
                write_json(version_path, pinned)
                prepared.add(logical_name)
            pending = [
                (
                    question,
                    repeat,
                    args.output_dir
                    / "checkpoints"
                    / (checkpoint_key(config, question["question_id"], repeat) + ".json"),
                )
                for repeat in range(args.repeats)
                for question in questions
            ]
            pending = [
                item
                for item in pending
                if not item[2].exists() or read_json(item[2])["status"] == "generation_error"
            ]
            if pending and args.warmup:
                logger.info("Warming %s", config.name)
                perform_query(pipeline, config, questions[0]["question"], args.temperature)
            for question, repeat, path in pending:
                row = {
                    **question,
                    "experiment": config.experiment,
                    "config": config.name,
                    "llm_model": config.llm_model,
                    "embedding_model": config.embedding_model,
                    "mode": config.mode,
                    "top_k": config.top_k,
                    "repeat": repeat,
                    "temperature": args.temperature,
                    "seed": settings.SEED,
                    "created_at": datetime.now(UTC).isoformat(),
                    "status": "generated",
                    "error": "",
                    **dict.fromkeys(METRICS),
                }
                logger.info(
                    "%s question=%s repeat=%s", config.name, question["question_id"], repeat
                )
                try:
                    result = perform_query(pipeline, config, question["question"], args.temperature)
                    row.update(result)
                    row["answer"] = strip_thinking(row["answer_raw"])
                    if config.mode == "rag":
                        row.update(
                            retrieval_metrics(
                                row["sources"],
                                question["expected_source_doc"],
                                config.top_k,
                            )
                        )
                except (httpx.HTTPError, ResponseError, ConnectionError, RuntimeError) as error:
                    row.update(status="generation_error", error=f"{type(error).__name__}: {error}")
                    logger.error("Generation failed: %s", error)
                row["rss_bytes"] = psutil.Process().memory_info().rss
                write_json(path, row)
            export_reports(args.output_dir)
    del pipeline
    gc.collect()


async def evaluate_runs(args, settings):
    from app.evaluation import Evaluator

    evaluator = Evaluator(settings)
    try:
        for path in sorted((args.output_dir / "checkpoints").glob("*.json")):
            row = read_json(path)
            if row["status"] not in {"generated", "evaluation_error"}:
                continue
            start = perf_counter()
            try:
                values = await evaluator.evaluate(
                    row["question"],
                    row["answer"],
                    row["ground_truth_answer"],
                    [source["text"] for source in row["sources"]],
                )
                row.update(values, status="evaluated", error="")
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                row.update(status="evaluation_error", error=f"{type(error).__name__}: {error}")
                logger.error("Evaluation failed: %s", error)
            row["evaluation_latency_s"] = perf_counter() - start
            write_json(path, row)
    finally:
        await evaluator.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local RAG factorial experiments")
    parser.add_argument("--data-path", type=Path, default=BASE_DIR / "data/test_questions.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--experiment", choices=["all", "exp1", "exp2", "exp3", "exp4"], default="all"
    )
    parser.add_argument("--phase", choices=["all", "generate", "evaluate"], default="all")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model", default="qwen3.5:9b", help="Baseline for exp1, exp2, exp4")
    parser.add_argument("--models", nargs="+", choices=LLM_MODELS, help="Subset for exp3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.repeats < 1 or not 0 <= args.temperature <= 2:
        parser.error("repeats must be positive and temperature must be in [0, 2]")
    return args


def run_manifest(args, configs, settings):
    packages = [
        "fastapi",
        "docling",
        "chromadb",
        "llama-index-core",
        "sentence-transformers",
        "torch",
        "ragas",
        "transformers",
        "numpy",
        "pandas",
    ]
    model_manifest = settings.MODEL_DIR / "manifest.json"
    sources = [Path(__file__), *sorted((BASE_DIR / "app").glob("*.py"))]
    with httpx.Client(base_url=settings.OLLAMA_BASE_URL, timeout=10, trust_env=False) as client:
        response = client.get("/api/tags")
        response.raise_for_status()
        required = {config.llm_model for config in configs}
        if args.phase == "evaluate":
            required = set()
        ollama_models = {
            item["name"]: {"digest": item["digest"], "details": item["details"]}
            for item in response.json()["models"]
            if item["name"] in required
        }
        if missing := required - set(ollama_models):
            raise ValueError(f"Ollama models missing: {sorted(missing)}")
        version = client.get("/api/version")
        version.raise_for_status()
    result = {
        "ollama_version": version.json()["version"],
        "ollama_models": ollama_models,
        "dataset_sha256": digest(args.data_path.read_bytes()),
        "corpus_sha256": corpus_fingerprint(settings),
        "models_manifest": read_json(model_manifest) if model_manifest.exists() else None,
        "configs": [asdict(config) for config in configs],
        "repeats": args.repeats,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "settings": settings.model_dump(mode="json", exclude={"JUDGE_MODEL"}),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_hash": digest(b"".join(path.read_bytes() for path in sources)),
    }

    if args.phase == "evaluate":
        previous = read_json(args.output_dir / "run_manifest.json")
        for field in ("ollama_version", "ollama_models"):
            result[field] = previous[field]
    return result


def evaluation_manifest(settings):
    with httpx.Client(base_url=settings.OLLAMA_BASE_URL, timeout=10, trust_env=False) as client:
        tags = client.get("/api/tags")
        tags.raise_for_status()
        model = next((m for m in tags.json()["models"] if m["name"] == settings.JUDGE_MODEL), None)
        if model is None:
            raise ValueError(f"Judge model missing: {settings.JUDGE_MODEL}")
        version = client.get("/api/version")
        version.raise_for_status()
    return {
        "judge": model["name"],
        "digest": model["digest"],
        "ollama_version": version.json()["version"],
    }


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir / "run.lock"), timeout=0):
        execute_run(args)


def execute_run(args):
    questions = load_questions(args.data_path)
    settings = Settings()
    configs = experiment_configs(args.experiment, args.model, args.models)
    manifest = run_manifest(args, configs, settings)
    path = args.output_dir / "run_manifest.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError("Run already exists; use --resume or a new --output-dir")
        if read_json(path) != manifest:
            raise ValueError("Run inputs or environment changed; use a new --output-dir")
    else:
        if args.phase == "evaluate":
            raise FileNotFoundError("Run generation before evaluation")
        write_json(path, manifest)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    if args.phase in {"all", "generate"}:
        generate_runs(args, configs, questions, settings)
    if args.phase in {"all", "evaluate"}:
        judge_manifest = evaluation_manifest(settings)
        judge_path = args.output_dir / "evaluation_manifest.json"
        if judge_path.exists() and read_json(judge_path) != judge_manifest:
            raise ValueError("Judge changed; use a new output directory to avoid mixing scores")
        write_json(judge_path, judge_manifest)
        asyncio.run(evaluate_runs(args, settings))
    frame = export_reports(args.output_dir)
    if not frame.empty:
        plot_results(frame, args.output_dir / "plots")
        failures = frame.status.isin(["generation_error", "evaluation_error"]).sum()
        logger.info("Saved %s rows, %s failed", len(frame), failures)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
