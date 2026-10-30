"""Prepare model artifacts online; the service itself remains offline."""

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

MODELS = ("BAAI/bge-m3", "intfloat/multilingual-e5-large", "BAAI/bge-reranker-v2-m3")


def validate_weights(root: Path) -> None:
    for model in MODELS:
        directory = root / re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing model configuration: {directory}")
        candidates = [directory / "model.safetensors", directory / "pytorch_model.bin"]
        indexes = [
            directory / "model.safetensors.index.json",
            directory / "pytorch_model.bin.index.json",
        ]
        if any(p.is_file() and p.stat().st_size > 0 for p in candidates):
            continue
        for index in indexes:
            if index.is_file():
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if shards and all(
                    (directory / shard).is_file() and (directory / shard).stat().st_size > 0
                    for shard in shards
                ):
                    break
        else:
            raise FileNotFoundError(f"Missing or incomplete model weights: {directory}")


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path != root / "manifest.json"
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify local model artifacts")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "models"
    )
    parser.add_argument("--revision", action="append", default=[], metavar="MODEL=REVISION")
    parser.add_argument(
        "--verify", action="store_true", help="Verify transferred artifacts offline"
    )
    args = parser.parse_args(argv)
    manifest_path = args.output_dir / "manifest.json"
    if args.verify:
        validate_weights(args.output_dir)
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        if artifact_hashes(args.output_dir) != expected:
            raise ValueError("Artifact checksums differ from manifest")
        print(f"Verified {len(expected)} local model files")
        return
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from docling.utils.model_downloader import download_models
    from huggingface_hub import HfApi, snapshot_download

    revisions = dict(item.split("=", 1) for item in args.revision)
    if unknown := set(revisions) - set(MODELS):
        raise ValueError(f"Unknown model revisions: {sorted(unknown)}")
    resolved = {}
    for model in MODELS:
        sha = HfApi().model_info(model, revision=revisions.get(model, "main")).sha
        directory = args.output_dir / re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        snapshot_download(
            model,
            revision=sha,
            local_dir=directory,
            allow_patterns=["*.json", "*.safetensors", "pytorch_model*.bin", "*.model", "*.txt"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        resolved[model] = sha
    validate_weights(args.output_dir)
    download_models(
        output_dir=args.output_dir / "docling",
        with_layout=True,
        with_tableformer=True,
        with_code_formula=True,
        with_picture_classifier=False,
        with_rapidocr=False,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "revisions": resolved,
        "files": artifact_hashes(args.output_dir),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Prepared {len(manifest['files'])} model files in {args.output_dir}")


if __name__ == "__main__":
    main()
