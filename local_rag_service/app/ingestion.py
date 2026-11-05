import argparse
import csv
import io
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from app.config import Settings
from app.storage import atomic_write, digest, read_json, write_json
from app.text_cleanup import repair_pdf_encoding

logger = logging.getLogger(__name__)

Tokenizer = Callable[[str], list[Any]]
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def local_tokenizer(settings: Settings) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(settings.model_path("intfloat/multilingual-e5-large")),
        local_files_only=True,
        trust_remote_code=False,
    )


def source_files(settings: Settings) -> list[Path]:
    files = sorted(
        p
        for p in settings.RAW_DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for path in files:
        if not path.resolve().is_relative_to(settings.RAW_DATA_DIR.resolve()):
            raise ValueError(f"Source symlink leaves RAW_DATA_DIR: {path}")
    return files


def corpus_fingerprint(settings: Settings) -> str:
    return digest(
        "\n".join(
            f"{p.relative_to(settings.RAW_DATA_DIR).as_posix()}:{digest(p.read_bytes())}"
            for p in source_files(settings)
        )
    )


def document_converter(settings: Settings):
    artifacts = settings.MODEL_DIR / "docling"
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(artifacts_path=artifacts)
    options.accelerator_options = AcceleratorOptions(device=settings.DEVICE)
    options.enable_remote_services = False
    options.do_ocr = settings.OCR_ENABLED
    options.do_table_structure = True
    options.do_formula_enrichment = True
    if settings.OCR_ENABLED:
        options.ocr_options = TesseractCliOcrOptions(lang=settings.OCR_LANGUAGES)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )


def convert_documents_to_markdown(
    settings: Settings | None = None, *, reuse_existing: bool = False
) -> list[Document]:
    settings = settings or Settings()
    settings.create_directories()
    files = source_files(settings)
    if not files:
        raise ValueError(f"No supported documents in {settings.RAW_DATA_DIR}")
    model_manifest = settings.MODEL_DIR / "manifest.json"
    conversion_config = {
        "schema_version": 1,
        "docling_version": version("docling"),
        "device": settings.DEVICE,
        "ocr": settings.OCR_ENABLED,
        "ocr_languages": settings.OCR_LANGUAGES,
        "formula_enrichment": True,
        "table_structure": True,
        "models_hash": digest(model_manifest.read_bytes()) if model_manifest.is_file() else None,
        "text_cleanup_hash": digest(Path(__file__).with_name("text_cleanup.py").read_bytes()),
    }
    converter = None
    documents = []
    rows = []
    for number, path in enumerate(files, 1):
        started = perf_counter()
        name = path.relative_to(settings.RAW_DATA_DIR).as_posix()
        doc_id = digest(name)
        source_hash = digest(path.read_bytes())
        output = settings.PROCESSED_MD_DIR / f"{name}.md"
        receipt = output.with_suffix(output.suffix + ".cache.json")
        key = {"source_hash": source_hash, "conversion": conversion_config}
        cached = read_json(receipt) if receipt.is_file() else {}
        saved = output.read_text(encoding="utf-8") if output.is_file() else ""
        valid_cache = cached.get("key") == key and cached.get("markdown_hash") == digest(saved)
        adopt = reuse_existing and not receipt.exists() and bool(saved.strip())
        origin = "converted"
        logger.info(
            "[%s/%s] %s: %s",
            number,
            len(files),
            name,
            "cached" if valid_cache else "reuse existing" if adopt else "converting",
        )
        if path.suffix.lower() in {".md", ".txt"}:
            markdown = path.read_text(encoding="utf-8-sig")
        elif valid_cache or adopt:
            markdown = saved
            origin = cached.get("origin", "user_supplied_markdown")
            if adopt:
                logger.warning("Reusing %s without historical source verification", name)
        else:
            artifacts = settings.MODEL_DIR / "docling"
            if path.suffix.lower() == ".pdf" and not artifacts.is_dir():
                raise FileNotFoundError(f"Docling artifacts missing: {artifacts}")
            if converter is None:
                converter = document_converter(settings)
            result = converter.convert(path, raises_on_error=True)
            markdown = result.document.export_to_markdown()
            if path.suffix.lower() == ".pdf":
                markdown = repair_pdf_encoding(markdown)
        if not markdown.strip():
            raise ValueError(f"Document contains no text: {name}; check OCR settings")
        if digest(path.read_bytes()) != source_hash:
            raise ValueError(f"Source changed during conversion: {name}; retry")
        atomic_write(output, markdown)
        write_json(receipt, {"key": key, "markdown_hash": digest(markdown), "origin": origin})
        logger.info(
            "[%s/%s] %s: ready in %.1f s", number, len(files), name, perf_counter() - started
        )
        metadata = {
            "file_name": name,
            "doc_id": doc_id,
            "source_hash": source_hash,
            "conversion_origin": origin,
        }
        documents.append(Document(text=markdown, id_=doc_id, metadata=metadata))
        rows.append(
            {
                "doc_id": doc_id,
                "filename": name,
                "source_format": path.suffix.lower()[1:],
                "char_count": len(markdown),
                "token_estimate": (len(markdown) + 3) // 4,
                "processed_at": datetime.now(UTC).isoformat(),
            }
        )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(settings.METADATA_CSV_PATH, stream.getvalue())
    return documents


def markdown_blocks(text: str) -> list[tuple[str, bool]]:
    """Separate pipe tables while leaving fenced code intact."""
    lines = text.splitlines()
    blocks: list[tuple[str, bool]] = []
    buffer: list[str] = []
    fence = ""
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = "" if fence == marker else (fence or marker)
        if (
            not fence
            and i + 1 < len(lines)
            and "|" in lines[i]
            and TABLE_SEPARATOR.match(lines[i + 1])
        ):
            if buffer:
                blocks.append(("\n".join(buffer), False))
                buffer = []
            table = lines[i : i + 2]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table.append(lines[i])
                i += 1
            blocks.append(("\n".join(table), True))
        else:
            buffer.append(lines[i])
            i += 1
    if buffer:
        blocks.append(("\n".join(buffer), False))
    return [(block.strip(), table) for block, table in blocks if block.strip()]


def split_prefixed_text(text: str, prefix: str, budget: int, tokenizer: Tokenizer) -> list[str]:
    """Repeat context without truncating content or exceeding the combined token budget."""
    available = budget - len(tokenizer(prefix))
    while available > 0:
        parts = SentenceSplitter(
            chunk_size=available,
            chunk_overlap=0,
            tokenizer=tokenizer,
            include_metadata=False,
        ).split_text(text)
        chunks = [prefix + part for part in parts]
        overflow = max((len(tokenizer(chunk)) - budget for chunk in chunks), default=0)
        if overflow <= 0:
            return chunks
        available -= max(1, overflow)
    raise ValueError("Table context leaves no room for text; shorten the heading path")


def table_cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith(r"\|"):
        row = row[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row)]


def split_long_table_row(
    header: list[str], row: str, number: int, budget: int, tokenizer: Tokenizer
) -> list[str]:
    labels, cells = table_cells(header[0]), table_cells(row)
    if len(labels) != len(cells):
        # Ambiguous pipes (for example formulas): preserve the raw row and header.
        return split_prefixed_text(
            "\n".join(header + [row]), f"Строка таблицы {number}:\n", budget, tokenizer
        )
    chunks = []
    for column, (label, cell) in enumerate(zip(labels, cells, strict=True), 1):
        prefix = f"Строка таблицы {number}; столбец {column} ({label}):\n"
        content = cell or "(пустая ячейка)"
        if len(tokenizer(prefix)) > budget // 2:
            prefix = f"Строка таблицы {number}; столбец {column}:\n"
            content = f"Заголовок: {label}\nЗначение: {content}"
        chunks.extend(split_prefixed_text(content, prefix, budget, tokenizer))
    return chunks


def split_table(text: str, budget: int, tokenizer: Tokenizer) -> list[str]:
    lines = text.splitlines()
    header = lines[:2]
    chunks: list[str] = []
    current = header.copy()
    for number, row in enumerate(lines[2:], 1):
        if len(tokenizer("\n".join(header + [row]))) > budget:
            if len(current) > 2:
                chunks.append("\n".join(current))
                current = header.copy()
            chunks.extend(split_long_table_row(header, row, number, budget, tokenizer))
            continue
        if len(tokenizer("\n".join(current + [row]))) > budget:
            chunks.append("\n".join(current))
            current = header.copy()
        current.append(row)
    if len(current) > 2:
        chunks.append("\n".join(current))
    if not chunks:
        return split_prefixed_text(text, "", budget, tokenizer)
    return chunks


def split_documents(
    documents: list[Document],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Tokenizer | None = None,
) -> list[BaseNode]:
    """Use a shared E5 tokenizer; reserve tokens for model prefixes and special tokens."""
    if strategy not in {"naive", "structural"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if not 64 <= chunk_size <= 512 or not 0 <= chunk_overlap < chunk_size - 16:
        raise ValueError("Invalid chunk size or overlap")
    if tokenizer is None:
        encoder = local_tokenizer(Settings())

        def tokenizer(text):
            return encoder.encode(text, add_special_tokens=False)

    budget = chunk_size - 16
    nodes: list[BaseNode] = []
    for document in documents:
        if "file_name" not in document.metadata:
            raise ValueError("Document metadata requires file_name")
        sections = (
            MarkdownNodeParser().get_nodes_from_documents([document])
            if strategy == "structural"
            else [document]
        )
        for section in sections:
            text = section.get_content(metadata_mode=MetadataMode.NONE)
            metadata = dict(document.metadata) | dict(section.metadata)
            header_path = str(metadata.get("header_path", ""))
            heading = re.match(r"^#+\s+([^\n]+)(?:\n|$)", text)
            if strategy == "structural" and heading:
                header_path = f"{header_path.rstrip('/')}/{heading.group(1)}"
                metadata["header_path"] = header_path
                text = text[heading.end() :].strip() or heading.group(1)
            prefix = f"Раздел: {header_path}\n\n" if header_path.strip("/") else ""
            available = budget - len(tokenizer(prefix))
            if available <= chunk_overlap:
                raise ValueError("Heading path leaves insufficient chunk budget")
            blocks = markdown_blocks(text) if strategy == "structural" else [(text, False)]
            for block, table in blocks:
                block_budget = available
                # Token counts are not additive across the heading/body boundary.
                # Re-split the original block until the complete chunks fit.
                while block_budget > chunk_overlap:
                    parts = (
                        split_table(block, block_budget, tokenizer)
                        if table
                        else SentenceSplitter(
                            chunk_size=block_budget,
                            chunk_overlap=chunk_overlap,
                            tokenizer=tokenizer,
                            chunking_tokenizer_fn=lambda s: re.split(r"(?<=[.!?])\s+", s),
                            include_metadata=False,
                        ).split_text(block)
                    )
                    overflow = max(
                        (len(tokenizer(prefix + part)) - budget for part in parts), default=0
                    )
                    if overflow <= 0:
                        break
                    block_budget -= overflow
                else:
                    raise ValueError(
                        f"Cannot fit text with heading {header_path!r} in "
                        f"{metadata['file_name']!r}; shorten the heading or reduce chunk_overlap"
                    )
                for part in parts:
                    content = prefix + part
                    identity = json.dumps(
                        [document.id_, strategy, chunk_size, chunk_overlap, len(nodes), content],
                        ensure_ascii=False,
                    )
                    node_metadata = metadata | {"is_table": table}
                    nodes.append(
                        TextNode(
                            id_=digest(identity),
                            text=content,
                            metadata=node_metadata,
                            excluded_embed_metadata_keys=list(node_metadata),
                            excluded_llm_metadata_keys=list(node_metadata),
                        )
                    )
    if not nodes:
        raise ValueError("No chunks produced")
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Convert documents with per-file checkpoints")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Trust existing Markdown without cache records (explicit recovery)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    documents = convert_documents_to_markdown(reuse_existing=args.reuse_existing)
    print(f"Обработано документов: {len(documents)}", flush=True)


if __name__ == "__main__":
    main()
