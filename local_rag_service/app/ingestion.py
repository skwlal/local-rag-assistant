import csv
import io
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from app.config import Settings
from app.storage import atomic_write, digest
from app.text_cleanup import repair_pdf_encoding

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


def convert_documents_to_markdown(settings: Settings | None = None) -> list[Document]:
    settings = settings or Settings()
    settings.create_directories()
    files = source_files(settings)
    if not files:
        raise ValueError(f"No supported documents in {settings.RAW_DATA_DIR}")
    converter = None
    documents = []
    rows = []
    for path in files:
        name = path.relative_to(settings.RAW_DATA_DIR).as_posix()
        doc_id = digest(name)
        if path.suffix.lower() in {".md", ".txt"}:
            markdown = path.read_text(encoding="utf-8-sig")
        else:
            artifacts = settings.MODEL_DIR / "docling"
            if path.suffix.lower() == ".pdf" and not artifacts.is_dir():
                raise FileNotFoundError(f"Docling artifacts missing: {artifacts}")
            if converter is None:
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
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    }
                )
            result = converter.convert(path, raises_on_error=True)
            markdown = result.document.export_to_markdown()
            if path.suffix.lower() == ".pdf":
                markdown = repair_pdf_encoding(markdown)
        if not markdown.strip():
            raise ValueError(f"Document contains no text: {name}; check OCR settings")
        # A relative source path plus extension avoids lecture.pdf/lecture.docx collisions.
        output = settings.PROCESSED_MD_DIR / f"{name}.md"
        atomic_write(output, markdown)
        metadata = {"file_name": name, "doc_id": doc_id, "source_hash": digest(path.read_bytes())}
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


def split_table(text: str, budget: int, tokenizer: Tokenizer) -> list[str]:
    lines = text.splitlines()
    header = lines[:2]
    chunks: list[str] = []
    current = header.copy()
    if len(tokenizer("\n".join(header))) > budget:
        raise ValueError("Table header exceeds chunk budget; increase chunk_size")
    for row in lines[2:]:
        if len(tokenizer("\n".join(header + [row]))) > budget:
            raise ValueError("A table row exceeds chunk budget; shorten it or raise chunk_size")
        if len(tokenizer("\n".join(current + [row]))) > budget:
            chunks.append("\n".join(current))
            current = header.copy()
        current.append(row)
    chunks.append("\n".join(current))
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
            splitter = SentenceSplitter(
                chunk_size=available,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
                chunking_tokenizer_fn=lambda s: re.split(r"(?<=[.!?])\s+", s),
                include_metadata=False,
            )
            for block, table in blocks:
                parts = (
                    split_table(block, available, tokenizer)
                    if table
                    else splitter.split_text(block)
                )
                for part in parts:
                    content = prefix + part
                    if len(tokenizer(content)) > budget:
                        raise ValueError("Chunk exceeds token budget after heading composition")
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
