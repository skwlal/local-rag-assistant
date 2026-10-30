"""Extract a page-addressable PDF text preview without downloading model weights."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from app.text_cleanup import repair_pdf_encoding  # noqa: E402


def main():
    output = BASE / "data/processed_md/preview"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    audit = []
    for source in sorted((BASE / "data/raw").glob("*.pdf")):
        pages = []
        with pdfium.PdfDocument(source) as document:
            for number in range(len(document)):
                page = document[number]
                textpage = page.get_textpage()
                text = textpage.get_text_range().replace("\r\n", "\n")
                text = text.replace("\ufffe", "-").replace("\u00ad", "").replace("\xa0", " ")
                text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
                text = repair_pdf_encoding(text)
                pages.append({"page": number + 1, "text": text})
                audit.append(
                    {
                        "source": source.name,
                        "page": number + 1,
                        "characters": len(text),
                        "review_low_text": len(text) < 100,
                    }
                )
                textpage.close()
                page.close()
        markdown = (
            "# "
            + source.stem
            + "\n\n"
            + "\n\n".join(f"## Страница {p['page']}\n\n{p['text']}" for p in pages)
            + "\n"
        )
        (output / (source.name + ".md")).write_text(markdown, encoding="utf-8")
        (output / (source.name + ".pages.json")).write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(
            {
                "source": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pages": len(pages),
                "method": "pypdfium2 text layer; no OCR",
                "status": "preview_requires_layout_review",
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "page_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["source", "page", "characters", "review_low_text"]
        )
        writer.writeheader()
        writer.writerows(audit)
    print(f"{len(manifest)} PDFs, {len(audit)} pages -> {output}")


if __name__ == "__main__":
    main()
