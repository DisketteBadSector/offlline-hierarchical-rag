"""Extract structured content from a single document using Docling.

This script is a thin CLI wrapper around the project's DocumentExtractorRegistry.
It selects the correct extractor based on the file extension (.doc, .docx, .pdf),
runs Docling to produce an ExtractedDocumentContent value object, optionally
enriches flagged pages with the Mock LVM adapter, and dumps the full result
to a JSON file for inspection.

Usage:
    python scripts/extract_with_docling.py <path-to-document>

Example:
    python scripts/extract_with_docling.py data/sample-documents/pdf/versioning-and-confidentiality-policy.pdf
"""

from __future__ import annotations
from datetime import datetime

import json
import sys
from pathlib import Path

# Ensure the project's src/ directory is importable when running this script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from offline_rag.infrastructure.ingestion import (  # noqa: E402
    DocumentExtractorRegistry,
    MockLvmLayoutAdapter,
)


def _print_summary(extracted) -> None:  # type: ignore[no-untyped-def]
    """Print a human-readable summary of the extracted document to stdout."""
    print("=" * 70)
    print(f"Document: {extracted.document_id}")
    print(f"Format: {extracted.file_format.value}")
    print(f"Page count: {extracted.page_count}")
    print(f"Blocks: {len(extracted.blocks)}")
    print(f"Tables: {len(extracted.tables)}")
    print(f"Requires LVM assistance: {extracted.requires_lvm_assistance}")
    print(f"Flagged LVM pages: {extracted.flagged_lvm_pages}")
    print(f"Content hash (sha256): {extracted.raw_content_hash[:16]}...")
    print("=" * 70)

def _print_tables(extracted) -> None:  # type: ignore[no-untyped-def]
    """Print a short summary of every extracted table."""
    if not extracted.tables:
        return
    print("\n--- Tables ---")
    for t in extracted.tables:
        print(f"  page={t.page_number} caption={t.caption} rows={len(t.rows)}")


def _run_lvm_enrichment(extracted):  # type: ignore[no-untyped-def]
    """Run the Mock LVM adapter on flagged pages and return the enriched content."""
    print("\n--- LVM enrichment (Mock adapter) ---")
    lvm = MockLvmLayoutAdapter()
    enriched = lvm.enrich_extracted_content(extracted, version_id="v1")
    print(f"Blocks after LVM enrichment: {len(enriched.blocks)}")
    print(f"LVM annotations: {len(enriched.lvm_annotations)}")
    return enriched


def _dump_json(extracted, out_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Serialize the full ExtractedDocumentContent to a JSON file (UTF-8, non-ASCII preserved)."""
    payload = {
        "document_id": extracted.document_id,
        "file_format": extracted.file_format.value,
        "page_count": extracted.page_count,
        "requires_lvm_assistance": extracted.requires_lvm_assistance,
        "flagged_lvm_pages": list(extracted.flagged_lvm_pages),
        "raw_content_hash": extracted.raw_content_hash,
        "blocks": [
            {
                "text": b.text,
                "heading_path": list(b.heading_path),
                "page_number": b.page_number,
                "block_type": b.block_type,
                "section_reference": b.section_reference,
                "heading_level": b.heading_level,
            }
            for b in extracted.blocks
        ],
        "tables": [
            {
                "rows": [list(row) for row in t.rows],
                "page_number": t.page_number,
                "caption": t.caption,
                "heading_path": list(t.heading_path),
            }
            for t in extracted.tables
        ],
        "lvm_annotations": list(extracted.lvm_annotations),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(file_path: str) -> int:
    """Extract one document and print + persist the structured result."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    # 1. Select the correct extractor (.doc / .docx / .pdf) by suffix.
    registry = DocumentExtractorRegistry()
    extractor = registry.get_extractor(path.suffix)

    # 2. Run Docling to produce a domain-level ExtractedDocumentContent.
    extracted = extractor.extract(path, document_id=path.stem)

    # 3. Report what was extracted.
    _print_summary(extracted)
    _print_tables(extracted)

    # 4. If any page was flagged as needing visual/layout interpretation,
    #    run the Mock LVM adapter to demonstrate the enrichment hook.
    if extracted.requires_lvm_assistance:
        _run_lvm_enrichment(extracted)

    # 5. Persist the full structured output with a filename that encodes
    #    the document name and a timestamp, e.g.
    #    docling-output-versioning-and-confidentiality-policy-20261101-205525.json
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Sanitize the document name so it is safe to use inside a filename
    # on all common filesystems (replace path separators and spaces).
    safe_name = "".join(
        ch if (ch.isalnum() or ch in "-_.") else "-"
        for ch in extracted.document_id
    ).strip("-") or "document"
    out_path = REPO_ROOT / f"temp/docling-outputs/doc-{safe_name}-{timestamp}.json"
    _dump_json(extracted, out_path)
    print(f"\nFull JSON output written to: {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/extract_with_docling.py <path-to-document>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(sys.argv[1]))