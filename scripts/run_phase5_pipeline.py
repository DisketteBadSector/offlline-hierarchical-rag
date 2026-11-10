"""Phase 5 end-to-end pipeline runner over real sample documents.

This script exercises the full Phase 5 feature set against documents under
``data/sample-documents/`` (DOC, DOCX, PDF):

1. Bootstrap fixtures (layers, nodes, models, prompts) into an in-memory UoW.
2. Ingest every sample document via ``DocumentIngestionService``:
   - extractor (Docling) -> chunking (``KnowledgeChunker``) -> LVM hook (if flagged)
   - exact/near-duplicate + overlap detection (``KnowledgeDeduplicationService``)
   - persist ``KnowledgeItem`` v1 + chunks + optional vector index
3. Re-ingest a subset of documents to prove exact-duplicate rejection.
4. Simulate a Phase-5 update cycle: create v2 with mutated content and
   verify the new duplicate report excludes the item itself.
5. Emit a JSON summary and two CSV reports (versioning ledger + duplicate report).
6. Optionally (``--dump-chunks``) dump full chunk texts (v1 and v2) into
   per-document JSON files and a combined JSONL index.

Usage:
    python scripts/run_phase5_pipeline.py
    python scripts/run_phase5_pipeline.py --samples data/sample-documents --out temp/phase5
    python scripts/run_phase5_pipeline.py --dedup-policy reject
    python scripts/run_phase5_pipeline.py --near-threshold 0.85 --overlap-threshold 0.70
    python scripts/run_phase5_pipeline.py --dump-chunks
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from offline_rag.application.services import (
    DeduplicationPolicy,
    DocumentIngestionResult,
    DocumentIngestionService,
    KnowledgeApplicationService,
    KnowledgeDeduplicationService,
)
from offline_rag.domain.knowledge import (
    ChunkingPolicy,
    DuplicateDetectionResult,
    KnowledgeChunker,
    KnowledgeItem,
    MetadataRecord,
)
from offline_rag.domain.shared.errors import (
    DomainError,
    InvariantViolationError,
)
from offline_rag.domain.shared.value_objects import NodeIdentifier, VersionTag
from offline_rag.infrastructure import (
    DeterministicMockEmbeddingAdapter,
    FixtureConfigLoader,
    InMemoryDocumentStorage,
    InMemoryVectorIndex,
    MockLvmLayoutAdapter,
)

DEFAULT_SAMPLES = REPO_ROOT / "data" / "sample-documents"
DEFAULT_FIXTURES = REPO_ROOT / "data" / "fixtures"
DEFAULT_OUT = REPO_ROOT / "temp" / "phase5-report"

# (relative path, owner node, confidentiality)
SAMPLE_DOCUMENTS: tuple[tuple[str, str, str], ...] = (
    ("docx/managerial-report-parent-org-a.docx", "N-100", "internal"),
    ("docx/strategic-quarterly-report.docx", "N-100", "internal"),
    ("docx/گزارش عملکرد سالانه — شرکت فناوری داده_پردازان پارس.docx", "N-100", "internal"),
    ("doc/detailed-issue-report-project-team-p.doc", "N-300", "restricted"),
    ("doc/technical-summary-research-center-b.doc", "N-200", "internal"),
    ("pdf/research-notes-small-model-team-t.pdf", "N-301", "restricted"),
    ("pdf/versioning-and-confidentiality-policy.pdf", "N-200", "internal"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_doc_id(relative_path: str) -> str:
    """Derive a filesystem-safe, deterministic document_id from a relative path."""
    stem = Path(relative_path).stem
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "-" for ch in stem
    ).strip("-")
    return f"PHASE5-{cleaned or 'doc'}"


def _metadata_for(node_id: NodeIdentifier, doc_id: str, confidentiality: str) -> MetadataRecord:
    v1 = VersionTag.initial()
    return MetadataRecord(
        domain="مدیریت دانش سازمانی",
        node_id=node_id,
        document_type=f"phase5-{doc_id.split('.')[-1].lower() if '.' in doc_id else 'sample'}",
        confidentiality=confidentiality,
        initial_version_tag=v1,
        active_version_tag=v1,
        author_or_owner="Phase5 Runner",
        tags=("phase5", "sample"),
    )


def _serialize_duplicate(result: DuplicateDetectionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "query_content_hash": result.query_content_hash,
        "is_exact_duplicate": result.is_exact_duplicate,
        "exact_matches": list(result.exact_matches),
        "near_duplicate_matches": [
            {
                "existing_item_id": m.existing_item_id,
                "existing_version_id": m.existing_version_id,
                "similarity": round(m.similarity.value, 4),
                "matched_chunk_ids": list(m.matched_chunk_ids),
            }
            for m in result.near_duplicate_matches
        ],
        "overlap_matches": [
            {
                "existing_item_id": m.existing_item_id,
                "existing_version_id": m.existing_version_id,
                "similarity": round(m.similarity.value, 4),
                "matched_chunk_ids": list(m.matched_chunk_ids),
            }
            for m in result.overlap_matches
        ],
        "has_any_duplicate": result.has_any_duplicate,
    }


def _serialize_ingestion(result: DocumentIngestionResult) -> dict[str, Any]:
    return {
        "document_id": result.knowledge_item.item_id,
        "title": result.knowledge_item.title,
        "owner_node": str(result.knowledge_item.scope.owner_node_id),
        "file_format": result.knowledge_item.document_format.value,
        "active_version": result.active_version.version_tag.tag,
        "page_count": result.extracted_content.page_count,
        "block_count": len(result.extracted_content.blocks),
        "table_count": len(result.extracted_content.tables),
        "chunk_count": len(result.chunks),
        "lvm_assisted": result.lvm_assisted,
        "flagged_lvm_pages": list(result.extracted_content.flagged_lvm_pages),
        "storage_uri": result.storage_uri,
        "content_hash": result.extracted_content.raw_content_hash[:16] + "...",
        "duplicate_detection": _serialize_duplicate(result.duplicate_detection),
    }


def _serialize_chunk(chk: Any) -> dict[str, Any]:
    """Serialize a single KnowledgeChunk with full source-referencing detail."""
    ref = chk.source_reference
    return {
        "chunk_id": chk.chunk_id,
        "source_item_id": chk.source_item_id,
        "version_id": chk.version_tag.tag,
        "sequence_index": chk.sequence_index,
        "text": chk.text,
        "text_length": len(chk.text),
        "source_reference": {
            "document_id": ref.document_id,
            "version_id": ref.version_id,
            "file_format": ref.file_format.value,
            "page_number": ref.page_number,
            "section_reference": ref.section_reference,
            "heading_path": list(ref.heading_path),
            "citation": ref.to_citation_string(sequence_index=chk.sequence_index),
        },
        "context_boundary": {
            "start": chk.context_boundary_start,
            "end": chk.context_boundary_end,
        },
        "inherited_metadata": {
            "domain": chk.inherited_metadata.domain,
            "node_id": str(chk.inherited_metadata.node_id),
            "document_type": chk.inherited_metadata.document_type,
            "confidentiality": chk.inherited_metadata.confidentiality,
            "active_version": (
                chk.inherited_metadata.active_version_tag.tag
                if chk.inherited_metadata.active_version_tag is not None
                else None
            ),
        },
    }


def _mutate_text(text: str) -> str:
    """Append an update marker so the new content hash differs from v1."""
    return f"{text}\n\n[به‌روزرسانی فاز ۵ — بازبینی محتوا و افزودن یادداشت نسخه دوم.]"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def bootstrap(fixtures_dir: Path) -> tuple[Any, InMemoryDocumentStorage, InMemoryVectorIndex]:
    uow = FixtureConfigLoader.bootstrap_in_memory_uow(fixtures_dir)
    storage = InMemoryDocumentStorage()
    vector_index = InMemoryVectorIndex()
    return uow, storage, vector_index


def build_services(
    uow: Any,
    storage: InMemoryDocumentStorage,
    vector_index: InMemoryVectorIndex,
    dedup_policy: DeduplicationPolicy,
) -> tuple[DocumentIngestionService, KnowledgeDeduplicationService, KnowledgeApplicationService]:
    embedding = DeterministicMockEmbeddingAdapter(dimension=64)
    dedup_service = KnowledgeDeduplicationService(uow, embedding, policy=dedup_policy)
    ingestion_service = DocumentIngestionService(
        uow=uow,
        chunker=KnowledgeChunker(ChunkingPolicy(max_chunk_chars=600, chunk_overlap_chars=60)),
        lvm_assistant=MockLvmLayoutAdapter(),
        document_storage=storage,
        vector_index=vector_index,
        embedding_port=embedding,
        deduplication_service=dedup_service,
    )
    knowledge_service = KnowledgeApplicationService(
        uow=uow,
        vector_index=vector_index,
        embedding_port=embedding,
        deduplication_service=dedup_service,
    )
    return ingestion_service, dedup_service, knowledge_service


def ingest_all_samples(
    ingestion_service: DocumentIngestionService,
    samples_dir: Path,
    dedup_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ingest all sample documents; return (successes, failures)."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for relative_path, node_str, confidentiality in SAMPLE_DOCUMENTS:
        path = samples_dir / relative_path
        doc_id = _safe_doc_id(relative_path)
        record: dict[str, Any] = {
            "relative_path": relative_path,
            "document_id": doc_id,
            "owner_node": node_str,
        }
        if not path.exists():
            failures.append({**record, "error": f"file not found: {path}"})
            continue
        try:
            result = ingestion_service.ingest_file(
                file_path=path,
                owner_node_id=NodeIdentifier(node_str),
                metadata=_metadata_for(NodeIdentifier(node_str), doc_id, confidentiality),
                document_id=doc_id,
                deduplication_policy=dedup_policy,
            )
            successes.append(_serialize_ingestion(result))
        except DomainError as exc:
            failures.append({**record, "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - surfaced in report
            failures.append(
                {**record, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            )
    return successes, failures


def prove_exact_duplicate_rejection(
    ingestion_service: DocumentIngestionService,
    samples_dir: Path,
) -> list[dict[str, Any]]:
    """Re-ingest the first three sample documents with dedup_policy=reject."""
    evidence: list[dict[str, Any]] = []
    for relative_path, node_str, confidentiality in SAMPLE_DOCUMENTS[:3]:
        path = samples_dir / relative_path
        if not path.exists():
            continue
        doc_id = _safe_doc_id(relative_path)
        try:
            ingestion_service.ingest_file(
                file_path=path,
                owner_node_id=NodeIdentifier(node_str),
                metadata=_metadata_for(NodeIdentifier(node_str), doc_id, confidentiality),
                document_id=doc_id,
                deduplication_policy="reject",
            )
            evidence.append(
                {
                    "document_id": doc_id,
                    "expected": "InvariantViolationError",
                    "actual": "no error",
                    "passed": False,
                }
            )
        except InvariantViolationError as exc:
            evidence.append(
                {
                    "document_id": doc_id,
                    "expected": "InvariantViolationError",
                    "actual": f"InvariantViolationError: {str(exc)[:120]}...",
                    "passed": True,
                }
            )
        except DomainError as exc:
            evidence.append(
                {
                    "document_id": doc_id,
                    "expected": "InvariantViolationError",
                    "actual": f"{type(exc).__name__}: {exc}",
                    "passed": False,
                }
            )
    return evidence


def run_update_cycle(
    uow: Any,
    knowledge_service: KnowledgeApplicationService,
    dedup_service: KnowledgeDeduplicationService,
    targets: list[str],
) -> list[dict[str, Any]]:
    """For each target item, create v2 with mutated content and check duplicate report."""
    ledger: list[dict[str, Any]] = []
    for item_id in targets:
        item: KnowledgeItem | None = uow.knowledge.get_by_id(item_id)
        if item is None:
            ledger.append({"item_id": item_id, "error": "not found"})
            continue

        # 1. Duplicate check before update (must exclude self)
        pre_update_dedup = knowledge_service.check_duplicate_before_update(
            item_id=item_id,
            content_hash=f"MUTATED-{item_id}",
            candidate_chunks=item.active_chunks,
            deduplication_service=dedup_service,
        )

        # 2. Build v2 chunks with mutated text and a matching SourceReference for v2
        v2 = item.active_version_tag.next_version()
        mutated_metadata = item.metadata.with_active_version(v2)
        mutated_chunks = tuple(
            type(chk)(
                chunk_id=f"{item_id}:{v2.tag}:chk-{idx:04d}",
                source_item_id=item_id,
                version_tag=v2,
                sequence_index=idx,
                text=_mutate_text(chk.text),
                inherited_metadata=mutated_metadata,
                source_reference=replace(chk.source_reference, version_id=v2.tag),
                context_boundary_start=chk.context_boundary_start,
                context_boundary_end=chk.context_boundary_end,
            )
            for idx, chk in enumerate(item.active_chunks)
        )

        # 3. Apply update via application service
        updated = knowledge_service.update_knowledge_item(
            item_id=item_id,
            content_hash=f"MUTATED-{item_id}",
            change_reason="Phase 5 update cycle",
            chunks=mutated_chunks,
            diff_summary="افزودن یادداشت نسخه دوم و بازبینی محتوا",
        )

        # 4. Inspect v1 (old version) safely — fetch once, then narrow
        v1_snapshot = updated.get_version("v1")
        v1_present = v1_snapshot is not None
        v1_is_active: bool | None = None
        if v1_snapshot is not None:
            v1_is_active = v1_snapshot.is_active

        prev_tag_obj = updated.active_version.previous_version_tag
        previous_version: str | None = prev_tag_obj.tag if prev_tag_obj is not None else None

        ledger.append(
            {
                "item_id": item_id,
                "versions_total": len(updated.versions),
                "active_version": updated.active_version.version_tag.tag,
                "previous_version": previous_version,
                "initial_version": (
                    updated.metadata.initial_version_tag.tag
                    if updated.metadata.initial_version_tag is not None
                    else None
                ),
                "old_version_still_present": v1_present,
                "old_version_is_active": v1_is_active,
                "pre_update_duplicate_check": _serialize_duplicate(pre_update_dedup),
            }
        )
    return ledger


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_versioning_ledger_csv(
    path: Path,
    successes: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
    updated = {row["item_id"]: row for row in ledger if "item_id" in row}
    columns = (
        "document_id",
        "owner_node",
        "file_format",
        "page_count",
        "chunk_count",
        "active_version",
        "versions_total",
        "previous_version",
        "old_version_still_present",
        "lvm_assisted",
        "content_hash",
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for row in successes:
            update = updated.get(row["document_id"], {})
            writer.writerow(
                [
                    row["document_id"],
                    row["owner_node"],
                    row["file_format"],
                    row["page_count"],
                    row["chunk_count"],
                    update.get("active_version", row["active_version"]),
                    update.get("versions_total", 1),
                    update.get("previous_version", ""),
                    update.get("old_version_still_present", ""),
                    row["lvm_assisted"],
                    row["content_hash"],
                ]
            )


def write_duplicate_report_csv(
    path: Path,
    successes: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
    columns = (
        "document_id",
        "stage",
        "is_exact_duplicate",
        "exact_matches",
        "near_duplicates",
        "overlaps",
        "has_any_duplicate",
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for row in successes:
            det = row.get("duplicate_detection")
            if det is None:
                continue
            writer.writerow(
                [
                    row["document_id"],
                    "ingest",
                    det["is_exact_duplicate"],
                    ";".join(det["exact_matches"]),
                    ";".join(m["existing_item_id"] for m in det["near_duplicate_matches"]),
                    ";".join(m["existing_item_id"] for m in det["overlap_matches"]),
                    det["has_any_duplicate"],
                ]
            )
        for row in ledger:
            det = row.get("pre_update_duplicate_check")
            if det is None:
                continue
            writer.writerow(
                [
                    row["item_id"],
                    "pre_update",
                    det["is_exact_duplicate"],
                    ";".join(det["exact_matches"]),
                    ";".join(m["existing_item_id"] for m in det["near_duplicate_matches"]),
                    ";".join(m["existing_item_id"] for m in det["overlap_matches"]),
                    det["has_any_duplicate"],
                ]
            )


# ---------------------------------------------------------------------------
# Chunk dumping (--dump-chunks)
# ---------------------------------------------------------------------------

def dump_chunks(
    out_dir: Path,
    successes: list[dict[str, Any]],
    uow: Any,
) -> dict[str, Any]:
    """Dump every chunk of every ingested item (all versions, including v2).

    Produces per-document JSON files and one combined JSONL index.
    """
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "phase5-chunks.jsonl"

    manifest: dict[str, Any] = {
        "documents": [],
        "total_chunks": 0,
        "jsonl_index": str(jsonl_path),
        "per_document_dir": str(chunks_dir),
    }

    with jsonl_path.open("w", encoding="utf-8") as jsonl_fh:
        for row in successes:
            item_id = row["document_id"]
            item: KnowledgeItem | None = uow.knowledge.get_by_id(item_id)
            if item is None:
                continue

            versions_dump: list[dict[str, Any]] = []
            for version in item.versions:
                version_chunks = [_serialize_chunk(chk) for chk in version.chunks]
                prev_tag = version.previous_version_tag
                versions_dump.append(
                    {
                        "version_tag": version.version_tag.tag,
                        "sequence_number": version.version_tag.sequence_number,
                        "is_active": version.is_active,
                        "content_hash": version.content_hash,
                        "change_reason": version.change_reason,
                        "previous_version": prev_tag.tag if prev_tag is not None else None,
                        "chunk_count": len(version_chunks),
                        "chunks": version_chunks,
                    }
                )

                for chk_dict in version_chunks:
                    jsonl_fh.write(
                        json.dumps(
                            {
                                "document_id": item_id,
                                "version": version.version_tag.tag,
                                "is_active": version.is_active,
                                **chk_dict,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            doc_path = chunks_dir / f"{item_id}.chunks.json"
            doc_payload = {
                "document_id": item_id,
                "title": item.title,
                "owner_node": str(item.scope.owner_node_id),
                "document_format": item.document_format.value,
                "source_path": item.source_path,
                "requires_lvm_interpretation": item.requires_lvm_interpretation,
                "versions": versions_dump,
            }
            doc_path.write_text(
                json.dumps(doc_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            manifest["documents"].append(
                {
                    "document_id": item_id,
                    "versions": [
                        {
                            "version_tag": v["version_tag"],
                            "is_active": v["is_active"],
                            "chunk_count": v["chunk_count"],
                        }
                        for v in versions_dump
                    ],
                    "file": str(doc_path),
                }
            )
            manifest["total_chunks"] += sum(v["chunk_count"] for v in versions_dump)

    return manifest


# ---------------------------------------------------------------------------
# CLI / output
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_ingest_summary(
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    _print_header(f"[1] INGESTION — {len(successes)} ok / {len(failures)} failed")
    for row in successes:
        det = row.get("duplicate_detection") or {}
        dup_tag = ""
        if det.get("is_exact_duplicate"):
            dup_tag = f" | EXACT dup -> {det['exact_matches']}"
        elif det.get("near_duplicate_matches"):
            dup_tag = f" | near -> {[m['existing_item_id'] for m in det['near_duplicate_matches']]}"
        elif det.get("overlap_matches"):
            dup_tag = f" | overlap -> {[m['existing_item_id'] for m in det['overlap_matches']]}"
        print(
            f"  ✓ {row['document_id']:<40} node={row['owner_node']:<5} "
            f"fmt={row['file_format']:<4} pages={row['page_count']:<2} "
            f"chunks={row['chunk_count']:<3} lvm={row['lvm_assisted']}{dup_tag}"
        )
    for row in failures:
        print(f"  ✗ {row['document_id']:<40} {row.get('error', 'unknown error')}")


def _print_dedup_evidence(evidence: list[dict[str, Any]]) -> None:
    _print_header("[2] EXACT-DUPLICATE REJECTION EVIDENCE")
    for row in evidence:
        mark = "✓" if row["passed"] else "✗"
        print(f"  {mark} {row['document_id']:<40} {row['actual'][:90]}")


def _print_update_cycle(ledger: list[dict[str, Any]]) -> None:
    _print_header("[3] VERSIONING UPDATE CYCLE (v1 -> v2)")
    for row in ledger:
        if "error" in row:
            print(f"  ✗ {row['item_id']}: {row['error']}")
            continue
        print(
            f"  ✓ {row['item_id']:<40} versions={row['versions_total']} "
            f"active={row['active_version']} prev={row['previous_version']} "
            f"v1_preserved={row['old_version_still_present']} "
            f"v1_active={row['old_version_is_active']}"
        )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(description="Phase 5 pipeline runner over sample documents.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dedup-policy",
        choices=("allow", "warn", "reject"),
        default="warn",
        help="Deduplication behavior during ingestion (default: warn).",
    )
    parser.add_argument("--near-threshold", type=float, default=0.92)
    parser.add_argument("--overlap-threshold", type=float, default=0.80)
    parser.add_argument(
        "--skip-update-cycle",
        action="store_true",
        help="Skip the v1 -> v2 update cycle step.",
    )
    parser.add_argument(
        "--dump-chunks",
        action="store_true",
        help=(
            "Dump full chunk texts (per-document JSON + combined JSONL) "
            "for all versions (v1 and v2) into <out>/chunks/."
        ),
    )
    args = parser.parse_args(argv)

    if not args.samples.exists():
        print(f"ERROR: samples directory not found: {args.samples}", file=sys.stderr)
        return 2
    if not args.fixtures.exists():
        print(f"ERROR: fixtures directory not found: {args.fixtures}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Bootstrap
    uow, storage, vector_index = bootstrap(args.fixtures)
    dedup_policy = DeduplicationPolicy(
        near_duplicate_threshold=args.near_threshold,
        overlap_threshold=args.overlap_threshold,
    )
    ingestion_service, dedup_service, knowledge_service = build_services(
        uow, storage, vector_index, dedup_policy
    )

    # Step 1: ingest all samples
    successes, failures = ingest_all_samples(
        ingestion_service, args.samples, dedup_policy=args.dedup_policy
    )
    _print_ingest_summary(successes, failures)

    # Step 2: prove exact duplicate rejection
    evidence = prove_exact_duplicate_rejection(ingestion_service, args.samples)
    _print_dedup_evidence(evidence)

    # Step 3: versioning update cycle
    ledger: list[dict[str, Any]] = []
    if not args.skip_update_cycle and successes:
        targets = [row["document_id"] for row in successes[:3]]
        ledger = run_update_cycle(uow, knowledge_service, dedup_service, targets)
        _print_update_cycle(ledger)

    # Step 4: emit reports
    json_path = args.out / "phase5-report.json"
    ledger_csv = args.out / "phase5-versioning-ledger.csv"
    dup_csv = args.out / "phase5-duplicate-report.csv"

    summary: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples_dir": str(args.samples),
        "fixtures_dir": str(args.fixtures),
        "deduplication_policy": {
            "ingestion_policy": args.dedup_policy,
            "near_duplicate_threshold": args.near_threshold,
            "overlap_threshold": args.overlap_threshold,
        },
        "ingestion": {"successes": successes, "failures": failures},
        "exact_duplicate_rejection_evidence": evidence,
        "update_cycle_ledger": ledger,
        "counts": {
            "documents_ingested": len(successes),
            "documents_failed": len(failures),
            "documents_with_any_duplicate": sum(
                1
                for row in successes
                if (row.get("duplicate_detection") or {}).get("has_any_duplicate")
            ),
            "items_updated_to_v2": len(ledger),
        },
    }

    # Step 5 (optional): dump every chunk of every item (all versions)
    chunks_manifest: dict[str, Any] | None = None
    if args.dump_chunks:
        chunks_manifest = dump_chunks(args.out, successes, uow)
        summary["chunks_dump"] = chunks_manifest

    # Persist all text artifacts
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_versioning_ledger_csv(ledger_csv, successes, ledger)
    write_duplicate_report_csv(dup_csv, successes, ledger)

    _print_header("[4] ARTIFACTS")
    print(f"  JSON summary         : {json_path}")
    print(f"  Versioning ledger    : {ledger_csv}")
    print(f"  Duplicate report     : {dup_csv}")
    if chunks_manifest is not None:
        print(f"  Chunks (JSONL)       : {chunks_manifest['jsonl_index']}")
        print(f"  Chunks (per-doc dir) : {chunks_manifest['per_document_dir']}")
        print(
            f"    └─ {len(chunks_manifest['documents'])} documents, "
            f"{chunks_manifest['total_chunks']} total chunks (all versions)"
        )
    print("=" * 78)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())