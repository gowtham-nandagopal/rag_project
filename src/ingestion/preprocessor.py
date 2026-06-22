import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from src.ingestion.pii_detector import PIIDetector
from src.config import get_settings

cfg = get_settings()


@dataclass
class ProcessedDocument:
    """
    Standardized document representation after preprocessing.
    This is what gets passed to the chunking stage.
    """
    doc_id:          str          # SHA256 hash of content (stable ID)
    content:         str          # anonymized text (safe to embed)
    original_content: str         # original (kept for authorized access)
    metadata:        Dict          # Qdrant payload — access control fields
    pii_found:       List[str]   # list of PII entity types found
    pii_count:       int          # total PII items removed
    source_file:     str          # original filename for audit trail


class DocumentPreprocessor:
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.processed: List[ProcessedDocument] = []

    def _make_doc_id(self, content: str) -> str:
        """Stable, content-based ID. Same content = same ID (idempotent)."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _process_text(self, text: str, metadata: Dict, source: str) -> ProcessedDocument:
        """Core processing: anonymize PII, attach metadata."""
        pii_result = self.pii_detector.anonymize(text)

        # Enrich metadata with PII detection flags
        enriched_metadata = {
            **metadata,
            "pii_detected": pii_result["pii_detected"],
            "pii_types": pii_result["pii_found"],
        }

        return ProcessedDocument(
            doc_id=self._make_doc_id(text),
            content=pii_result["anonymized_text"],
            original_content=text,
            metadata=enriched_metadata,
            pii_found=pii_result["pii_found"],
            pii_count=pii_result["pii_count"],
            source_file=source
        )

    # ── Loaders for each document type ───────────────────────────────

    def load_sec_filing(self, json_path: Path) -> List[ProcessedDocument]:
        """
        SEC filings have multiple sections (section_1, section_7 etc.).
        We process each section as a separate document so chunking can
        work at the section level — better context boundaries.
        """
        data = json.loads(json_path.read_text(encoding="utf-8"))
        docs = []

        for section_name, section_text in data.get("sections", {}).items():
            if not section_text or len(section_text.strip()) < 100:
                continue

            metadata = {
                **data["metadata"],         # tier, source, ticker, year
                "section": section_name,     # e.g. "section_7" (MD&A)
                "company": data["ticker"],
                "doc_type": "10k_section"
            }

            doc = self._process_text(
                text=section_text,
                metadata=metadata,
                source=json_path.name
            )
            docs.append(doc)

        return docs

    def load_hr_handbook(self, md_path: Path, base_metadata: Dict) -> ProcessedDocument:
        """Load a single markdown file from an HR handbook repo."""
        text = md_path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            **base_metadata,
            "filename": md_path.name,
            "doc_type": "hr_handbook"
        }
        return self._process_text(text, metadata, source=str(md_path))

    def load_performance_review(self, json_path: Path) -> ProcessedDocument:
        """Load a synthetic performance review. These are PII-heavy."""
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return self._process_text(
            text=data["content"],
            metadata=data["metadata"],
            source=json_path.name
        )

    # ── Main pipeline ─────────────────────────────────────────────────

    def run(self) -> List[ProcessedDocument]:
        """Process all raw documents across all tiers."""
        all_docs = []

        # ── Public: SEC filings ───────────────────────────────────────
        sec_dir = cfg.data_raw / "public"
        print(f"\n📂 Processing SEC filings from {sec_dir}...")
        for path in sec_dir.glob("*.json"):
            docs = self.load_sec_filing(path)
            all_docs.extend(docs)
            print(f"  ✓ {path.name} → {len(docs)} sections")

        # ── HR: Handbooks (all .md files across all cloned repos) ─────
        hr_dir = cfg.data_raw / "hr"
        print(f"\n📂 Processing HR handbooks from {hr_dir}...")
        for repo_dir in hr_dir.iterdir():
            if not repo_dir.is_dir(): continue

            meta_file = repo_dir / "_metadata.json"
            base_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {
                "tier": "hr", "dept": "all", "source": "hr_handbook"
            }
            base_meta = base_meta.get("metadata", base_meta)

            for md_path in repo_dir.rglob("*.md"):
                if md_path.stat().st_size < 200: continue
                doc = self.load_hr_handbook(md_path, base_meta)
                all_docs.append(doc)
            print(f"  ✓ {repo_dir.name}")

        # ── Confidential: Performance reviews ─────────────────────────
        conf_dir = cfg.data_raw / "confidential"
        print(f"\n📂 Processing performance reviews from {conf_dir}...")
        for path in conf_dir.glob("*.json"):
            doc = self.load_performance_review(path)
            all_docs.append(doc)
            pii_info = f"{doc.pii_count} PII items" if doc.pii_count else "no PII"
            print(f"  ✓ {path.name} ({pii_info})")

        # ── Save processed docs to disk ───────────────────────────────
        out_dir = cfg.data_processed
        out_dir.mkdir(parents=True, exist_ok=True)
        for doc in all_docs:
            fname = f"{doc.doc_id}.json"
            data = {
                "doc_id":    doc.doc_id,
                "content":   doc.content,
                "metadata":  doc.metadata,
                "pii_found": doc.pii_found,
                "pii_count": doc.pii_count,
                "source_file": doc.source_file,
            }
            (out_dir / fname).write_text(json.dumps(data, ensure_ascii=False))

        print(f"\n✅ Preprocessed {len(all_docs)} documents → {out_dir}")
        self.processed = all_docs
        return all_docs


if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    docs = preprocessor.run()

    # Print summary by tier
    from collections import Counter
    tier_counts = Counter(d.metadata.get("tier") for d in docs)
    pii_total = sum(d.pii_count for d in docs)
    print(f"\n📊 Summary:")
    for tier, count in tier_counts.items():
        print(f"   {tier}: {count} documents")
    print(f"   PII items removed (total): {pii_total}")