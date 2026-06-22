from typing import List, Dict
from src.ingestion.preprocessor import ProcessedDocument
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.chunking.sentence_window_chunker import SentenceWindowChunker


class ChunkingRouter:
    """
    Routes each document to the correct chunking strategy:

      tier=public  + doc_type=10k_section  → SemanticChunker
        (long narrative text, topic boundaries matter)

      tier=hr      + doc_type=hr_handbook  → HierarchicalChunker
        (structured policies, need context around retrieved sections)

      tier=confidential + doc_type=performance_review → SentenceWindowChunker
        (dense short sentences, need surrounding context)
    """

    def __init__(self):
        # Lazy initialisation — only create chunkers that are actually used
        self._semantic = None
        self._hierarchical = None
        self._sentence_window = None

    @property
    def semantic(self) -> SemanticChunker:
        if self._semantic is None:
            self._semantic = SemanticChunker(breakpoint_percentile_threshold=95)
        return self._semantic

    @property
    def hierarchical(self) -> HierarchicalChunker:
        if self._hierarchical is None:
            self._hierarchical = HierarchicalChunker()
        return self._hierarchical

    @property
    def sentence_window(self) -> SentenceWindowChunker:
        if self._sentence_window is None:
            self._sentence_window = SentenceWindowChunker(window_size=3)
        return self._sentence_window

    def _select_strategy(self, doc: ProcessedDocument) -> str:
        tier = doc.metadata.get("tier", "")
        doc_type = doc.metadata.get("doc_type", "")

        if tier == "public" and "10k" in doc_type:
            return "semantic"
        elif tier == "hr":
            return "hierarchical"
        elif tier == "confidential":
            return "sentence_window"
        else:
            return "semantic"  # safe default

    def chunk(self, doc: ProcessedDocument) -> List[dict]:
        """
        Chunk a document using the appropriate strategy.
        Always returns a flat list of chunk dicts for Qdrant ingestion.
        """
        strategy = self._select_strategy(doc)

        if strategy == "semantic":
            return self.semantic.chunk(doc)

        elif strategy == "hierarchical":
            parents, children = self.hierarchical.chunk(doc)
            # Index only children in Qdrant; parents live in docstore
            # Tag children so retrieval knows to expand to parent
            return children

        elif strategy == "sentence_window":
            return self.sentence_window.chunk(doc)

        return []

    def chunk_all(self, docs: List[ProcessedDocument]) -> List[dict]:
        """Process all documents and return all chunks ready for indexing."""
        all_chunks = []
        for i, doc in enumerate(docs):
            strategy = self._select_strategy(doc)
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)
            print(f"[{i+1}/{len(docs)}] {doc.source_file} → {len(chunks)} chunks ({strategy})")
        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        return all_chunks


# ── Run the full chunking pipeline ────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.config import get_settings
    from src.ingestion.preprocessor import ProcessedDocument

    cfg = get_settings()

    # Load processed documents from disk
    docs = []
    for path in cfg.data_processed.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs.append(ProcessedDocument(
            doc_id=data["doc_id"],
            content=data["content"],
            original_content="",   # not needed for chunking
            metadata=data["metadata"],
            pii_found=data["pii_found"],
            pii_count=data["pii_count"],
            source_file=data["source_file"]
        ))

    print(f"Loaded {len(docs)} processed documents")

    router = ChunkingRouter()
    chunks = router.chunk_all(docs)

    # Save chunks for inspection / debugging
    out = Path("data/chunks.json")
    out.write_text(json.dumps(chunks[:10], indent=2))  # preview first 10
    print(f"Preview saved to {out}")