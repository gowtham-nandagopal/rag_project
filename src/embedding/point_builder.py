import uuid
from typing import List, Dict, Any
from qdrant_client.models import PointStruct, SparseVector
# CHANGED: Swapped out Gemini for your local open-source 3072-dim embedder
from src.embedding.huggingface_embedder import HuggingFaceEmbedder
from src.embedding.bm25_vectorizer import BM25Vectorizer


def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Convert a string chunk ID (e.g. "abc123_sem_4") to a UUID.
    Qdrant requires point IDs to be UUIDs or unsigned integers.
    Using uuid5 is deterministic — same chunk_id always → same UUID.
    This means re-indexing the same chunk is an idempotent upsert.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def build_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the Qdrant payload from a chunk dict.

    The payload stores everything needed for:
      1. Access control filtering (tier, dept, user_id)
      2. Result display (source, text snippet, ticker, year)
      3. Debugging (chunk_strategy, chunk_index, pii_detected)
      4. Hierarchical retrieval (parent_id for expanding context)

    IMPORTANT: The payload also stores the chunk's raw text so
    we can return it in search results without a second lookup.
    """
    metadata = chunk.get("metadata", {})

    return {
        # ── Access control fields (filtered at query time) ────────────
        "tier":    metadata.get("tier", "public"),   # "public"|"hr"|"confidential"
        "dept":    metadata.get("dept", "all"),      # "engineering"|"finance"|"all"

        # ── Document identity ─────────────────────────────────────────
        "doc_id":      metadata.get("doc_id"),
        "chunk_id":    chunk["chunk_id"],
        "source":      metadata.get("source", ""),
        "source_file": metadata.get("source_file", ""),
        "doc_type":    metadata.get("doc_type", ""),

        # ── Content — stored so results don't need a second lookup ────
        "text": chunk["text"],
        # For sentence-window chunks, store the full context window too
        "window": metadata.get("window", chunk["text"]),

        # ── SEC-specific fields (for public docs) ─────────────────────
        "ticker":  metadata.get("ticker", ""),
        "year":    metadata.get("year", 0),
        "section": metadata.get("section", ""),

        # ── HR-specific fields ────────────────────────────────────────
        "company": metadata.get("company", ""),

        # ── Chunking metadata ─────────────────────────────────────────
        "chunk_strategy": metadata.get("chunk_strategy", ""),
        "chunk_index":    metadata.get("chunk_index", 0),
        "total_chunks":   metadata.get("total_chunks", 1),
        "char_count":     metadata.get("char_count", len(chunk["text"])),

        # ── Hierarchical retrieval link ───────────────────────────────
        # If this is a child chunk, parent_id lets us expand to parent at query time
        "parent_id": chunk.get("parent_id"),
        "chunk_type": metadata.get("chunk_type", "leaf"),

        # ── PII audit trail ───────────────────────────────────────────
        "pii_detected": metadata.get("pii_detected", False),
        "pii_types":    metadata.get("pii_types", []),
    }


def build_point(
    chunk: Dict[str, Any],
    dense_vector: List[float],
    sparse_vector: Dict,
) -> PointStruct:
    """
    Assemble a complete Qdrant PointStruct from:
      - chunk:         the chunk dict from Phase 3
      - dense_vector:  3072-dim Open-Source Hugging Face embedding
      - sparse_vector: BM25 sparse vector {"indices": [...], "values": [...]}
    """
    point_id = chunk_id_to_uuid(chunk["chunk_id"])
    payload  = build_payload(chunk)

    return PointStruct(
        id=point_id,
        vector={
            "dense":  dense_vector,   # 3072-dim open-source float list → semantic search
            "sparse": SparseVector(   # BM25 sparse vector → keyword search
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            )
        },
        payload=payload
    )