from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    HnswConfigDiff,
    OptimizersConfigDiff,
)
from src.config import get_settings
import logging

cfg = get_settings()
logger = logging.getLogger(__name__)

# ── Access-control payload fields we will filter on at query time ──
# Indexing these fields dramatically speeds up filtered search.
INDEXED_PAYLOAD_FIELDS = {
    "tier":         PayloadSchemaType.KEYWORD,  # "public" | "hr" | "confidential"
    "dept":         PayloadSchemaType.KEYWORD,  # "engineering" | "finance" | "all"
    "doc_type":     PayloadSchemaType.KEYWORD,  # "10k_section" | "hr_handbook" | ...
    "ticker":       PayloadSchemaType.KEYWORD,  # "AAPL" | "MSFT" | ...  (SEC filings)
    "year":         PayloadSchemaType.INTEGER,   # 2020 | 2021 | ...  (range filters)
    "pii_detected": PayloadSchemaType.BOOL,     # True | False
    "chunk_strategy": PayloadSchemaType.KEYWORD, # "semantic" | "hierarchical" | ...
}


def get_qdrant_client() -> QdrantClient:
    """Return a configured Qdrant client. Singleton pattern via lru_cache."""
    return QdrantClient(url=cfg.qdrant_url)


def setup_collection(
    client: QdrantClient,
    embed_dim: int,  # <-- Add this parameter line
    collection_name: str = cfg.qdrant_collection,
    recreate: bool = False
) -> bool:
    """
    Create (or verify) the Qdrant collection with:
      - Dense vector space:  "dense"  → 3072-dim, cosine distance
      - Sparse vector space: "sparse" → BM25 tokens, dot product
      - Payload indexes for all access-control and filter fields

    Args:
        recreate: If True, drop and recreate. Use only during dev reset.

    Returns:
        True if created, False if already existed.
    """
    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.warning(f"Dropping existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists — skipping creation")
            return False

    # ── Create the collection ─────────────────────────────────────────
    client.create_collection(
        collection_name=collection_name,

        # Dense vectors: Gemini 3072-dim, cosine similarity
        vectors_config={
            "dense": VectorParams(
                size=embed_dim,          # 3072
                distance=Distance.COSINE,
                # HNSW index config — controls speed/accuracy trade-off
                hnsw_config=HnswConfigDiff(
                    m=16,             # num neighbours per node (16 = good balance)
                    ef_construct=100, # construction accuracy (higher = better index)
                    full_scan_threshold=10000,  # below this size → brute force (faster)
                )
            )
        },

        # Sparse vectors: BM25 token-frequency vectors
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False  # keep in RAM for speed (fits on 16GB)
                )
            )
        },

        # Optimiser: balance indexing speed vs search speed
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=10000,  # start building HNSW index at 10k points
            memmap_threshold=50000,   # use memory-mapped files above 50k points
        )
    )

    logger.info(f"Created collection: {collection_name}")

    # ── Create payload indexes for filtered search ────────────────────
    # This makes tier/dept filtering O(log n) instead of O(n)
    for field_name, field_type in INDEXED_PAYLOAD_FIELDS.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type,
        )
        logger.info(f"  Created payload index: {field_name} ({field_type})")

    print(f"✅ Collection '{collection_name}' created with dense + sparse vectors")
    print(f"   Payload indexes: {list(INDEXED_PAYLOAD_FIELDS.keys())}")
    return True


def verify_collection(client: QdrantClient, collection_name: str = cfg.qdrant_collection):
    """Print collection info — useful for debugging and status checks."""
    info = client.get_collection(collection_name)
    print(f"\n📊 Collection: {collection_name}")
    print(f"   Points count:   {info.points_count}")
    
    # FIX: Safely read vector stats if available, otherwise default to points
    v_count = getattr(info, "vectors_count", "N/A (Use points count)")
    print(f"   Vectors count:  {v_count}")
    
    print(f"   Status:         {info.status}")
    print(f"   Dense dim:      {info.config.params.vectors['dense'].size}")
    return info


if __name__ == "__main__":
    client = get_qdrant_client()
    setup_collection(client)
    verify_collection(client)