import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Set
from qdrant_client import QdrantClient
from src.config import get_settings
from src.embedding.huggingface_embedder import HuggingFaceEmbedder
from src.embedding.bm25_vectorizer import BM25Vectorizer
from src.embedding.point_builder import build_point, chunk_id_to_uuid
from src.embedding.qdrant_setup import get_qdrant_client, setup_collection

cfg = get_settings()
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("data/indexing_checkpoint.json")


class IndexingPipeline:
    """
    Orchestrates the local-first embedding and indexing pipeline.
    Optimized for batch-processing on local CPU/GPU hardware.
    """

    BATCH_SIZE = 50   # Optimal chunk packet size for local memory buffers

    def __init__(self):
        self.client     = get_qdrant_client()
        self.embedder   = HuggingFaceEmbedder()
        self.vectorizer = BM25Vectorizer()

        # Ensure collection exists and matches the embedder layout
        setup_collection(self.client, embed_dim=self.embedder.EMBED_DIM)

    # ── Checkpoint management ─────────────────────────────────────────

    def _load_checkpoint(self) -> Set[str]:
        if CHECKPOINT_PATH.exists():
            data = json.loads(CHECKPOINT_PATH.read_text())
            done = set(data.get("indexed_chunk_ids", []))
            print(f"📌 Checkpoint found: {len(done)} chunks already indexed")
            return done
        return set()

    def _save_checkpoint(self, indexed_ids: Set[str]):
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(
            json.dumps({"indexed_chunk_ids": list(indexed_ids)})
        )

    # ── Core indexing logic ───────────────────────────────────────────

    def _index_batch(self, batch: List[Dict]) -> int:
        """Process one batch using local vectorization methods."""
        texts = [chunk["text"] for chunk in batch]

        # Step 1: BM25 sparse vectors (Batched & local)
        sparse_vecs = self.vectorizer.vectorize_batch(texts)

        # Step 2: Hugging Face dense vectors (Processed locally via list comprehension)
        dense_vecs = [self.embedder.embed_document(text) for text in texts]

        # Step 3: Build Qdrant points
        points = [
            build_point(chunk, dense_vecs[i], sparse_vecs[i])
            for i, chunk in enumerate(batch)
        ]

        # Step 4: Upsert to Qdrant
        self.client.upsert(
            collection_name=cfg.qdrant_collection,
            points=points,
            wait=True
        )
        return len(points)

    def run(
        self,
        chunks_path: Path = Path("data/chunks.json"),
        max_chunks: int = None
    ):
        print(f"\n📂 Loading chunks from {chunks_path}...")
        all_chunks: List[Dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
        print(f"   Total chunks: {len(all_chunks)}")

        if max_chunks:
            all_chunks = all_chunks[:max_chunks]
            print(f"   Limited to: {max_chunks} chunks (test mode)")

        done_ids  = self._load_checkpoint()
        remaining = [c for c in all_chunks if c["chunk_id"] not in done_ids]
        print(f"   To index: {len(remaining)} chunks ({len(done_ids)} already done)\n")

        if not remaining:
            print("✅ All chunks already indexed!")
            return

        total_indexed = 0
        start_time = time.monotonic()

        for batch_start in range(0, len(remaining), self.BATCH_SIZE):
            batch = remaining[batch_start : batch_start + self.BATCH_SIZE]
            batch_num = batch_start // self.BATCH_SIZE + 1
            total_batches = (len(remaining) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            print(f"[Batch {batch_num}/{total_batches}] Indexing {len(batch)} chunks...")

            try:
                count = self._index_batch(batch)
                total_indexed += count

                for chunk in batch:
                    done_ids.add(chunk["chunk_id"])
                self._save_checkpoint(done_ids)

                elapsed   = time.monotonic() - start_time
                rate      = total_indexed / elapsed  
                remaining_count = len(remaining) - total_indexed
                eta_min   = (remaining_count / rate / 60) if rate > 0 else 0
                print(f"  ✓ {total_indexed}/{len(remaining)} indexed | {rate:.1f} chunks/s | ETA: {eta_min:.0f}min\n")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                print(f"  ❌ Batch failed — checkpoint saved, safe to retry")
                raise

        elapsed = time.monotonic() - start_time
        print(f"✅ Indexing complete!")
        print(f"   Total indexed: {total_indexed}")
        print(f"   Time taken:    {elapsed/60:.1f} minutes")
        print(f"   Rate:          {total_indexed/elapsed:.1f} chunks/second")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.json",
                        help="Path to chunks JSON from Phase 3")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit chunks for testing (e.g. --max 20)")
    args = parser.parse_args()

    pipeline = IndexingPipeline()
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Error: chunks file not found: {chunks_path}")
        raise SystemExit(1)

    pipeline.run(chunks_path=chunks_path, max_chunks=args.max)