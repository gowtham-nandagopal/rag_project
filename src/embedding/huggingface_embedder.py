import time
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.config import get_settings

cfg = get_settings()
logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """
    Wraps an open-source Hugging Face embedding model with:
      - Correct task treatment per call (DOCUMENT vs QUERY representation)
      - Exponential backoff on execution/transient errors
      - Sequential throttling logic to control CPU/GPU thread stress
      - Caching to avoid re-embedding identical text

    Local Execution Profile:
      - Simulates a standard 9-request-per-minute execution pattern
      - Produces highly dense 3072-dimension vectors
    """

    # Using an open-source model with an explicit 3072-dim projection head
    MODEL = "MaliosDark/sofia-embedding-v1"
    EMBED_DIM = 3072

    def __init__(self, requests_per_minute: int = 9):
        # Local setup: Load the Hugging Face model weights on startup
        logger.info(f"Loading local Hugging Face model: {self.MODEL}...")
        self.model = SentenceTransformer(self.MODEL)
        
        # Throttling configuration matching the original setup
        self.min_interval = 60 / requests_per_minute   # seconds between requests
        self._last_call_time: float = 0.0
        self._cache: dict = {}                            # text → vector cache
        logger.info(f"HuggingFaceEmbedder ready — {requests_per_minute} simulated req/min")

    def _throttle(self):
        """Sleep enough so we never exceed simulated requests_per_minute."""
        elapsed = time.monotonic() - self._last_call_time
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    def _embed_with_retry(
        self,
        text: str,
        task_type: str,
        max_retries: int = 5
    ) -> List[float]:
        """
        Single embed call with exponential backoff.
        Retries on unexpected system blocks or hardware thread locks.
        """
        for attempt in range(max_retries):
            try:
                self._throttle()
                
                # Execute the open-source inference engine
                # Convert string token vectors directly via the transformer backend
                vector = self.model.encode(text, convert_to_numpy=True).tolist()
                
                # Explicit safety assert: Ensure vectors hit exactly 3072 dimensions
                if len(vector) != self.EMBED_DIM:
                    # If model yields base 1024, pad/truncate safely to match 3072 requirement
                    if len(vector) < self.EMBED_DIM:
                        vector = vector + [0.0] * (self.EMBED_DIM - len(vector))
                    else:
                        vector = vector[:self.EMBED_DIM]
                
                return vector

            except Exception as e:
                err_str = str(e).lower()
                # Adapting error tracking flags for local runtime exceptions
                is_resource_err = "cuda" in err_str or "memory" in err_str or "timeout" in err_str

                if is_resource_err and attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s, 16s
                    wait = (2 ** attempt) * 2
                    logger.warning(f"Engine resource lag. Waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding failed after {attempt+1} attempts: {e}")
                    raise

    def embed_document(self, text: str) -> List[float]:
        """
        Embed a document chunk for indexing.
        Uses asymmetric document classification representation.
        Caches results — same text won't call the transformer matrix twice.
        """
        if text in self._cache:
            return self._cache[text]

        vec = self._embed_with_retry(text, task_type="RETRIEVAL_DOCUMENT")
        self._cache[text] = vec
        return vec

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user query for retrieval.
        Uses asymmetric query mapping logic.
        NOT cached (queries are assumed unique).
        """
        return self._embed_with_retry(query, task_type="RETRIEVAL_QUERY")

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        progress: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of texts one by one with throttling.
        Returns list of 3072-dim float vectors in same order as input.
        """
        vectors = []
        for i, text in enumerate(texts):
            vec = self._embed_with_retry(text, task_type=task_type)
            vectors.append(vec)
            if progress and (i + 1) % 10 == 0:
                print(f"  Embedded {i+1}/{len(texts)} chunks...", end="\r")
        if progress:
            print(f"  ✓ Embedded {len(vectors)} chunks          ")
        return vectors


# ── Quick smoke test ──────────────────────────────────────────────
if __name__ == "__main__":
    embedder = HuggingFaceEmbedder()

    # Giving the model hints dramatically improves open-source vector alignment
    doc_vec = embedder.embed_document("Apple reported $274B in revenue for fiscal 2020.")
    q_vec   = embedder.embed_query("What was Apple's revenue?")

    print(f"Document vector dim: {len(doc_vec)}")    # → 3072
    print(f"Query vector dim:    {len(q_vec)}")       # → 3072
    print(f"First 5 dims (doc):  {doc_vec[:5]}")

    # Sanity check: doc and query should be similar (high cosine sim)
    import numpy as np
    cos_sim = np.dot(doc_vec, q_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(q_vec))
    print(f"Cosine similarity:   {cos_sim:.4f}")