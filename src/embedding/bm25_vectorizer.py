from typing import List, Dict, Tuple
from fastembed import SparseTextEmbedding, SparseEmbedding
import logging

logger = logging.getLogger(__name__)


class BM25Vectorizer:
    """
    Converts text to BM25 sparse vectors using fastembed.

    The Qdrant/bm25 model:
      - Runs 100% locally (no API, no rate limit)
      - Memory: ~50MB for the model
      - Speed: ~5,000 chunks/second on CPU
      - Vocabulary: ~30,000 English tokens
    """

    MODEL_NAME = "Qdrant/bm25"

    def __init__(self):
        # Downloads model on first run (~50MB), cached to ~/.cache/fastembed
        print("⬇ Loading BM25 model (first run downloads ~50MB)...")
        self.model = SparseTextEmbedding(model_name=self.MODEL_NAME)
        print("✅ BM25 model loaded")

    def vectorize(self, text: str) -> Dict:
        """
        Returns a Qdrant-compatible sparse vector dict:
          {"indices": [23, 4891, 12003, ...],
           "values":  [0.42, 1.87, 0.93, ...]}

        The indices are token IDs in the BM25 vocabulary.
        The values are the BM25 TF-IDF weights.
        """
        # embed() returns a generator — next() gets the first (only) result
        sparse: SparseEmbedding = next(self.model.embed([text]))
        return {
            "indices": sparse.indices.tolist(),
            "values":  sparse.values.tolist()
        }

    def vectorize_batch(self, texts: List[str]) -> List[Dict]:
        """
        Vectorize a list of texts efficiently.
        fastembed batches internally — much faster than looping.
        """
        results = []
        for sparse in self.model.embed(texts, batch_size=64):
            results.append({
                "indices": sparse.indices.tolist(),
                "values":  sparse.values.tolist()
            })
        return results


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    vectorizer = BM25Vectorizer()

    text1 = "Apple reported $274 billion revenue in fiscal year 2020."
    text2 = "Employee EMP-12345 submitted a leave request."

    v1 = vectorizer.vectorize(text1)
    v2 = vectorizer.vectorize(text2)

    print(f"Text 1 — non-zero tokens: {len(v1['indices'])}")
    print(f"Text 2 — non-zero tokens: {len(v2['indices'])}")
    print(f"Sample indices: {v1['indices'][:5]}")
    print(f"Sample values:  {[round(v,3) for v in v1['values'][:5]]}")