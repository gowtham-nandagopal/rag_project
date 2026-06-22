import logging
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.ingestion.preprocessor import ProcessedDocument

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_local_embedder() -> HuggingFaceEmbedding:
    """
    Loads a local Hugging Face embedding model.
    Runs 100% offline on your CPU/GPU with zero rate limits or costs.
    """
    logger.info("Loading local Hugging Face embedding model (bge-small-en-v1.5)...")
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=64  # Local models can handle much larger batches safely
    )


class SemanticChunker:
    """
    Topic-boundary chunker for long narrative documents.
    Powered by a local Hugging Face embedding model.
    """

    def __init__(self, breakpoint_percentile_threshold: int = 95):
        # 1. Use the local embedder instead of Gemini
        embed_model = build_local_embedder()
        
        self.parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=1,  # Compare 1 sentence at a time for precise topic cuts
        )

    def chunk(self, doc: ProcessedDocument) -> List[dict]:
        """
        Chunk a ProcessedDocument into semantically coherent nodes.
        Returns a list of dicts ready for Qdrant upsert.
        """
        llama_doc = Document(
            text=doc.content,
            metadata={
                **doc.metadata,
                "doc_id":         doc.doc_id,
                "chunk_strategy": "semantic",  # Fixed the strategy tracking tag
            }
        )

        logger.info(f"Starting local semantic chunking for document: {doc.doc_id}")
        
        # 2. Runs completely offline on your computer
        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks = []
        for i, node in enumerate(nodes):
            chunks.append({
                "chunk_id":    f"{doc.doc_id}_sem_{i}",
                "text":        node.text,
                "metadata": {
                    **node.metadata,
                    "chunk_index":  i,
                    "total_chunks": len(nodes),
                    "char_count":   len(node.text),
                }
            })
            
        return chunks