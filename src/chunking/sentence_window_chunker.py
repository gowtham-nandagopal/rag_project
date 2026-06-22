from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from typing import List
from src.ingestion.preprocessor import ProcessedDocument


class SentenceWindowChunker:
    """
    Sentence-level precision with contextual window for comprehension.

    Each node stores:
      node.text          → single sentence (what gets embedded & matched)
      node.metadata['window'] → ±window_size sentences around it (what LLM reads)

    window_size=3 means: 3 sentences before + matched sentence + 3 after.
    So LLM always sees 7 sentences of context even if search matched just 1.
    """

    def __init__(self, window_size: int = 3):
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",           # where window text is stored
            original_text_metadata_key="original_sentence"  # original matched sentence
        )

    def chunk(self, doc: ProcessedDocument) -> List[dict]:
        """
        Returns list of chunk dicts. Each has both 'text' (single sentence)
        and 'window' in metadata (7-sentence context block).
        """
        llama_doc = Document(
            text=doc.content,
            metadata={
                **doc.metadata,
                "doc_id":         doc.doc_id,
                "chunk_strategy": "sentence_window",
            }
        )

        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks = []
        for i, node in enumerate(nodes):
            chunks.append({
                "chunk_id": f"{doc.doc_id}_sw_{i}",
                "text":     node.text,         # single sentence → embedded
                "metadata": {
                    **node.metadata,
                    "chunk_index": i,
                    "total_chunks": len(nodes),
                    # 'window' key holds the ±3 sentence context block
                    # (automatically populated by SentenceWindowNodeParser)
                }
            })
        return chunks