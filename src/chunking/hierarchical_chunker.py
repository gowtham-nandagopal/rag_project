from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from typing import List, Dict, Tuple, Optional
from src.ingestion.preprocessor import ProcessedDocument


class HierarchicalChunker:
    """
    Parent-child chunking strategy (Small-to-Big Retrieval).

    Creates two levels:
      Parent: 512 tokens  → sent to LLM for rich context
      Child:  128 tokens  → indexed in Qdrant for precise retrieval

    Workflow:
      1. Embed and index CHILD chunks in Qdrant
      2. At query time, retrieve child chunks
      3. Look up their parent_id → fetch parent from docstore
      4. Send parent text to LLM (more context, better answer)
    """

    def __init__(self):
        # chunk_sizes=[512, 128]: first split into 512-token parents,
        # then each parent is split into 128-token children
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[512, 128],
            chunk_overlap=20,   # small overlap prevents boundary misses
        )
        # In-memory store to look up parents by ID at retrieval time
        self.docstore = SimpleDocumentStore()

    def chunk(self, doc: ProcessedDocument) -> Tuple[List[dict], List[dict]]:
        """
        Returns:
          parents: list of parent chunk dicts (stored in docstore, sent to LLM)
          children: list of child chunk dicts (indexed in Qdrant for retrieval)
        """
        llama_doc = Document(
            text=doc.content,
            metadata={
                **doc.metadata,
                "doc_id":         doc.doc_id,
                "chunk_strategy": "hierarchical",
            }
        )

        all_nodes = self.parser.get_nodes_from_documents([llama_doc])

        # get_leaf_nodes() returns the smallest (128-token) children
        # get_root_nodes() returns the largest (512-token) parents
        leaf_nodes = get_leaf_nodes(all_nodes)
        root_nodes = get_root_nodes(all_nodes)

        # Store all nodes in docstore (needed for parent lookup at retrieval)
        self.docstore.add_documents(all_nodes)

        # Build parent chunks dict
        parents = []
        for node in root_nodes:
            parents.append({
                "chunk_id": node.node_id,
                "text":     node.text,
                "metadata": {
                    **node.metadata,
                    "chunk_type": "parent",
                    "char_count": len(node.text),
                }
            })

        # Build child chunks dict — THESE are indexed in Qdrant
        # Each child stores parent_id so we can look up the parent
        children = []
        for node in leaf_nodes:
            parent_id = node.parent_node.node_id if node.parent_node else None
            children.append({
                "chunk_id":  node.node_id,
                "text":      node.text,
                "parent_id": parent_id,  # ← key link: child → parent
                "metadata": {
                    **node.metadata,
                    "chunk_type": "child",
                    "parent_id":  parent_id,
                    "char_count": len(node.text),
                }
            })

        return parents, children

    def get_parent_text(self, parent_id: str) -> Optional[str]:
        """Look up parent chunk text by ID (called at retrieval time)."""
        node = self.docstore.get_document(parent_id)
        return node.text if node else None