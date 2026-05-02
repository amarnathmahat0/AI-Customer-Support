"""
retriever.py — ChromaDB vector search retrieval for FAQ knowledge base.

Provides semantic search over the ingested FAQ entries using Ollama embeddings.
Falls back gracefully if ChromaDB or embedding model is unavailable.
"""

import logging
import os
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

COLLECTION_NAME = "support_faqs"
_chroma_client = None
_collection = None


def get_chroma_path() -> str:
    """Return the ChromaDB persistence path from environment."""
    return os.getenv("CHROMA_PATH", "./chroma_db")


def get_embed_model() -> str:
    """Return the primary embedding model name."""
    return os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest")


def get_embed_fallback() -> str:
    """Return the fallback embedding model name."""
    return os.getenv("OLLAMA_EMBED_FALLBACK", "all-minilm:latest")


def get_ollama_url() -> str:
    """Return the Ollama base URL."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_collection():
    """
    Lazily initialize and return the ChromaDB collection.
    
    Returns:
        ChromaDB collection object or None if unavailable.
    """
    global _chroma_client, _collection
    
    if _collection is not None:
        return _collection
    
    chroma_path = get_chroma_path()
    ollama_url = get_ollama_url()
    
    try:
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB at {chroma_path}: {e}")
        return None
    
    # Try primary then fallback embedding model
    embed_fn = None
    for model in [get_embed_model(), get_embed_fallback()]:
        try:
            embed_fn = embedding_functions.OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings",
                model_name=model
            )
            # Quick test
            embed_fn(["connectivity test"])
            logger.info(f"Retriever using embedding model: {model}")
            break
        except Exception as e:
            logger.warning(f"Embedding model {model} unavailable: {e}")
            embed_fn = None
    
    if embed_fn is None:
        logger.warning("Using default embedding function as fallback")
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )
        count = _collection.count()
        logger.info(f"Connected to FAQ collection with {count} entries")
    except Exception as e:
        logger.error(f"FAQ collection not found: {e}. Run 'python knowledge_base/ingest.py' first.")
        _collection = None
    
    return _collection


def retrieve_faqs(
    query: str,
    n_results: int = 3,
    category_filter: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve the most relevant FAQ entries for a given query.
    
    Uses semantic similarity search via ChromaDB. Returns formatted
    results including the question, answer, and relevance metadata.
    
    Args:
        query: The user's question or message to search against
        n_results: Number of top results to return (default 3)
        category_filter: Optional category to filter results (orders, 
                         returns, account, payments, shipping, products)
        
    Returns:
        List of result dictionaries with keys:
          - question: The FAQ question
          - answer: The FAQ answer  
          - category: FAQ category
          - faq_id: Unique FAQ identifier
          - distance: Semantic distance (lower = more relevant)
          - tags: Comma-separated tags
    """
    collection = _get_collection()
    
    if collection is None:
        logger.warning("FAQ collection unavailable — returning empty results")
        return []
    
    if not query or not query.strip():
        logger.warning("Empty query provided to retriever")
        return []
    
    try:
        # Build where clause for category filter
        where = None
        if category_filter:
            where = {"category": category_filter}
        
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                dist = results["distances"][0][i] if results.get("distances") else 1.0
                
                formatted.append({
                    "question": meta.get("question", ""),
                    "answer": _extract_answer(doc),
                    "category": meta.get("category", ""),
                    "faq_id": meta.get("faq_id", ""),
                    "distance": round(dist, 4),
                    "tags": meta.get("tags", ""),
                    "relevance_score": round(1.0 - min(dist, 1.0), 4)
                })
        
        logger.info(f"Retrieved {len(formatted)} FAQs for query: '{query[:50]}...'")
        return formatted
        
    except Exception as e:
        logger.error(f"FAQ retrieval failed for query '{query[:50]}': {e}")
        return []


def _extract_answer(doc_text: str) -> str:
    """
    Extract just the answer portion from a combined Q&A document string.
    
    Args:
        doc_text: Combined string in format "Q: question\\nA: answer"
        
    Returns:
        Just the answer portion of the document.
    """
    if "\nA: " in doc_text:
        return doc_text.split("\nA: ", 1)[1]
    return doc_text


def format_context_for_llm(results: List[Dict]) -> str:
    """
    Format retrieved FAQ results into a clean context string for the LLM.
    
    Args:
        results: List of FAQ result dictionaries from retrieve_faqs()
        
    Returns:
        Formatted string suitable for inclusion in LLM prompts.
    """
    if not results:
        return "No relevant FAQ entries found."
    
    lines = ["RELEVANT FAQ KNOWLEDGE BASE ENTRIES:"]
    lines.append("=" * 40)
    
    for i, r in enumerate(results, 1):
        lines.append(f"\n[FAQ {i}] Category: {r['category'].upper()}")
        lines.append(f"Q: {r['question']}")
        lines.append(f"A: {r['answer']}")
        lines.append(f"Relevance: {r['relevance_score']:.0%}")
    
    lines.append("=" * 40)
    return "\n".join(lines)


def check_collection_health() -> Dict:
    """
    Verify ChromaDB collection is accessible and populated.
    
    Returns:
        Dictionary with 'healthy' bool, 'count' int, and 'message' string.
    """
    collection = _get_collection()
    
    if collection is None:
        return {
            "healthy": False,
            "count": 0,
            "message": "Collection not found — run ingest.py"
        }
    
    try:
        count = collection.count()
        return {
            "healthy": count > 0,
            "count": count,
            "message": f"{count} FAQ entries loaded" if count > 0 else "Collection is empty"
        }
    except Exception as e:
        return {
            "healthy": False,
            "count": 0,
            "message": f"Health check failed: {e}"
        }
