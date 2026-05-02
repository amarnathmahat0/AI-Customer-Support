"""
ingest.py — Ingests FAQ entries from faqs.json into ChromaDB.

Uses the all-minilm:latest Ollama model for embeddings and falls back to
the same lightweight embedding model when needed.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

FAQ_PATH = Path(__file__).parent.parent / "data" / "faqs.json"
COLLECTION_NAME = "support_faqs"


def get_chroma_path() -> str:
    """Return the ChromaDB persistence path from environment variables."""
    return os.getenv("CHROMA_PATH", "./chroma_db")


def get_embed_model() -> str:
    """Return the primary embedding model name from environment."""
    return os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest")


def get_embed_fallback() -> str:
    """Return the fallback embedding model name from environment."""
    return os.getenv("OLLAMA_EMBED_FALLBACK", "all-minilm:latest")


def get_ollama_url() -> str:
    """Return the Ollama base URL from environment variables."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_embedding_function(model: str = None):
    """
    Create a ChromaDB embedding function using an Ollama model.
    
    Args:
        model: Ollama model name. Defaults to OLLAMA_EMBED_MODEL env var.
        
    Returns:
        ChromaDB OllamaEmbeddingFunction instance.
    """
    if model is None:
        model = get_embed_model()
    
    ollama_url = get_ollama_url()
    
    return embedding_functions.OllamaEmbeddingFunction(
        url=f"{ollama_url}/api/embeddings",
        model_name=model
    )


def load_faqs() -> List[Dict]:
    """
    Load FAQ entries from the JSON file.
    
    Returns:
        List of FAQ dictionaries with id, question, answer, category, tags.
    """
    if not FAQ_PATH.exists():
        logger.error(f"FAQ file not found: {FAQ_PATH}")
        return []
    
    with open(FAQ_PATH, "r") as f:
        faqs = json.load(f)
    
    logger.info(f"Loaded {len(faqs)} FAQ entries from {FAQ_PATH}")
    return faqs


def ingest_faqs(force_reload: bool = False) -> bool:
    """
    Ingest FAQ entries into ChromaDB vector store.
    
    Creates the collection if it doesn't exist, and only re-ingests
    if force_reload=True or the collection is empty.
    
    Args:
        force_reload: If True, delete existing collection and re-ingest.
        
    Returns:
        True if successful, False otherwise.
    """
    chroma_path = get_chroma_path()
    logger.info(f"Initializing ChromaDB at {chroma_path}")
    
    try:
        client = chromadb.PersistentClient(path=chroma_path)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return False
    
    # Try primary embedding model, fall back if needed
    embed_fn = None
    for model in [get_embed_model(), get_embed_fallback()]:
        try:
            embed_fn = get_embedding_function(model)
            # Test it with a simple embedding
            test = embed_fn(["test"])
            if test:
                logger.info(f"Using embedding model: {model}")
                break
        except Exception as e:
            logger.warning(f"Embedding model {model} failed: {e}")
            embed_fn = None
    
    if embed_fn is None:
        logger.error("All embedding models failed — using default embedding")
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
    
    # Handle force reload
    if force_reload:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass
    
    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"description": "Customer support FAQ knowledge base"}
        )
    except Exception as e:
        logger.error(f"Failed to create/get collection: {e}")
        return False
    
    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0 and not force_reload:
        logger.info(f"Collection already has {existing_count} entries — skipping ingest")
        return True
    
    # Load FAQs
    faqs = load_faqs()
    if not faqs:
        logger.error("No FAQs to ingest")
        return False
    
    # Prepare documents for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for faq in faqs:
        # Combine question + answer for better semantic search
        doc_text = f"Q: {faq['question']}\nA: {faq['answer']}"
        documents.append(doc_text)
        metadatas.append({
            "faq_id": faq["id"],
            "category": faq["category"],
            "question": faq["question"],
            "tags": ", ".join(faq.get("tags", []))
        })
        ids.append(faq["id"])
    
    # Ingest in batches
    batch_size = 10
    total = len(documents)
    
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            logger.info(f"Ingested batch {i//batch_size + 1}: {len(batch_docs)} entries")
        except Exception as e:
            logger.error(f"Failed to ingest batch at index {i}: {e}")
            return False
    
    logger.info(f"Successfully ingested {total} FAQ entries into ChromaDB")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    force = "--force" in sys.argv
    success = ingest_faqs(force_reload=force)
    
    if success:
        print("✅ FAQ ingestion complete")
    else:
        print("❌ FAQ ingestion failed")
        sys.exit(1)
