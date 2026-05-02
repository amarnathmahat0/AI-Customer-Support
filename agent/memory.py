"""
agent/memory.py — Cross-session persistent memory using mem0.

Configures mem0 with local Ollama LLM + ChromaDB vector store for
fully offline, cross-session user memory storage and retrieval.
"""

import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

_memory_instance = None


def get_mem0_config():
    """
    Build the mem0 configuration object for local Ollama + ChromaDB.
    
    Returns:
        mem0 MemoryConfig instance with vector_store, llm, and embedder settings.
    """
    from mem0.configs.base import MemoryConfig
    from mem0.vector_stores.configs import VectorStoreConfig
    from mem0.llms.configs import LlmConfig
    from mem0.embeddings.configs import EmbedderConfig

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    mem0_db_path = os.getenv("MEM0_DB_PATH", "./mem0_db")
    primary_model = os.getenv("OLLAMA_PRIMARY_MODEL", "gemma3:1b")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest")
    
    return MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="chroma",
            config={
                "collection_name": "support_mem",
                "path": mem0_db_path
            }
        ),
        llm=LlmConfig(
            provider="ollama",
            config={
                "model": primary_model,
                "ollama_base_url": ollama_url,
                "temperature": 0.0,
                "max_tokens": 2048
            }
        ),
        embedder=EmbedderConfig(
            provider="ollama",
            config={
                "model": embed_model,
                "ollama_base_url": ollama_url
            }
        )
    )


def get_memory():
    """
    Get or initialize the mem0 Memory singleton instance.
    
    Returns:
        mem0 Memory instance, or None if initialization fails.
    """
    global _memory_instance
    
    if _memory_instance is not None:
        return _memory_instance
    
    try:
        from mem0 import Memory
        config = get_mem0_config()
        _memory_instance = Memory(config=config)
        logger.info("mem0 initialized successfully with Ollama + ChromaDB")
    except ImportError:
        logger.error("mem0 not installed. Run: pip install mem0ai")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize mem0: {e}")
        return None
    
    return _memory_instance


def retrieve_memories(user_id: str, query: str, limit: int = 5) -> List[Dict]:
    """
    Retrieve relevant memories for a user based on the current query.
    
    Args:
        user_id: The unique user identifier
        query: The current message to find relevant memories for
        limit: Maximum number of memories to retrieve
        
    Returns:
        List of memory dictionaries with 'memory', 'id', 'created_at' fields.
    """
    memory = get_memory()
    
    if memory is None:
        logger.warning("mem0 unavailable — returning empty memories")
        return []
    
    if not query or not query.strip():
        logger.warning(f"Empty query for memory retrieval — skipping search for user {user_id}")
        return []
    
    try:
        results = memory.search(query=query, user_id=user_id, limit=limit)
        
        memories = []
        # Handle different mem0 response formats
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    memories.append({
                        "memory": r.get("memory", r.get("text", str(r))),
                        "id": r.get("id", ""),
                        "created_at": r.get("created_at", ""),
                        "score": r.get("score", 0.0)
                    })
        elif isinstance(results, dict) and "results" in results:
            for r in results["results"]:
                memories.append({
                    "memory": r.get("memory", ""),
                    "id": r.get("id", ""),
                    "created_at": r.get("created_at", ""),
                    "score": r.get("score", 0.0)
                })
        
        logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
        return memories
        
    except Exception as e:
        # Fall back to get_all on embedding failures (e.g. model not ready)
        logger.warning(f"Memory search failed for user {user_id}: {e} — falling back to get_all")
        try:
            results = memory.get_all(user_id=user_id)
            memories = []
            if isinstance(results, list):
                for r in results[:limit]:
                    if isinstance(r, dict):
                        memories.append({
                            "memory": r.get("memory", r.get("text", str(r))),
                            "id": r.get("id", ""),
                            "created_at": r.get("created_at", ""),
                            "score": 0.0
                        })
            return memories
        except Exception as e2:
            logger.error(f"Memory get_all fallback also failed for user {user_id}: {e2}")
            return []


def save_memory(user_id: str, messages: List[Dict]) -> bool:
    """
    Save conversation turn to persistent memory for a user.
    
    Args:
        user_id: The unique user identifier
        messages: List of message dicts with 'role' and 'content' keys
                  e.g. [{"role": "user", "content": "..."}, 
                         {"role": "assistant", "content": "..."}]
        
    Returns:
        True if save successful, False otherwise.
    """
    memory = get_memory()
    
    if memory is None:
        logger.warning("mem0 unavailable — memory not saved")
        return False
    
    try:
        memory.add(messages=messages, user_id=user_id)
        logger.info(f"Saved memory for user {user_id}: {len(messages)} messages")
        return True
    except Exception as e:
        logger.error(f"Failed to save memory for user {user_id}: {e}")
        return False


def get_all_memories(user_id: str) -> List[Dict]:
    """
    Retrieve all stored memories for a user (for UI display).
    
    Args:
        user_id: The unique user identifier
        
    Returns:
        List of all memory dictionaries for this user.
    """
    memory = get_memory()
    
    if memory is None:
        return []
    
    try:
        results = memory.get_all(user_id=user_id)

        
        memories = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    memories.append({
                        "memory": r.get("memory", r.get("text", str(r))),
                        "id": r.get("id", ""),
                        "created_at": r.get("created_at", ""),
                    })
        elif isinstance(results, dict) and "results" in results:
            for r in results["results"]:
                memories.append({
                    "memory": r.get("memory", ""),
                    "id": r.get("id", ""),
                    "created_at": r.get("created_at", ""),
                })
        
        return memories
    except Exception as e:
        logger.error(f"Failed to get all memories for user {user_id}: {e}")
        return []


def clear_memories(user_id: str) -> bool:
    """
    Delete all memories for a user.
    
    Args:
        user_id: The unique user identifier
        
    Returns:
        True if successful, False otherwise.
    """
    memory = get_memory()
    
    if memory is None:
        return False
    
    try:
        memory.delete_all(user_id=user_id)
        logger.info(f"Cleared all memories for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear memories for user {user_id}: {e}")
        return False


def format_memory_context(memories: List[Dict]) -> str:
    """
    Format retrieved memories into a clean context string for the LLM.
    
    Args:
        memories: List of memory dicts from retrieve_memories()
        
    Returns:
        Formatted string for LLM prompt context injection.
    """
    if not memories:
        return "No prior context available for this user."
    
    lines = ["USER MEMORY CONTEXT (from previous conversations):"]
    for mem in memories:
        text = mem.get("memory", "")
        if text:
            lines.append(f"  • {text}")
    
    return "\n".join(lines)


def check_memory_health() -> Dict:
    """
    Verify mem0 is accessible and can perform basic operations.
    
    Returns:
        Dictionary with 'healthy' bool and 'message' string.
    """
    try:
        memory = get_memory()
        if memory is None:
            return {"healthy": False, "message": "mem0 initialization failed"}
        
        # Try a simple write/read cycle
        test_user = "_health_check_user_"
        memory.add(
            messages=[{"role": "user", "content": "health check test"}],
            user_id=test_user
        )
        results = memory.get_all(filters={"user_id": test_user})
        memory.delete_all(user_id=test_user)
        
        return {"healthy": True, "message": "mem0 read/write working"}
    except Exception as e:
        return {"healthy": False, "message": f"mem0 health check failed: {e}"}
