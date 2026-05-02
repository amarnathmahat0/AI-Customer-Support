"""
api/server.py — FastAPI backend for the customer support agent.

Provides:
  - WebSocket /ws/{user_id} — real-time chat with streaming node events
  - GET /health — service health status
  - GET /metrics — session and agent performance metrics
  - GET /sessions/{user_id} — list sessions for a user
  - GET /sessions/{user_id}/{session_id}/history — conversation history
  - DELETE /memory/{user_id} — clear user memories

WebSocket message protocol:
  Client → Server: {"message": "user text", "session_id": "optional"}
  Server → Client: {"type": "node_active"|"node_complete"|"response"|"error", ...}
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Import agent (lazy to avoid circular imports)
from agent.graph import run_agent
from agent.memory import get_all_memories, clear_memories, check_memory_health
from knowledge_base.retriever import check_collection_health
from tools.order_db import init_order_db


app = FastAPI(
    title="Customer Support Agent API",
    description="Production-grade autonomous customer support with LangGraph + mem0",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# In-memory metrics store (per session)
# ──────────────────────────────────────────────────────────────────

class MetricsStore:
    """Thread-safe in-memory store for session metrics."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict] = defaultdict(self._new_session)
        self._global = {
            "total_messages": 0,
            "total_escalations": 0,
            "total_guardrail_retries": 0,
            "intent_counts": defaultdict(int),
            "model_usage": defaultdict(int),
            "response_times": []
        }
    
    def _new_session(self) -> Dict:
        return {
            "session_id": "",
            "user_id": "",
            "start_time": time.time(),
            "messages": 0,
            "escalations": 0,
            "guardrail_retries": 0,
            "response_times": [],
            "intents": defaultdict(int),
            "model_used": "",
            "conversation": []
        }
    
    def record(self, session_id: str, user_id: str, result: Dict, message: str) -> None:
        """Record metrics from an agent run result."""
        s = self._sessions[session_id]
        s["session_id"] = session_id
        s["user_id"] = user_id
        s["messages"] += 1
        s["model_used"] = result.get("model_used", "")
        
        latency = result.get("latency_ms", 0)
        s["response_times"].append(latency)
        
        intent = result.get("intent", "")
        if intent:
            s["intents"][intent] += 1
            self._global["intent_counts"][intent] += 1
        
        if result.get("escalated"):
            s["escalations"] += 1
            self._global["total_escalations"] += 1
        
        retries = result.get("guardrail_retries", 0)
        s["guardrail_retries"] += retries
        self._global["total_guardrail_retries"] += retries
        
        self._global["total_messages"] += 1
        self._global["response_times"].append(latency)
        
        if result.get("model_used"):
            self._global["model_usage"][result["model_used"]] += 1
        
        # Store conversation turn
        s["conversation"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        s["conversation"].append({
            "role": "assistant",
            "content": result.get("response", ""),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "model": result.get("model_used"),
                "latency_ms": result.get("latency_ms"),
                "guardrail_retries": result.get("guardrail_retries"),
                "tool_used": result.get("tool_used"),
                "escalated": result.get("escalated")
            }
        })
    
    def get_session_metrics(self, session_id: str) -> Dict:
        """Get metrics for a specific session."""
        s = self._sessions.get(session_id, self._new_session())
        times = s["response_times"]
        return {
            "session_id": session_id,
            "messages": s["messages"],
            "avg_response_ms": round(sum(times) / len(times)) if times else 0,
            "escalations": s["escalations"],
            "guardrail_retries": s["guardrail_retries"],
            "model_used": s["model_used"],
            "intent_distribution": dict(s["intents"])
        }
    
    def get_global_metrics(self) -> Dict:
        """Get aggregate metrics across all sessions."""
        times = self._global["response_times"]
        return {
            "total_messages": self._global["total_messages"],
            "total_escalations": self._global["total_escalations"],
            "total_guardrail_retries": self._global["total_guardrail_retries"],
            "avg_response_ms": round(sum(times) / len(times)) if times else 0,
            "intent_distribution": dict(self._global["intent_counts"]),
            "model_usage": dict(self._global["model_usage"]),
            "active_sessions": len(self._sessions)
        }
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id, {}).get("conversation", [])
    
    def list_sessions_for_user(self, user_id: str) -> List[str]:
        """List all session IDs for a user."""
        return [
            sid for sid, s in self._sessions.items()
            if s.get("user_id") == user_id
        ]


metrics = MetricsStore()


# ──────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize databases and services on startup."""
    logger.info("Support Agent API starting up...")
    
    try:
        init_order_db()
        logger.info("✓ Order database initialized")
    except Exception as e:
        logger.error(f"✗ Order database initialization failed: {e}")
    
    # Try to ingest FAQs if ChromaDB collection is empty
    try:
        from knowledge_base.ingest import ingest_faqs
        ingest_faqs()
        logger.info("✓ FAQ knowledge base ready")
    except Exception as e:
        logger.warning(f"FAQ ingestion warning: {e}")
    
    logger.info("Support Agent API ready")


# ──────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat with the support agent.
    
    Accepts messages from the client, runs the agent pipeline,
    and streams node lifecycle events back in real time.
    
    Message format (client → server):
      {"message": "user text", "session_id": "optional_session_id"}
    
    Events streamed (server → client):
      {"type": "node_active", "node": "memory_retrieval", "elapsed_ms": 0}
      {"type": "node_complete", "node": "intent_classifier", "elapsed_ms": 340, "metadata": {...}}
      {"type": "response", "text": "...", "metadata": {...}}
      {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: user={user_id}")
    
    # Session state
    session_id = None
    conversation_history = []
    
    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()
            
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
                continue
            
            user_message = data.get("message", "").strip()
            if not user_message:
                continue
            
            # Use provided session ID or generate a new one
            session_id = data.get("session_id") or session_id or str(uuid.uuid4())[:8]
            
            logger.info(f"Message from {user_id} (session {session_id}): {user_message[:80]}")
            
            # Create event queue for streaming
            event_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
            
            # Start agent run as background task
            agent_task = asyncio.create_task(
                run_agent(
                    user_id=user_id,
                    message=user_message,
                    session_id=session_id,
                    event_queue=event_queue,
                    conversation_history=conversation_history.copy()
                )
            )
            
            # Stream events as they come in
            result = None
            
            while True:
                try:
                    # Wait for next event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    await websocket.send_json(event)
                    
                    if event.get("type") == "response":
                        result = event.get("metadata", {})
                        result["response"] = event.get("text", "")
                        break
                    elif event.get("type") == "error":
                        break
                        
                except asyncio.TimeoutError:
                    # Check if agent task is done
                    if agent_task.done():
                        # Drain remaining events
                        while not event_queue.empty():
                            try:
                                event = event_queue.get_nowait()
                                await websocket.send_json(event)
                                if event.get("type") in ("response", "error"):
                                    if event.get("type") == "response":
                                        result = event.get("metadata", {})
                                        result["response"] = event.get("text", "")
                                    break
                            except asyncio.QueueEmpty:
                                break
                        break
                    continue
            
            # Await agent completion
            if not agent_task.done():
                try:
                    agent_result = await asyncio.wait_for(agent_task, timeout=120)
                    if result is None:
                        result = agent_result
                except asyncio.TimeoutError:
                    logger.error(f"Agent timed out for user {user_id}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Request timed out. Please try again."
                    })
                    continue
            else:
                agent_result = agent_task.result()
                if result is None:
                    result = agent_result
            
            if result is None:
                result = {}
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_message})
            response_text = result.get("response", "")
            if response_text:
                conversation_history.append({"role": "assistant", "content": response_text})
            
            # Keep history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            # Record metrics
            metrics.record(session_id, user_id, result, user_message)
            
            logger.info(
                f"Completed: user={user_id}, session={session_id}, "
                f"intent={result.get('intent')}, latency={result.get('latency_ms')}ms"
            )
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Check health status of all service dependencies.
    
    Returns status for: Ollama, ChromaDB, mem0, LangSmith, database.
    """
    health = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags")
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            primary = os.getenv("OLLAMA_PRIMARY_MODEL", "gemma3:1b")
            health["services"]["ollama"] = {
                "status": "online",
                "models_loaded": len(models),
                "primary_model_available": any(primary in m for m in model_names)
            }
    except Exception as e:
        health["services"]["ollama"] = {"status": "offline", "error": str(e)}
        health["status"] = "degraded"
    
    # Check ChromaDB
    try:
        chroma_health = check_collection_health()
        health["services"]["chromadb"] = {
            "status": "online" if chroma_health["healthy"] else "empty",
            "faq_count": chroma_health["count"],
            "message": chroma_health["message"]
        }
    except Exception as e:
        health["services"]["chromadb"] = {"status": "offline", "error": str(e)}
        health["status"] = "degraded"
    
    # Check mem0
    try:
        mem0_health = check_memory_health()
        health["services"]["mem0"] = {
            "status": "online" if mem0_health["healthy"] else "error",
            "message": mem0_health["message"]
        }
    except Exception as e:
        health["services"]["mem0"] = {"status": "offline", "error": str(e)}
        health["status"] = "degraded"
    
    # Check LangSmith
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
    if langsmith_key and langsmith_key != "your_langsmith_api_key_here":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.smith.langchain.com/ok",
                    headers={"x-api-key": langsmith_key}
                )
                health["services"]["langsmith"] = {
                    "status": "online" if response.status_code == 200 else "error",
                    "project": os.getenv("LANGSMITH_PROJECT")
                }
        except Exception as e:
            health["services"]["langsmith"] = {"status": "offline", "error": str(e)}
    else:
        health["services"]["langsmith"] = {
            "status": "not_configured",
            "message": "Set LANGSMITH_API_KEY to enable tracing"
        }
    
    # Check order database
    try:
        order_db = os.getenv("ORDER_DB", "./orders.db")
        conn = sqlite3.connect(order_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orders")
        count = cursor.fetchone()[0]
        conn.close()
        health["services"]["order_db"] = {"status": "online", "order_count": count}
    except Exception as e:
        health["services"]["order_db"] = {"status": "offline", "error": str(e)}
    
    return health


@app.get("/metrics")
async def get_metrics():
    """
    Return aggregate performance metrics across all sessions.
    
    Includes: message counts, response times, intent distribution,
    escalation rate, guardrail retry rate, model usage breakdown.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        **metrics.get_global_metrics()
    }


@app.get("/metrics/{session_id}")
async def get_session_metrics(session_id: str):
    """Return metrics for a specific session."""
    return metrics.get_session_metrics(session_id)


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    """List all conversation sessions for a user."""
    sessions = metrics.list_sessions_for_user(user_id)
    return {"user_id": user_id, "sessions": sessions}


@app.get("/sessions/{user_id}/{session_id}/history")
async def get_session_history(user_id: str, session_id: str):
    """Get full conversation history for a specific session."""
    conversation = metrics.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "user_id": user_id,
        "session_id": session_id,
        "conversation": conversation
    }


@app.get("/memory/{user_id}")
async def get_user_memories(user_id: str):
    """Get all persistent memories for a user."""
    memories = get_all_memories(user_id)
    return {
        "user_id": user_id,
        "memories": memories,
        "count": len(memories)
    }


@app.delete("/memory/{user_id}")
async def clear_user_memories(user_id: str):
    """Delete all persistent memories for a user."""
    success = clear_memories(user_id)
    return {
        "user_id": user_id,
        "success": success,
        "message": "Memories cleared" if success else "Failed to clear memories"
    }


@app.get("/")
async def root():
    """API root — returns service info and links."""
    return {
        "service": "Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "websocket": "ws://HOST:PORT/ws/{user_id}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        reload=False,
        log_level="info"
    )
