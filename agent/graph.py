"""
agent/graph.py — LangGraph stateful graph for the customer support agent.

Defines the full pipeline as a directed graph with conditional routing:

  memory_retrieval
       ↓
  intent_classifier
       ↓
  [conditional routing based on intent]
  ORDER_QUERY → order_tool
  GENERAL_FAQ → faq_tool
  COMPLAINT   → complaint_tool
  ESCALATE    → escalate_tool
       ↓
  llm_generate
       ↓
  guardrails
       ↓
  memory_save

Uses AsyncSqliteSaver for persistent checkpointing across sessions.
LangSmith tracing is configured via environment variables.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

import aiosqlite
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.nodes import _EVENT_QUEUES
from agent.serde import FilteredJsonPlusSerializer
from agent.nodes import (
    AgentState,
    node_memory_retrieval,
    node_intent_classifier,
    node_order_tool,
    node_faq_tool,
    node_complaint_tool,
    node_escalate_tool,
    node_llm_generate,
    node_guardrails,
    node_memory_save,
    emit_event
)

logger = logging.getLogger(__name__)

# LangSmith configuration
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "support-agent-portfolio"))
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))


def get_checkpoint_db() -> str:
    """Return the path to the SQLite checkpoint database (kept for reference)."""
    return os.getenv("CHECKPOINT_DB", "./checkpoints.db")


# Module-level in-memory checkpointer with custom serde that filters asyncio.Queue
_checkpointer = MemorySaver(serde=FilteredJsonPlusSerializer())


def route_by_intent(state: AgentState) -> Literal["order_tool", "faq_tool", "complaint_tool", "escalate_tool"]:
    """
    Conditional routing function — determines which tool node to execute.

    Routes based on the classified intent:
    - ORDER_QUERY  → order_tool
    - GENERAL_FAQ  → faq_tool
    - COMPLAINT    → complaint_tool
    - ESCALATE     → escalate_tool (default for unknown intents)

    Args:
        state: Current agent state with 'intent' set

    Returns:
        Name of the next node to execute.
    """
    intent = state.get("intent", "GENERAL_FAQ")

    routing_map = {
        "ORDER_QUERY": "order_tool",
        "GENERAL_FAQ": "faq_tool",
        "COMPLAINT": "complaint_tool",
        "ESCALATE": "escalate_tool"
    }

    next_node = routing_map.get(intent, "faq_tool")
    logger.info(f"Routing intent '{intent}' → {next_node}")
    return next_node


async def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph support agent pipeline.

    Uses MemorySaver for in-process checkpointing — avoids stale SQLite
    checkpoints from previous server runs corrupting the agent state.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("memory_retrieval", node_memory_retrieval)
    graph.add_node("intent_classifier", node_intent_classifier)
    graph.add_node("order_tool", node_order_tool)
    graph.add_node("faq_tool", node_faq_tool)
    graph.add_node("complaint_tool", node_complaint_tool)
    graph.add_node("escalate_tool", node_escalate_tool)
    graph.add_node("llm_generate", node_llm_generate)
    graph.add_node("guardrails", node_guardrails)
    graph.add_node("memory_save", node_memory_save)
    
    # Set entry point
    graph.set_entry_point("memory_retrieval")
    
    # Linear edges for initial pipeline
    graph.add_edge("memory_retrieval", "intent_classifier")
    
    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "order_tool": "order_tool",
            "faq_tool": "faq_tool",
            "complaint_tool": "complaint_tool",
            "escalate_tool": "escalate_tool"
        }
    )
    
    # All tool nodes connect to LLM generation
    graph.add_edge("order_tool", "llm_generate")
    graph.add_edge("faq_tool", "llm_generate")
    graph.add_edge("complaint_tool", "llm_generate")
    graph.add_edge("escalate_tool", "llm_generate")
    
    # Post-generation pipeline
    graph.add_edge("llm_generate", "guardrails")
    graph.add_edge("guardrails", "memory_save")
    graph.add_edge("memory_save", END)
    
    # Compile with in-memory checkpointer (clean per process, no stale state)
    compiled = graph.compile(checkpointer=_checkpointer)
    
    logger.info("LangGraph pipeline compiled successfully")
    return compiled


# Module-level compiled graph (lazy initialization)
_compiled_graph = None


async def get_graph():
    """
    Get or create the compiled LangGraph instance (singleton).
    
    Returns:
        Compiled LangGraph StateGraph.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = await build_graph()
    return _compiled_graph


async def run_agent(
    user_id: str,
    message: str,
    session_id: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Run the support agent pipeline for a single user message.
    
    This is the main entry point called by the FastAPI WebSocket handler.
    It initializes the agent state, runs the LangGraph pipeline, and
    returns the final state with response and metadata.
    
    Args:
        user_id: Unique user identifier for memory and session tracking
        message: The user's current message
        session_id: Optional session ID (generated if not provided)
        event_queue: asyncio.Queue for streaming node events to WebSocket
        conversation_history: Previous messages in this conversation
        
    Returns:
        Dictionary containing:
          - response: Final agent response text
          - intent: Classified intent
          - confidence: Classification confidence
          - model_used: Which LLM model was used
          - latency_ms: LLM generation latency
          - guardrail_retries: Number of guardrail retries
          - tool_used: Which tool was invoked
          - escalated: Whether human escalation was triggered
          - session_id: Session identifier
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    start_time = time.time()
    
    # Register event queue in module-level registry (keyed by user+session)
    # Queue MUST NOT be in the state dict — asyncio.Queue is not msgpack-serializable
    queue_key = f"{user_id}_{session_id}"
    if event_queue:
        _EVENT_QUEUES[queue_key] = event_queue
    
    # Build initial state — no event_queue field
    initial_state = AgentState(
        user_id=user_id,
        session_id=session_id,
        messages=conversation_history or [],
        current_message=message,
        memories=[],
        intent="",
        confidence=0.0,
        tool_result={},
        tool_used="",
        llm_response="",
        guardrail_retries=0,
        model_used="",
        latency_ms=0,
        escalated=False,
        tone_soften=False,
        start_time=start_time,
        node_times={},
        langsmith_run_id=None
    )
    
    # Unique thread_id per message — prevents stale checkpoint from overriding state
    config = {
        "configurable": {
            "thread_id": f"{user_id}_{session_id}_{uuid.uuid4().hex[:8]}"
        }
    }
    
    try:
        graph = await get_graph()
        
        # Run the graph
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # Guard against None or completely empty state
        if final_state is None:
            logger.error("Graph execution returned None — likely a node failed")
            final_state = {}
        
        total_ms = round((time.time() - start_time) * 1000)
        
        llm_response = final_state.get("llm_response", "")
        # If guardrails exhausted retries and left an empty response, use a safe fallback
        if not llm_response or not llm_response.strip():
            llm_response = (
                "I'm sorry, I had trouble generating a proper response. "
                "Please try rephrasing your question, or contact support@company.com directly."
            )
            logger.warning("Empty llm_response after graph run — using fallback")
        
        result = {
            "response": llm_response,
            "intent": final_state.get("intent", ""),
            "confidence": final_state.get("confidence", 0.0),
            "model_used": final_state.get("model_used", ""),
            "latency_ms": final_state.get("latency_ms", 0),
            "total_latency_ms": total_ms,
            "guardrail_retries": final_state.get("guardrail_retries", 0),
            "tool_used": final_state.get("tool_used", ""),
            "escalated": final_state.get("escalated", False),
            "session_id": session_id,
            "memories_retrieved": len(final_state.get("memories", []))
        }
        
        # Emit final response event
        if event_queue:
            try:
                event_queue.put_nowait({
                    "type": "response",
                    "text": result["response"],
                    "metadata": {
                        k: v for k, v in result.items() if k != "response"
                    }
                })
            except Exception:
                pass
        
        logger.info(
            f"Agent run complete: user={user_id}, intent={result['intent']}, "
            f"model={result['model_used']}, latency={result['latency_ms']}ms, "
            f"total={total_ms}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent run failed for user {user_id}: {e}", exc_info=True)
        
        error_response = {
            "response": (
                "I apologize, but I encountered an unexpected error. "
                "Please try again or contact support@company.com directly."
            ),
            "intent": "ERROR",
            "confidence": 0.0,
            "model_used": "none",
            "latency_ms": 0,
            "total_latency_ms": round((time.time() - start_time) * 1000),
            "guardrail_retries": 0,
            "tool_used": "",
            "escalated": False,
            "session_id": session_id,
            "memories_retrieved": 0,
            "error": str(e)
        }
        
        if event_queue:
            try:
                event_queue.put_nowait({
                    "type": "error",
                    "message": f"Agent error: {str(e)}"
                })
                event_queue.put_nowait({
                    "type": "response",
                    "text": error_response["response"],
                    "metadata": error_response
                })
            except Exception:
                pass
        
        return error_response
