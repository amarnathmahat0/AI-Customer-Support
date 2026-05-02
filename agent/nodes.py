"""
agent/nodes.py — LangGraph node functions for the customer support agent.
Optimized for Mac M2 with 3-4GB RAM: background memory save, streaming,
thinking events, and M2-specific Ollama tuning.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.memory import retrieve_memories, save_memory, format_memory_context
from agent.guardrails import run_guardrails
from knowledge_base.retriever import retrieve_faqs, format_context_for_llm
from tools.order_db import get_order_status
from tools.slack_tool import escalate_to_human

logger = logging.getLogger(__name__)

# Model cascade
PRIMARY_MODEL = os.getenv("OLLAMA_PRIMARY_MODEL", "qwen2.5:0.5b")
FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "gemma3:1b")
ULTRA_LIGHT_MODEL = os.getenv("OLLAMA_ULTRA_LIGHT_MODEL", "qwen2.5:0.5b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.6"))

# Mac M2 Ollama tuning
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "8"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))

# Escalation trigger words
ESCALATION_WORDS = re.compile(
    r'\b(human|agent|manager|supervisor|real person|speak to someone|'
    r'escalate|transfer|connect me|live agent|call me|phone)\b',
    re.IGNORECASE
)


class AgentState(TypedDict, total=False):
    """
    LangGraph state for the support agent.
    NOTE: event_queue is intentionally excluded — asyncio.Queue cannot
    be msgpack-serialized. Use _EVENT_QUEUES registry instead.
    """
    user_id: str
    session_id: str
    messages: List[Dict]
    current_message: str
    memories: List[Dict]
    intent: str
    confidence: float
    tool_result: Dict
    tool_used: str
    llm_response: str
    guardrail_retries: int
    model_used: str
    latency_ms: float
    escalated: bool
    tone_soften: bool
    start_time: float
    node_times: Dict
    langsmith_run_id: Optional[str]


# Thread-local event queue registry
_EVENT_QUEUES: Dict[str, Any] = {}


def emit_event(state: AgentState, event_type: str, node: str, **kwargs) -> None:
    """
    Emit a node lifecycle event to the WebSocket event queue.
    """
    thread_id = state.get("user_id", "") + "_" + state.get("session_id", "")
    queue: Optional[asyncio.Queue] = _EVENT_QUEUES.get(thread_id)
    if queue is None:
        return
    
    elapsed = round((time.time() - state.get("start_time", time.time())) * 1000)
    
    event = {
        "type": event_type,
        "node": node,
        "elapsed_ms": elapsed,
        **kwargs
    }
    
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning(f"Event queue full — dropping event for node {node}")
    except Exception as e:
        logger.error(f"Failed to emit event: {e}")


# ──────────────────────────────────────────────────────────────────
# Ollama LLM calling with streaming + M2 optimizations
# ──────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
)
async def _call_ollama(model: str, prompt: str, system: str = "", state: AgentState = None) -> str:
    """
    Make an Ollama API call with Mac M2 optimizations and optional streaming.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": state is not None,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
            "num_thread": OLLAMA_NUM_THREAD,
            "num_ctx": OLLAMA_NUM_CTX,
            "keep_alive": OLLAMA_KEEP_ALIVE
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload
        )
        response.raise_for_status()
        
        # Non-streaming fallback
        if state is None:
            data = response.json()
            return data["message"]["content"]
        
        # ── STREAMING MODE ──
        full_text = ""
        first_token = True
        start_tok = time.time()
        queue_key = f"{state.get('user_id', '')}_{state.get('session_id', '')}"
        queue = _EVENT_QUEUES.get(queue_key)
        
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if first_token:
                        first_token = False
                        if queue:
                            ttft = round((time.time() - start_tok) * 1000)
                            queue.put_nowait({
                                "type": "ttft",
                                "elapsed_ms": ttft,
                                "node": "llm_generate"
                            })
                    full_text += token
                    if queue:
                        queue.put_nowait({
                            "type": "token",
                            "text": token,
                            "node": "llm_generate"
                        })
            except Exception:
                continue
        
        return full_text


async def call_llm_with_fallback(prompt: str, system: str = "", state: AgentState = None) -> Tuple[str, str]:
    """
    Call Ollama LLM with automatic fallback cascade.
    """
    seen = set()
    models = []
    for candidate in [PRIMARY_MODEL, FALLBACK_MODEL, ULTRA_LIGHT_MODEL]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            models.append(candidate)
    
    for model in models:
        try:
            logger.info(f"Calling LLM: {model} (streaming={state is not None})")
            response = await _call_ollama(model, prompt, system, state=state)
            return response, model
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            continue
    
    logger.error("All LLM models failed")
    return (
        "I'm sorry, I'm experiencing technical difficulties right now. "
        "Please try again in a moment.",
        "none"
    )


# ──────────────────────────────────────────────────────────────────
# Node 1: Memory Retrieval
# ──────────────────────────────────────────────────────────────────

async def node_memory_retrieval(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "memory_retrieval")
    
    user_id = state.get("user_id", "anonymous")
    current_message = state.get("current_message", "")
    
    try:
        memories = retrieve_memories(
            user_id=user_id,
            query=current_message,
            limit=int(os.getenv("MEMORY_CONTEXT_TURNS", "3"))
        )
        state["memories"] = memories
        logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        state["memories"] = []
    
    emit_event(state, "node_complete", "memory_retrieval",
               metadata={"memory_count": len(state.get("memories", []))})
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 2: Intent Classifier
# ──────────────────────────────────────────────────────────────────

async def node_intent_classifier(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "intent_classifier")
    
    message = state.get("current_message", "")
    memories = state.get("memories", [])
    
    # Fast-path: explicit order ID
    if re.search(r'ORD-\d{6}', message, re.IGNORECASE):
        state["intent"] = "ORDER_QUERY"
        state["confidence"] = 0.98
        emit_event(state, "thinking", "intent_classifier",
                   text="Detected explicit order ID — skipping LLM classification")
        emit_event(state, "node_complete", "intent_classifier",
                   metadata={"intent": "ORDER_QUERY", "confidence": 0.98})
        return state

    # Fast-path: escalation keywords
    if ESCALATION_WORDS.search(message):
        state["intent"] = "ESCALATE"
        state["confidence"] = 0.95
        emit_event(state, "thinking", "intent_classifier",
                   text="Detected escalation keyword — routing to human agent")
        emit_event(state, "node_complete", "intent_classifier",
                   metadata={"intent": "ESCALATE", "confidence": 0.95})
        return state

    # Fast-path: keyword classification
    keyword_intent, keyword_confidence = _keyword_classify(message)
    if keyword_confidence >= 0.80:
        state["intent"] = keyword_intent
        state["confidence"] = keyword_confidence
        emit_event(state, "thinking", "intent_classifier",
                   text=f"Keyword match: {keyword_intent} (confidence: {keyword_confidence:.2f})")
        emit_event(state, "node_complete", "intent_classifier",
                   metadata={"intent": keyword_intent, "confidence": keyword_confidence})
        return state

    # LLM-based classification
    memory_context = format_memory_context(memories)
    
    system_prompt = """You are an intent classification engine for a customer support system.
Classify the user message into EXACTLY ONE of these intents and provide a confidence score.

INTENTS:
- ORDER_QUERY: User asking about a specific order status, tracking, delivery, or order-related action
- GENERAL_FAQ: General questions about products, policies, returns, shipping, payments, accounts
- COMPLAINT: User is frustrated, upset, or expressing dissatisfaction about a product or service
- ESCALATE: User explicitly wants to speak with a human, manager, or the issue is beyond bot capability

Respond ONLY with valid JSON in this exact format:
{"intent": "INTENT_NAME", "confidence": 0.XX, "reasoning": "brief reason"}

Do not include any other text."""

    prompt = f"""User context: {memory_context}

Current message: "{message}"

Classify this intent:"""
    
    try:
        response, _ = await call_llm_with_fallback(prompt, system_prompt)
        
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            intent = parsed.get("intent", "GENERAL_FAQ").upper()
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            
            valid_intents = {"ORDER_QUERY", "GENERAL_FAQ", "COMPLAINT", "ESCALATE"}
            if intent not in valid_intents:
                intent = "GENERAL_FAQ"
                confidence = 0.5
        else:
            intent, confidence = _keyword_classify(message)
            reasoning = "Fallback keyword classification"
            
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        intent, confidence = _keyword_classify(message)
        reasoning = "Error fallback"
    
    # Override to ESCALATE if confidence is too low
    if confidence < ESCALATION_THRESHOLD and intent != "ESCALATE":
        logger.info(f"Low confidence ({confidence:.2f}) — escalating")
        reasoning = f"Low confidence ({confidence:.2f}) — escalating to human"
        intent = "ESCALATE"
    
    state["intent"] = intent
    state["confidence"] = confidence
    
    emit_event(state, "thinking", "intent_classifier",
               text=f"Intent: {intent} (confidence: {confidence:.2f}) — {reasoning}")
    emit_event(state, "node_complete", "intent_classifier",
               metadata={"intent": intent, "confidence": confidence})
    
    return state


def _keyword_classify(message: str) -> Tuple[str, float]:
    msg_lower = message.lower()
    
    if any(word in msg_lower for word in ["order", "tracking", "shipped", "delivery", "ord-"]):
        return "ORDER_QUERY", 0.75
    elif any(word in msg_lower for word in ["angry", "frustrated", "unacceptable", "terrible", "awful", "worst"]):
        return "COMPLAINT", 0.75
    elif any(word in msg_lower for word in ["human", "agent", "manager", "person", "escalate"]):
        return "ESCALATE", 0.90
    else:
        return "GENERAL_FAQ", 0.65


# ──────────────────────────────────────────────────────────────────
# Node 3a: Order Query Tool
# ──────────────────────────────────────────────────────────────────

async def node_order_tool(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "order_tool")
    
    message = state.get("current_message", "")
    
    order_match = re.search(r'ORD-\d{6}', message, re.IGNORECASE)
    
    if order_match:
        order_id = order_match.group().upper()
        try:
            result = get_order_status(order_id)
            state["tool_result"] = result
            state["tool_used"] = "get_order_status"
            status_msg = result.get("status", "unknown")
            emit_event(state, "thinking", "order_tool",
                       text=f"Found order {order_id} — status: {status_msg}")
        except Exception as e:
            logger.error(f"Order lookup failed: {e}")
            state["tool_result"] = {"error": str(e), "found": False}
            state["tool_used"] = "get_order_status"
            emit_event(state, "thinking", "order_tool",
                       text=f"Order lookup failed: {str(e)}")
    else:
        state["tool_result"] = {
            "found": False,
            "message": "No order ID detected. Please provide your order ID in format ORD-XXXXXX"
        }
        state["tool_used"] = "get_order_status"
        emit_event(state, "thinking", "order_tool",
                   text="No order ID found in message — asking user to provide one")
    
    emit_event(state, "node_complete", "order_tool",
               metadata={"order_found": state["tool_result"].get("found", False)})
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 3b: FAQ Tool
# ──────────────────────────────────────────────────────────────────

async def node_faq_tool(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "faq_tool")
    
    message = state.get("current_message", "")
    
    try:
        results = retrieve_faqs(query=message, n_results=3)
        state["tool_result"] = {
            "type": "faq_results",
            "results": results,
            "context": format_context_for_llm(results)
        }
        state["tool_used"] = "faq_search"
        emit_event(state, "thinking", "faq_tool",
                   text=f"Retrieved {len(results)} FAQ entries from knowledge base")
    except Exception as e:
        logger.error(f"FAQ retrieval failed: {e}")
        state["tool_result"] = {"type": "faq_results", "results": [], "context": ""}
        state["tool_used"] = "faq_search"
        emit_event(state, "thinking", "faq_tool",
                   text="FAQ retrieval failed — continuing without context")
    
    emit_event(state, "node_complete", "faq_tool",
               metadata={"results_count": len(state["tool_result"].get("results", []))})
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 3c: Complaint Tool
# ──────────────────────────────────────────────────────────────────

async def node_complaint_tool(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "complaint_tool")
    
    state["tone_soften"] = True
    message = state.get("current_message", "")
    
    # Try order lookup
    order_result = None
    order_match = re.search(r'ORD-\d{6}', message, re.IGNORECASE)
    if order_match:
        order_id = order_match.group().upper()
        try:
            order_result = get_order_status(order_id)
        except Exception as e:
            logger.warning(f"Order lookup in complaint handler failed: {e}")
    
    # Run FAQ search
    faq_results = []
    try:
        faq_results = retrieve_faqs(query=message, n_results=2)
    except Exception as e:
        logger.warning(f"FAQ search in complaint handler failed: {e}")
    
    state["tool_result"] = {
        "type": "complaint_combined",
        "order_result": order_result,
        "faq_results": faq_results,
        "context": format_context_for_llm(faq_results)
    }
    state["tool_used"] = "complaint_handler"
    
    emit_event(state, "thinking", "complaint_tool",
               text="Customer complaint detected — enabling empathetic tone + combined lookup")
    emit_event(state, "node_complete", "complaint_tool",
               metadata={"tone_soften": True})
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 3d: Escalation Tool
# ──────────────────────────────────────────────────────────────────

async def node_escalate_tool(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "escalate_tool")
    
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "")
    confidence = state.get("confidence", 0.0)
    message = state.get("current_message", "")
    
    messages = state.get("messages", [])
    recent = messages[-4:] if len(messages) >= 4 else messages
    snippet = "\n".join([f"{m['role'].upper()}: {m['content'][:100]}" for m in recent])
    
    summary = (
        f"User requires human assistance. "
        f"Last message: '{message[:200]}'. "
        f"Classifier confidence: {confidence:.2f}. "
        f"Intent: {state.get('intent', 'ESCALATE')}."
    )
    
    try:
        result = await escalate_to_human(
            user_id=user_id,
            summary=summary,
            session_id=session_id,
            intent=state.get("intent"),
            confidence=confidence,
            conversation_snippet=snippet
        )
        state["tool_result"] = result
        state["tool_used"] = "escalate_to_human"
        state["escalated"] = True
        emit_event(state, "thinking", "escalate_tool",
                   text=f"Escalation sent to Slack — success: {result.get('success', False)}")
    except Exception as e:
        logger.error(f"Escalation failed: {e}")
        state["tool_result"] = {"success": False, "error": str(e)}
        state["tool_used"] = "escalate_to_human"
        state["escalated"] = True
        emit_event(state, "thinking", "escalate_tool",
                   text=f"Escalation failed: {str(e)} — logged locally")
    
    emit_event(state, "node_complete", "escalate_tool",
               metadata={"escalated": True, "slack_success": state["tool_result"].get("success", False)})
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 4: LLM Generate (with streaming)
# ──────────────────────────────────────────────────────────────────

async def node_llm_generate(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "llm_generate")
    
    start = time.time()
    
    message = state.get("current_message", "")
    intent = state.get("intent", "GENERAL_FAQ")
    memories = state.get("memories", [])
    tool_result = state.get("tool_result", {})
    tone_soften = state.get("tone_soften", False)
    escalated = state.get("escalated", False)
    max_words = int(os.getenv("MAX_RESPONSE_WORDS", "150"))
    
    tool_context = _build_tool_context(intent, tool_result, escalated)
    memory_context = format_memory_context(memories)
    
    tone_instruction = (
        "The customer is frustrated or upset. Use extra empathy, acknowledge their feelings, "
        "and be especially patient and understanding. Start with an apology if appropriate."
        if tone_soften else
        "Be helpful, professional, and concise."
    )
    
    system_prompt = f"""You are Alex, a friendly and professional customer support agent.
Your goal is to help customers efficiently and empathetically.

Guidelines:
- {tone_instruction}
- Keep responses under {max_words} words
- Be specific and actionable, not vague
- Only use order IDs exactly as provided in the context below — never invent them
- Do not make up information — only use what is provided in the context
- End with a helpful offer to assist further if appropriate"""

    prompt_parts = []
    if memory_context and memory_context != "No prior context available for this user.":
        prompt_parts.append(f"CUSTOMER CONTEXT:\n{memory_context}")
    if tool_context and tool_context != "No tool data available.":
        prompt_parts.append(f"RELEVANT INFORMATION:\n{tool_context}")
    prompt_parts.append(f"CUSTOMER MESSAGE:\n{message}")
    prompt_parts.append("Please provide a helpful response:")
    
    prompt = "\n\n".join(prompt_parts)
    
    try:
        response, model_used = await call_llm_with_fallback(prompt, system_prompt, state=state)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        response = (
            "I apologize for the inconvenience. I'm experiencing a technical issue "
            "right now. Please try again shortly or contact our support team."
        )
        model_used = "none"
    
    latency_ms = round((time.time() - start) * 1000)
    
    state["llm_response"] = response
    state["model_used"] = model_used
    state["latency_ms"] = latency_ms
    
    word_count = len(response.split()) if response else 0
    emit_event(state, "thinking", "llm_generate",
               text=f"Generated {word_count} words using {model_used} in {latency_ms}ms")
    emit_event(state, "node_complete", "llm_generate",
               metadata={"model": model_used, "latency_ms": latency_ms})
    
    return state


def _build_tool_context(intent: str, tool_result: Dict, escalated: bool) -> str:
    if not tool_result:
        return "No tool data available."
    
    if escalated:
        success = tool_result.get("success", False)
        if success:
            return "ESCALATION STATUS: Successfully notified human support team via Slack. A human agent will contact the customer shortly."
        else:
            return "ESCALATION STATUS: Escalation notification encountered an issue, but the request has been logged. Inform the customer a human agent will follow up."
    
    result_type = tool_result.get("type", "")
    
    if result_type == "faq_results":
        return tool_result.get("context", "No FAQ results found.")
    
    if result_type == "complaint_combined":
        parts = []
        order = tool_result.get("order_result")
        if order and order.get("found"):
            parts.append(f"ORDER INFO: {json.dumps(order, indent=2)}")
        parts.append(tool_result.get("context", ""))
        return "\n".join(parts) if parts else "No additional context found."
    
    if "order_id" in tool_result or "found" in tool_result:
        if tool_result.get("found"):
            return f"ORDER DETAILS:\n{json.dumps(tool_result, indent=2)}"
        else:
            error = tool_result.get("error", tool_result.get("message", "Order not found"))
            return f"ORDER LOOKUP RESULT: {error}"
    
    return str(tool_result)[:500]


# ──────────────────────────────────────────────────────────────────
# Node 5: Guardrails (simplified — no LLM retry)
# ──────────────────────────────────────────────────────────────────

async def node_guardrails(state: AgentState) -> AgentState:
    emit_event(state, "node_active", "guardrails")
    
    response = state.get("llm_response", "")
    
    result = run_guardrails(response)
    retry_count = 0
    
    # On failure, just trim or use safe fallback — no LLM retry (saves 2-5s)
    if not result.passed:
        failed = [v.check_name for v in result.validations if not v.passed]
        logger.warning(f"Guardrail failed: {failed} — applying safe fallback")
        
        # Simple fixes without LLM regeneration
        if any(c == "response_length" for c in failed):
            words = response.split()
            response = " ".join(words[:150]) + "..."
            retry_count = 1
        elif any(c == "non_empty_response" for c in failed):
            response = "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
            retry_count = 1
        elif any(c == "professional_tone" for c in failed):
            response = "I apologize for any confusion. How can I help you today?"
            retry_count = 1
    
    state["llm_response"] = response
    state["guardrail_retries"] = retry_count
    
    emit_event(state, "thinking", "guardrails",
               text=f"Guardrails: {'passed' if result.passed else 'applied safe fallback'} ({result.word_count} words)")
    emit_event(state, "node_complete", "guardrails",
               metadata={
                   "passed": result.passed,
                   "retries": retry_count,
                   "word_count": result.word_count
               })
    
    return state


# ──────────────────────────────────────────────────────────────────
# Node 6: Memory Save (BACKGROUND — non-blocking)
# ──────────────────────────────────────────────────────────────────

async def node_memory_save(state: AgentState) -> AgentState:
    """
    Persist conversation to mem0 in a BACKGROUND THREAD so it never
    blocks the WebSocket response. On M2 with 3-4GB RAM, mem0.add()
    can take 10-15s. We refuse to let the user wait for that.
    """
    emit_event(state, "node_active", "memory_save")
    
    user_id = state.get("user_id", "anonymous")
    user_message = state.get("current_message", "")
    agent_response = state.get("llm_response", "")
    
    # ── CRITICAL OPTIMIZATION: Fire-and-forget background save ──
    def _bg_save():
        try:
            msgs = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": agent_response}
            ]
            save_memory(user_id=user_id, messages=msgs)
        except Exception as e:
            logger.error(f"Background memory save failed: {e}")
    
    # Use asyncio's default thread pool executor
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _bg_save)
    # ─────────────────────────────────────────────────────────────
    
    metadata = {
        "intent": state.get("intent", ""),
        "confidence": state.get("confidence", 0.0),
        "tool_used": state.get("tool_used", ""),
        "llm_model_used": state.get("model_used", ""),
        "latency_ms": state.get("latency_ms", 0),
        "guardrail_retries": state.get("guardrail_retries", 0),
        "memory_entries_count": len(state.get("memories", [])),
        "escalated": state.get("escalated", False)
    }
    
    emit_event(state, "thinking", "memory_save",
               text="Saving conversation to memory (background — non-blocking)")
    emit_event(state, "node_complete", "memory_save", metadata=metadata)
    
    return state