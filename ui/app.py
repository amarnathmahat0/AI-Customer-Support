"""
ui/app.py — Streamlit 4-panel observability dashboard for the support agent.
Optimized for Mac M2: health fixes, thinking steps, TTFT, typewriter effect.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# ── FIX: Load .env from project root so env vars actually work ──
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)
# ────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Support Agent — Observability Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

API_BASE = os.getenv("FASTAPI_URL", "http://localhost:8000")
WS_BASE = API_BASE.replace("http://", "ws://").replace("https://", "wss://")
PIPELINE_NODES = ["memory", "classifier", "tool", "llm", "guardrails", "save", "done"]

NODE_LABELS = {
    "memory": "Memory",
    "classifier": "Classifier",
    "tool": "Tool",
    "llm": "LLM Gen",
    "guardrails": "Guardrails",
    "save": "Save",
    "done": "Done"
}

NODE_MAP = {
    "memory_retrieval": "memory",
    "intent_classifier": "classifier",
    "order_tool": "tool",
    "faq_tool": "tool",
    "complaint_tool": "tool",
    "escalate_tool": "tool",
    "llm_generate": "llm",
    "guardrails": "guardrails",
    "memory_save": "save"
}

# ──────────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "user_id": f"user_{str(uuid.uuid4())[:6]}",
        "session_id": str(uuid.uuid4())[:8],
        "messages": [],
        "memories": [],
        "metrics": {
            "messages": 0,
            "response_times": [],
            "escalations": 0,
            "guardrail_retries": 0,
            "model_used": "—",
            "intents": {}
        },
        "system_health": {},
        "last_health_check": -1,
        "node_states": {},
        "node_times": {},
        "last_metadata": {},
        "processing": False,
        "langsmith_run_id": None,
        "sessions": [],
        "escalated_this_turn": False,
        "last_tool": "",
        "last_intent": "",
        "last_thinking": [],
        "last_ttft_ms": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="stAppViewContainer"] { background: #f7f2ea; }
[data-testid="stMarkdownContainer"] { color: #3b342b; }

.pipeline-wrap {
    display: flex; align-items: center; gap: 0;
    background: #fffaf2; border-radius: 14px;
    padding: 16px 20px; margin-bottom: 12px;
    border: 1px solid #e3d7c3; overflow-x: auto;
    box-shadow: 0 12px 32px rgba(83, 65, 48, 0.08);
}
.node-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 14px; border-radius: 24px;
    font-size: 12px; font-weight: 600;
    border: 2px solid transparent; white-space: nowrap;
    transition: all 0.3s ease;
}
.node-pending { background: #f3ece3; color: #857056; border-color: #ddd0c1; }
.node-active  {
    background: #fff3de; color: #9f7e43; border-color: #c79f67;
    animation: pulse-border 1.2s ease-in-out infinite;
    box-shadow: 0 0 14px rgba(199, 159, 103, 0.25);
}
.node-complete { background: #eaf5ee; color: #386455; border-color: #88b19c; }
.node-escalate { background: #f8e8e6; color: #9b3f33; border-color: #d69c8b; }
.node-arrow { color: #b8a48b; font-size: 16px; padding: 0 4px; flex-shrink: 0; }
.node-time { font-size: 10px; color: #8a7f70; margin-top: 2px; }
@keyframes pulse-border {
    0%   { box-shadow: 0 0 8px rgba(199,159,103,0.2); border-color: #c79f67; }
    50%  { box-shadow: 0 0 20px rgba(199,159,103,0.35); border-color: #ebc89b; }
    100% { box-shadow: 0 0 8px rgba(199,159,103,0.2); border-color: #c79f67; }
}

.chat-user {
    display: flex; justify-content: flex-end; margin: 8px 0;
}
.chat-user .bubble {
    background: #c9b08c; color: #251f19;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; max-width: 72%; font-size: 14px;
}
.chat-agent {
    display: flex; justify-content: flex-start; margin: 8px 0;
    flex-direction: column;
}
.chat-agent .bubble {
    background: #fffbf5; color: #3c342c;
    border-radius: 4px 18px 18px 18px;
    padding: 10px 16px; max-width: 78%; font-size: 14px;
    border: 1px solid #e3d7c3;
}
.msg-meta {
    font-size: 10px; color: #8e7f6c; margin-top: 4px;
    padding-left: 4px;
}
.typing-indicator {
    display: inline-flex; gap: 4px; padding: 10px 14px;
    background: #fff7ec; border-radius: 4px 18px 18px 18px;
    border: 1px solid #e3d7c3;
}
.typing-dot {
    width: 8px; height: 8px; background: #b59868;
    border-radius: 50%;
    animation: typing-bounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
    40% { transform: translateY(-6px); opacity: 1; }
}
.escalation-banner {
    background: #fff4ea; border: 1px solid #d4a870;
    border-radius: 8px; padding: 10px 14px;
    color: #8b5a35; font-size: 13px; margin: 8px 0;
}
.guardrail-badge {
    display: inline-block; background: #f7efe6;
    border: 1px solid #d1ad7d; border-radius: 12px;
    padding: 2px 8px; font-size: 10px; color: #8b5a35;
    margin-left: 8px;
}

.memory-card {
    background: #fffbf5; border: 1px solid #e3d7c3;
    border-radius: 10px; padding: 10px 12px; margin: 6px 0;
    font-size: 12px; color: #3f362e;
}
.memory-time { font-size: 10px; color: #8e7f6c; margin-top: 4px; }
.memory-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: #f4ede5; color: #8b5a35;
    border-radius: 12px; padding: 2px 10px;
    font-size: 11px; font-weight: 600; margin-bottom: 10px;
}

.metric-section {
    background: #fffbf5; border: 1px solid #e3d7c3;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
}
.metric-section h4 {
    color: #7d6c55; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 0 0 10px 0; padding-bottom: 6px;
    border-bottom: 1px solid #e3d7c3;
}
.health-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 4px 0;
    font-size: 12px; color: #4d4134;
}
.health-dot-green { color: #3d7e5f; }
.health-dot-red   { color: #a04333; }
.health-dot-gray  { color: #8e7f6c; }
.intent-bar-row {
    margin: 4px 0; font-size: 11px; color: #7d6c55;
}
.langsmith-link {
    display: inline-block; background: #fff6ec;
    border: 1px solid #e3d7c3; border-radius: 6px;
    padding: 6px 12px; color: #8b5a35; font-size: 12px;
    text-decoration: none; margin-top: 6px;
}
.panel-header {
    color: #7d6c55; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #e3d7c3;
}
.chat-wrap {
    background: #fffbf5; border: 1px solid #e3d7c3;
    border-radius: 10px; padding: 16px;
    min-height: 380px; max-height: 420px;
    overflow-y: auto;
}

/* Typewriter fade-in animation */
@keyframes fadeInWord {
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────

def relative_time(iso_str: str) -> str:
    if not iso_str:
        return "just now"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", ""))
        delta = datetime.now() - dt
        if delta.seconds < 60:
            return f"{delta.seconds}s ago"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60}m ago"
        elif delta.days < 1:
            return f"{delta.seconds // 3600}h ago"
        else:
            return f"{delta.days}d ago"
    except Exception:
        return "recently"

def get_health_icon(status: str) -> str:
    status_map = {
        "online": "🟢",
        "offline": "🔴",
        "degraded": "🟡",
        "not_configured": "⚫",
        "empty": "🟡",
        "error": "🔴",
        "unknown": "⚫"
    }
    return status_map.get(status, "⚫")

def intent_color(intent: str) -> str:
    return {
        "ORDER_QUERY": "#60a5fa",
        "GENERAL_FAQ": "#34d399",
        "COMPLAINT": "#f87171",
        "ESCALATE": "#f59e0b"
    }.get(intent, "#8888aa")

def render_pipeline_nodes(node_states: Dict, node_times: Dict, escalated: bool = False) -> str:
    parts = ['<div class="pipeline-wrap">']
    
    for i, node_key in enumerate(PIPELINE_NODES):
        state = node_states.get(node_key, "pending")
        label = NODE_LABELS[node_key]
        time_ms = node_times.get(node_key, "")
        
        if state == "active":
            css = "node-pill node-active"
            icon = "⟳"
        elif state == "complete":
            css = "node-pill node-complete"
            icon = "✓"
        else:
            css = "node-pill node-pending"
            icon = ""
        
        time_str = f'<div class="node-time">{time_ms}ms</div>' if time_ms else ""
        parts.append(
            f'<div style="display:flex;flex-direction:column;align-items:center">'
            f'<div class="{css}">{icon} {label}</div>'
            f'{time_str}'
            f'</div>'
        )
        
        if i < len(PIPELINE_NODES) - 1:
            if node_key == "classifier" and escalated:
                parts.append(
                    '<div style="display:flex;flex-direction:column;align-items:center">'
                    '<span class="node-arrow">→</span>'
                    '<div class="node-pill node-escalate" style="font-size:10px">⚡ Slack</div>'
                    '</div>'
                )
            parts.append('<span class="node-arrow">→</span>')
    
    parts.append('</div>')
    return ''.join(parts)

def render_chat_message(msg: Dict) -> str:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    meta = msg.get("metadata", {})
    
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    if role == "user":
        return f'''
        <div class="chat-user">
            <div class="bubble">{content}</div>
        </div>'''
    else:
        meta_parts = []
        if meta.get("model"):
            meta_parts.append(f"via {meta['model']}")
        if meta.get("latency_ms"):
            meta_parts.append(f"{meta['latency_ms']/1000:.1f}s")
        if meta.get("intent"):
            meta_parts.append(meta["intent"])
        if meta.get("confidence") is not None:
            meta_parts.append(f"conf {meta['confidence']:.2f}")
        if meta.get("guardrail_retries") is not None:
            meta_parts.append(f"{meta['guardrail_retries']} retries")
        
        meta_str = " · ".join(meta_parts)
        
        guardrail_badge = ""
        if meta.get("guardrail_retries", 0) > 0:
            guardrail_badge = f'<span class="guardrail-badge">Guardrail: {meta["guardrail_retries"]} retry</span>'
        
        return f'''
        <div class="chat-agent">
            <div class="bubble">{content}{guardrail_badge}</div>
            <div class="msg-meta">{meta_str}</div>
        </div>'''

def typewriter_html(text: str, meta: str = "") -> str:
    """Render a typewriter-effect bubble using CSS word-by-word fade-in."""
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_meta = meta.replace("<", "&lt;").replace(">", "&gt;")
    
    words = safe_text.split(" ")
    word_spans = []
    for i, word in enumerate(words):
        delay = i * 0.04
        word_spans.append(
            f'<span style="animation: fadeInWord 0.1s ease forwards; '
            f'animation-delay: {delay:.2f}s; opacity:0;">{word}</span>'
        )
    
    words_html = " ".join(word_spans)
    
    return f'''
    <div class="chat-agent">
        <div class="bubble" style="white-space: pre-wrap;">{words_html}</div>
        <div class="msg-meta">{safe_meta}</div>
    </div>
    '''

def fetch_health_sync() -> Dict:
    try:
        response = httpx.get(f"{API_BASE}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "services": {}, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "offline", "services": {}, "error": str(e)}

def fetch_memories_sync(user_id: str) -> List[Dict]:
    try:
        response = httpx.get(f"{API_BASE}/memory/{user_id}", timeout=5.0)
        return response.json().get("memories", [])
    except Exception:
        return []

def clear_memories_sync(user_id: str) -> bool:
    try:
        response = httpx.delete(f"{API_BASE}/memory/{user_id}", timeout=5.0)
        return response.json().get("success", False)
    except Exception:
        return False

def send_message_sync(user_id: str, session_id: str, message: str) -> Dict:
    import websocket as ws_lib
    import threading
    
    result = {
        "response": "",
        "events": [],
        "thinking_steps": [],
        "tokens": [],
        "ttft_ms": 0,
        "error": None,
        "intent": "",
        "confidence": 0.0,
        "model_used": "",
        "latency_ms": 0,
        "guardrail_retries": 0,
        "tool_used": "",
        "escalated": False
    }
    done_event = threading.Event()
    
    ws_url = f"{WS_BASE}/ws/{user_id}"
    
    def on_message(ws, raw_msg):
        try:
            msg = json.loads(raw_msg)
            result["events"].append(msg)
            
            if msg.get("type") == "thinking":
                result.setdefault("thinking_steps", []).append(msg)
            
            elif msg.get("type") == "ttft":
                result["ttft_ms"] = msg.get("elapsed_ms", 0)
            
            elif msg.get("type") == "token":
                result.setdefault("tokens", []).append(msg.get("text", ""))
            
            elif msg.get("type") == "response":
                result["response"] = msg.get("text", "")
                meta = msg.get("metadata", {})
                result.update({
                    "intent": meta.get("intent", ""),
                    "confidence": meta.get("confidence", 0.0),
                    "model_used": meta.get("model_used", ""),
                    "latency_ms": meta.get("latency_ms", 0),
                    "guardrail_retries": meta.get("guardrail_retries", 0),
                    "tool_used": meta.get("tool_used", ""),
                    "escalated": meta.get("escalated", False)
                })
                done_event.set()
                ws.close()
                
            elif msg.get("type") == "error":
                result["error"] = msg.get("message", "Unknown error")
                result["response"] = "I'm sorry, I encountered an error. Please try again."
                done_event.set()
                ws.close()
                
        except Exception as e:
            pass
    
    def on_error(ws, error):
        result["error"] = str(error)
        result["response"] = (
            "Cannot connect to the agent server. "
            "Please ensure the FastAPI backend is running:\n"
            "`python -m uvicorn api.server:app --host 0.0.0.0 --port 8000`"
        )
        done_event.set()
    
    def on_close(ws, close_status_code, close_msg):
        done_event.set()
    
    def on_open(ws):
        payload = json.dumps({"message": message, "session_id": session_id})
        ws.send(payload)
    
    try:
        ws_app = ws_lib.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws_app.run_forever, daemon=True)
        thread.start()
        done_event.wait(timeout=120)
        
    except ImportError:
        result["error"] = "websocket-client not installed"
        result["response"] = (
            "WebSocket client library not found. "
            "Install with: `pip install websocket-client`"
        )
    except Exception as e:
        result["error"] = str(e)
        result["response"] = f"Connection error: {str(e)}"
    
    return result

def maybe_refresh_health(force: bool = False):
    now = time.time()
    if force or now - st.session_state.last_health_check > 5 or st.session_state.last_health_check == -1:
        st.session_state.system_health = fetch_health_sync()
        st.session_state.last_health_check = now

# ──────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────

maybe_refresh_health()

st.markdown("""
<div style="background: linear-gradient(135deg, #f6ede2, #faf4eb); 
     border: 1px solid #e3d7c3; border-radius: 12px; padding: 16px 20px; 
     margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
    <div style="font-size: 28px;">🤖</div>
    <div>
        <div style="font-size: 18px; font-weight: 700; color: #3c342a;">
            Autonomous Customer Support Agent
        </div>
        <div style="font-size: 12px; color: #7e6f5d;">
            LangGraph · mem0 · Ollama · ChromaDB · MCP Tools · Guardrails AI
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([0.65, 0.35], gap="medium")

# ══════════════════════════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════════════════════════
with col_left:
    
    st.markdown('<div class="panel-header">⚡ Live Workflow Visualizer</div>', unsafe_allow_html=True)
    
    pipeline_placeholder = st.empty()
    
    with pipeline_placeholder.container():
        pipeline_html = render_pipeline_nodes(
            st.session_state.node_states,
            st.session_state.node_times,
            st.session_state.escalated_this_turn
        )
        st.markdown(pipeline_html, unsafe_allow_html=True)
    
    st.divider()
    
    header_c1, header_c2 = st.columns([3, 1])
    with header_c1:
        st.markdown('<div class="panel-header">💬 Chat</div>', unsafe_allow_html=True)
    with header_c2:
        if st.button("Export JSON", key="export_btn"):
            export_data = {
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "messages": st.session_state.messages,
                "metrics": st.session_state.metrics,
                "exported_at": datetime.now().isoformat()
            }
            st.download_button(
                "Download",
                data=json.dumps(export_data, indent=2),
                file_name=f"session_{st.session_state.session_id}.json",
                mime="application/json"
            )
    
    st.caption(f"Session ID: `{st.session_state.session_id}` · User: `{st.session_state.user_id}`")
    
    if st.session_state.escalated_this_turn:
        st.markdown("""
        <div class="escalation-banner">
            ⚠️ A human agent has been notified and will contact you shortly.
        </div>
        """, unsafe_allow_html=True)
    
    # Chat messages container
    chat_html = '<div class="chat-wrap">'
    
    if not st.session_state.messages:
        chat_html += '''
        <div style="text-align:center;padding:40px;color:#3a3d52;">
            <div style="font-size:36px;margin-bottom:10px;">👋</div>
            <div style="font-size:14px;">Ask me about your orders, returns, shipping,<br>account questions, or anything else!</div>
            <div style="font-size:11px;margin-top:10px;color:#2a2d42;">
                Try: "What's the status of order ORD-100001?" or "How do I return an item?"
            </div>
        </div>'''
    else:
        for i, msg in enumerate(st.session_state.messages):
            is_last = (i == len(st.session_state.messages) - 1) and (msg.get("role") == "assistant")
            if is_last and not st.session_state.processing:
                meta = msg.get("metadata", {})
                meta_parts = []
                if meta.get("model"):
                    meta_parts.append(f"via {meta['model']}")
                if meta.get("latency_ms"):
                    meta_parts.append(f"{meta['latency_ms']/1000:.1f}s")
                if meta.get("intent"):
                    meta_parts.append(meta["intent"])
                meta_str = " · ".join(meta_parts)
                chat_html += typewriter_html(msg.get("content", ""), meta_str)
            else:
                chat_html += render_chat_message(msg)
        
        # Show Thinking Steps + TTFT for the latest turn
        if not st.session_state.processing and "last_thinking" in st.session_state:
            thinking_html = (
                '<div style="margin:8px 0;padding:8px 12px;background:#f4ede5;'
                'border-radius:8px;border-left:3px solid #c79f67;">'
                '<div style="font-size:10px;font-weight:700;color:#8b5a35;margin-bottom:4px;">'
                '🧠 Agent Reasoning</div>'
            )
            for th in st.session_state["last_thinking"]:
                t_text = th.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
                thinking_html += f'<div style="font-size:11px;color:#5d5141;margin:2px 0;">• {t_text}</div>'
            if "last_ttft_ms" in st.session_state:
                ttft = st.session_state["last_ttft_ms"]
                thinking_html += f'<div style="font-size:10px;color:#8e7f6c;margin-top:4px;">⏱️ TTFT: {ttft}ms</div>'
            thinking_html += '</div>'
            chat_html += thinking_html
    
    if st.session_state.processing:
        chat_html += '''
        <div class="chat-agent">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>'''
    
    chat_html += '</div>'
    
    chat_placeholder = st.empty()
    chat_placeholder.markdown(chat_html, unsafe_allow_html=True)
    
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input(
            "Message",
            key="chat_input",
            placeholder="Ask about orders, returns, shipping, account...",
            label_visibility="collapsed"
        )
    with btn_col:
        send_btn = st.button("Send →", key="send_btn", type="primary", use_container_width=True)
    
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("📦 Order Status", key="q1"):
            st.session_state["_quick_msg"] = "What's the status of order ORD-100002?"
    with qcol2:
        if st.button("↩️ Return Item", key="q2"):
            st.session_state["_quick_msg"] = "How do I return an item I ordered?"
    with qcol3:
        if st.button("💳 Payment Help", key="q3"):
            st.session_state["_quick_msg"] = "Why was my payment declined?"
    with qcol4:
        if st.button("👤 Speak to Human", key="q4"):
            st.session_state["_quick_msg"] = "I need to speak to a human agent please"
    
    message_to_send = None
    
    if (send_btn or user_input and user_input.endswith("\n")) and user_input.strip():
        message_to_send = user_input.strip()
    
    if "_quick_msg" in st.session_state:
        message_to_send = st.session_state.pop("_quick_msg")
    
    if message_to_send and not st.session_state.processing:
        st.session_state.node_states = {n: "pending" for n in PIPELINE_NODES}
        st.session_state.node_times = {}
        st.session_state.escalated_this_turn = False
        st.session_state.processing = True
        
        st.session_state.messages.append({
            "role": "user",
            "content": message_to_send,
            "timestamp": datetime.now().isoformat()
        })
        
        st.session_state.node_states["memory"] = "active"
        st.rerun()
    
    if st.session_state.processing and st.session_state.messages:
        last_user_msg = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "user"),
            None
        )
        
        if last_user_msg and (
            not st.session_state.messages or 
            st.session_state.messages[-1]["role"] != "assistant"
        ):
            result = send_message_sync(
                st.session_state.user_id,
                st.session_state.session_id,
                last_user_msg["content"]
            )
            
            for event in result.get("events", []):
                evt_type = event.get("type")
                evt_node = event.get("node", "")
                elapsed = event.get("elapsed_ms", "")
                
                mapped = NODE_MAP.get(evt_node, evt_node)
                
                if evt_type == "node_active":
                    st.session_state.node_states[mapped] = "active"
                elif evt_type == "node_complete":
                    st.session_state.node_states[mapped] = "complete"
                    if elapsed:
                        st.session_state.node_times[mapped] = elapsed
            
            for n in PIPELINE_NODES:
                if st.session_state.node_states.get(n) != "pending":
                    st.session_state.node_states[n] = "complete"
            
            meta = {
                "intent": result.get("intent", ""),
                "confidence": result.get("confidence", 0.0),
                "model": result.get("model_used", ""),
                "latency_ms": result.get("latency_ms", 0),
                "guardrail_retries": result.get("guardrail_retries", 0),
                "tool_used": result.get("tool_used", ""),
                "escalated": result.get("escalated", False)
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("response", "I couldn't generate a response."),
                "timestamp": datetime.now().isoformat(),
                "metadata": meta
            })
            
            # Persist thinking steps and TTFT
            if result.get("thinking_steps"):
                st.session_state["last_thinking"] = result["thinking_steps"]
            else:
                st.session_state.pop("last_thinking", None)
            
            if result.get("ttft_ms"):
                st.session_state["last_ttft_ms"] = result["ttft_ms"]
            else:
                st.session_state.pop("last_ttft_ms", None)
            
            m = st.session_state.metrics
            m["messages"] += 1
            m["model_used"] = result.get("model_used", "—")
            
            latency = result.get("latency_ms", 0)
            if latency:
                m["response_times"].append(latency)
            
            if result.get("escalated"):
                m["escalations"] += 1
                st.session_state.escalated_this_turn = True
            
            retries = result.get("guardrail_retries", 0)
            m["guardrail_retries"] += retries
            
            intent = result.get("intent", "")
            if intent:
                m["intents"][intent] = m["intents"].get(intent, 0) + 1
            
            st.session_state.memories = fetch_memories_sync(st.session_state.user_id)
            
            st.session_state.processing = False
            st.session_state.last_intent = result.get("intent", "")
            st.session_state.last_tool = result.get("tool_used", "")
            
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════════════════
with col_right:
    
    memories = st.session_state.memories
    if not memories:
        memories = fetch_memories_sync(st.session_state.user_id)
        st.session_state.memories = memories
    
    mem_count = len(memories)
    
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div class="panel-header" style="margin-bottom:0">🧠 What I Know About You</div>
        <span class="memory-badge">{mem_count} memories</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        if not memories:
            st.markdown("""
            <div style="color:#5d5141;font-size:12px;padding:12px;text-align:center;
                 border:1px dashed #e3d7c3;border-radius:8px;">
                No memories yet. Start chatting to build context!
            </div>
            """, unsafe_allow_html=True)
        else:
            for mem in memories[:8]:
                mem_text = mem.get("memory", "")
                mem_time = relative_time(mem.get("created_at", ""))
                st.markdown(f"""
                <div class="memory-card">
                    {mem_text}
                    <div class="memory-time">🕐 {mem_time}</div>
                </div>
                """, unsafe_allow_html=True)
        
        if memories:
            if st.button("🗑️ Clear Memory", key="clear_mem", use_container_width=True):
                if clear_memories_sync(st.session_state.user_id):
                    st.session_state.memories = []
                    st.success("Memory cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear memory")
    
    st.divider()
    
    st.markdown('<div class="panel-header">📊 Agent Metrics</div>', unsafe_allow_html=True)
    
    m = st.session_state.metrics
    rt = m["response_times"]
    avg_rt = round(sum(rt) / len(rt)) if rt else 0
    prev_rt = round(sum(rt[:-1]) / len(rt[:-1])) if len(rt) > 1 else None
    rt_delta = avg_rt - prev_rt if prev_rt else None
    
    st.markdown('<div class="metric-section"><h4>Session Stats</h4>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Messages", m["messages"])
        st.metric("Escalations", m["escalations"])
    with c2:
        st.metric(
            "Avg Response",
            f"{avg_rt/1000:.1f}s",
            delta=f"{rt_delta/1000:+.1f}s" if rt_delta else None,
            delta_color="inverse"
        )
        st.metric("Guard Retries", m["guardrail_retries"])
    
    model_display = m.get("model_used", "—") or "—"
    st.caption(f"Model: `{model_display}`")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-section"><h4>Intent Distribution</h4>', unsafe_allow_html=True)
    
    intents = m.get("intents", {})
    total_msgs = m["messages"] or 1
    intent_order = ["ORDER_QUERY", "GENERAL_FAQ", "COMPLAINT", "ESCALATE"]
    intent_emojis = {"ORDER_QUERY": "📦", "GENERAL_FAQ": "💬", "COMPLAINT": "😤", "ESCALATE": "🚨"}
    
    for intent in intent_order:
        count = intents.get(intent, 0)
        pct = count / total_msgs
        emoji = intent_emojis.get(intent, "")
        label = intent.replace("_", " ").title()
        
        st.markdown(f"""
        <div class="intent-bar-row">
            {emoji} {label} <span style="float:right;color:#6060a0">{count} ({pct:.0%})</span>
        </div>""", unsafe_allow_html=True)
        st.progress(pct)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    health_data = st.session_state.system_health
    services = health_data.get("services", {})
    
    st.markdown('<div class="metric-section"><h4>System Health</h4>', unsafe_allow_html=True)
    
    service_display = [
        ("Ollama", services.get("ollama", {}).get("status", "unknown")),
        ("ChromaDB", services.get("chromadb", {}).get("status", "unknown")),
        ("mem0", services.get("mem0", {}).get("status", "unknown")),
        ("Order DB", services.get("order_db", {}).get("status", "unknown")),
        ("LangSmith", services.get("langsmith", {}).get("status", "not_configured")),
    ]
    
    for svc_name, status in service_display:
        icon = get_health_icon(status)
        status_color = "#34d399" if status == "online" else "#f87171" if status == "offline" else "#f59e0b"
        st.markdown(f"""
        <div class="health-row">
            <span>{icon} {svc_name}</span>
            <span style="color:{status_color};font-weight:600;font-size:11px">{status.upper()}</span>
        </div>""", unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Health", key="refresh_health"):
        maybe_refresh_health(force=True)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # LangSmith Trace
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "support-agent-portfolio")
    
    st.markdown('<div class="metric-section"><h4>LangSmith Tracing</h4>', unsafe_allow_html=True)
    
    is_langsmith_ready = (
        langsmith_key 
        and len(langsmith_key) > 10 
        and "your_langsmith" not in langsmith_key.lower()
    )

    if is_langsmith_ready:
        project_url = "https://smith.langchain.com/o/f5fb86d5-bac4-4506-8ffd-a6ec558287d7/projects/p/47a4e6ed-584d-4847-ac08-fe2890dd958d"
        st.markdown(f"""
        <a href="{project_url}" target="_blank" class="langsmith-link">
            🔍 View Traces → {project}
        </a>""", unsafe_allow_html=True)
        st.caption(f"Key: •••{langsmith_key[-6:]}")
    else:
        st.markdown("""
        <div style="color:#5a6080;font-size:11px;padding:6px">
            Set <code>LANGSMITH_API_KEY</code> in <code>.env</code> to enable tracing
        </div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    if st.button("🆕 New Session", key="new_session", use_container_width=True):
        if st.session_state.session_id not in st.session_state.sessions:
            st.session_state.sessions.append(st.session_state.session_id)
        
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.node_states = {}
        st.session_state.node_times = {}
        st.session_state.processing = False
        st.session_state.escalated_this_turn = False
        st.session_state.metrics = {
            "messages": 0,
            "response_times": [],
            "escalations": 0,
            "guardrail_retries": 0,
            "model_used": "—",
            "intents": {}
        }
        st.session_state.pop("last_thinking", None)
        st.session_state.pop("last_ttft_ms", None)
        st.rerun()