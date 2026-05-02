"""
demo.py — Automated demo runner for the customer support agent.

Runs 3 showcase conversations sequentially to demonstrate:
  1. Cross-session memory recall
  2. Order lookup via MCP tool → SQLite
  3. Confidence-triggered human escalation

Usage:
    python demo.py
    python demo.py --verbose    # show full responses
    python demo.py --delay 2.0  # seconds between turns

Requires the FastAPI backend to be running on FASTAPI_PORT (default 8000).
"""

import asyncio
import json
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ANSI colors
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

API_BASE = f"http://localhost:{os.getenv('FASTAPI_PORT', '8000')}"
WS_BASE  = f"ws://localhost:{os.getenv('FASTAPI_PORT', '8000')}"

VERBOSE = "--verbose" in sys.argv
DELAY = float(next((sys.argv[sys.argv.index("--delay") + 1]
                    for i, a in enumerate(sys.argv) if a == "--delay"), "1.5"))


def print_header(text: str) -> None:
    """Print a styled section header."""
    print(f"\n{CYAN}{BOLD}{'═' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 60}{RESET}")


def print_user(msg: str) -> None:
    """Print a user message."""
    print(f"\n  {BLUE}👤 User:{RESET} {msg}")


def print_agent(msg: str, meta: Dict = None) -> None:
    """Print an agent response with optional metadata."""
    print(f"\n  {GREEN}🤖 Agent:{RESET} {msg}")
    if meta and VERBOSE:
        print(f"  {DIM}     intent={meta.get('intent')} conf={meta.get('confidence', 0):.2f} "
              f"model={meta.get('model_used')} latency={meta.get('latency_ms')}ms "
              f"retries={meta.get('guardrail_retries')} escalated={meta.get('escalated')}{RESET}")
    elif meta:
        intent = meta.get('intent', '')
        conf = meta.get('confidence', 0)
        model = meta.get('model_used', '')
        ms = meta.get('latency_ms', 0)
        print(f"  {DIM}     [{intent} · conf {conf:.2f} · {model} · {ms}ms]{RESET}")


def print_event(event: Dict) -> None:
    """Print a node lifecycle event if verbose mode is on."""
    if not VERBOSE:
        return
    
    evt_type = event.get("type")
    node = event.get("node", "")
    elapsed = event.get("elapsed_ms", "")
    
    if evt_type == "node_active":
        print(f"  {YELLOW}  ▶ {node}{RESET}", end="", flush=True)
    elif evt_type == "node_complete":
        meta = event.get("metadata", {})
        print(f" → {elapsed}ms ✓{RESET}")


def send_ws_message(user_id: str, session_id: str, message: str) -> Dict:
    """
    Send a message via WebSocket and collect the response.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        message: Message text to send
        
    Returns:
        Result dict with response, intent, confidence, metadata
    """
    try:
        import websocket as ws_lib
    except ImportError:
        print(f"{RED}websocket-client not installed. Run: pip install websocket-client{RESET}")
        return {"response": "WebSocket client not available", "error": True}
    
    result = {
        "response": "",
        "events": [],
        "intent": "",
        "confidence": 0.0,
        "model_used": "",
        "latency_ms": 0,
        "guardrail_retries": 0,
        "tool_used": "",
        "escalated": False,
        "error": None
    }
    done_event = threading.Event()
    
    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
            result["events"].append(msg)
            print_event(msg)
            
            if msg["type"] == "response":
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
            elif msg["type"] == "error":
                result["error"] = msg.get("message")
                result["response"] = f"[Error: {msg.get('message')}]"
                done_event.set()
                ws.close()
        except Exception:
            pass
    
    def on_error(ws, error):
        result["error"] = str(error)
        result["response"] = f"[Connection error: {error}]"
        done_event.set()
    
    def on_open(ws):
        ws.send(json.dumps({"message": message, "session_id": session_id}))
    
    ws_app = ws_lib.WebSocketApp(
        f"{WS_BASE}/ws/{user_id}",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=lambda ws, *a: done_event.set()
    )
    
    t = threading.Thread(target=ws_app.run_forever, daemon=True)
    t.start()
    done_event.wait(timeout=120)
    
    return result


def run_conversation(
    title: str,
    user_id: str,
    session_id: str,
    turns: List[str],
    description: str = ""
) -> List[Dict]:
    """
    Run a multi-turn conversation demo.
    
    Args:
        title: Demo scenario title
        user_id: User identifier for this demo
        session_id: Session identifier
        turns: List of user messages to send
        description: Brief description of what this demo shows
        
    Returns:
        List of result dicts for each turn.
    """
    print_header(f"Demo {title}")
    if description:
        print(f"  {DIM}{description}{RESET}")
    print(f"  {DIM}User: {user_id} · Session: {session_id}{RESET}")
    
    results = []
    
    for i, message in enumerate(turns, 1):
        print(f"\n  {DIM}Turn {i}/{len(turns)}{RESET}")
        print_user(message)
        
        result = send_ws_message(user_id, session_id, message)
        results.append(result)
        
        print_agent(result.get("response", "[no response]"), result)
        
        if result.get("escalated"):
            print(f"\n  {YELLOW}⚡ ESCALATED → Slack notification sent{RESET}")
        
        if result.get("guardrail_retries", 0) > 0:
            print(f"  {YELLOW}🛡️  Guardrail: {result['guardrail_retries']} retry(s){RESET}")
        
        if i < len(turns):
            time.sleep(DELAY)
    
    return results


def check_backend_running() -> bool:
    """Check if the FastAPI backend is reachable."""
    urls = [API_BASE, API_BASE.replace("localhost", "127.0.0.1")]
    for url in urls:
        try:
            response = httpx.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return True
        except Exception:
            continue
    return False


def print_summary(all_results: List[List[Dict]]) -> None:
    """Print aggregate demo statistics."""
    print_header("📊 Demo Summary")
    
    all_flat = [r for demo in all_results for r in demo]
    total = len(all_flat)
    
    if not total:
        print("  No results to summarize")
        return
    
    # Stats
    latencies = [r.get("latency_ms", 0) for r in all_flat if r.get("latency_ms")]
    avg_lat = round(sum(latencies) / len(latencies)) if latencies else 0
    escalations = sum(1 for r in all_flat if r.get("escalated"))
    retries = sum(r.get("guardrail_retries", 0) for r in all_flat)
    intents = {}
    for r in all_flat:
        intent = r.get("intent", "")
        if intent:
            intents[intent] = intents.get(intent, 0) + 1
    
    models = {}
    for r in all_flat:
        m = r.get("model_used", "")
        if m:
            models[m] = models.get(m, 0) + 1
    
    print(f"  Messages sent:      {total}")
    print(f"  Avg LLM latency:    {avg_lat}ms ({avg_lat/1000:.1f}s)")
    print(f"  Escalations:        {escalations}")
    print(f"  Guardrail retries:  {retries}")
    print(f"\n  Intent distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {intent:<15} {bar}  {pct:.0f}%")
    
    print(f"\n  Models used:")
    for model, count in models.items():
        print(f"    {model}: {count} calls")
    
    print(f"\n  {GREEN}{BOLD}✓ Demo complete!{RESET}")
    print(f"  {DIM}View traces at: https://smith.langchain.com{RESET}")


# ──────────────────────────────────────────────────────────────────
# Demo Scenarios
# ──────────────────────────────────────────────────────────────────

def demo_1_memory_recall() -> List[Dict]:
    """Demo 1: Cross-session memory recall."""
    
    # Session 1: Establish user preferences
    session1_results = run_conversation(
        title="1/3 — Cross-Session Memory Recall",
        user_id="demo_user_alice",
        session_id="demo_sess_1a",
        turns=[
            "Hi! I prefer to be contacted by email. Also, I'm a premium member.",
            "What's your return policy for electronics?"
        ],
        description="Session 1: Establishing user preferences in memory"
    )
    
    print(f"\n  {YELLOW}⏱  Simulating new session (memory should persist)...{RESET}")
    time.sleep(2)
    
    # Session 2: Memory should be recalled
    session2_results = run_conversation(
        title="1/3 (continued) — New Session, Same User",
        user_id="demo_user_alice",
        session_id="demo_sess_1b",  # Different session ID
        turns=[
            "I have a question about my shipment. Do you remember my preferences?",
            "How long does an exchange take?"
        ],
        description="Session 2: Agent should recall user prefers email and is a premium member"
    )
    
    return session1_results + session2_results


def demo_2_order_lookup() -> List[Dict]:
    """Demo 2: Order status lookup via MCP tool."""
    return run_conversation(
        title="2/3 — Order Status via MCP Tool",
        user_id="demo_user_bob",
        session_id="demo_sess_2",
        turns=[
            "What is the status of order ORD-100002?",
            "Will it arrive before the weekend?",
            "Can I change the shipping address for ORD-100002?",
            "What about ORD-100016? I heard it might be delayed."
        ],
        description="Demonstrates MCP get_order_status tool → SQLite lookup with order ORD-100002 (SHIPPED)"
    )


def demo_3_escalation() -> List[Dict]:
    """Demo 3: Confidence-triggered escalation."""
    return run_conversation(
        title="3/3 — Human Escalation",
        user_id="demo_user_carol",
        session_id="demo_sess_3",
        turns=[
            "I'm extremely frustrated. My order ORD-100009 was supposed to arrive last week and I still haven't received it!",
            "I've been waiting 3 weeks! This is absolutely unacceptable! I demand a refund immediately!",
            "I want to speak to a manager right now. This bot is useless!"
        ],
        description="Demonstrates complaint detection → tone softening → escalation trigger via Slack MCP"
    )


def main() -> None:
    """Run all three demo conversations sequentially."""
    
    print(f"\n{BOLD}{CYAN}{'═' * 60}")
    print("  🤖 Customer Support Agent — Live Demo")
    print(f"  {DIM}Showcasing: memory · MCP tools · escalation · guardrails{RESET}")
    print(f"{CYAN}{'═' * 60}{RESET}")
    print(f"\n  Backend: {API_BASE}")
    print(f"  Verbose: {VERBOSE}")
    print(f"  Delay:   {DELAY}s between turns")
    
    # Check backend
    print(f"\n  {DIM}Checking backend...{RESET}", end="", flush=True)
    if not check_backend_running():
        print(f" {RED}OFFLINE{RESET}")
        print(f"\n  {RED}✗ FastAPI backend is not running!{RESET}")
        print(f"\n  Start it first:")
        print(f"  {CYAN}python -m uvicorn api.server:app --host 0.0.0.0 --port 8000{RESET}")
        sys.exit(1)
    print(f" {GREEN}ONLINE ✓{RESET}")
    
    all_results = []
    
    try:
        # Demo 1: Memory
        results_1 = demo_1_memory_recall()
        all_results.append(results_1)
        time.sleep(DELAY * 2)
        
        # Demo 2: Order lookup
        results_2 = demo_2_order_lookup()
        all_results.append(results_2)
        time.sleep(DELAY * 2)
        
        # Demo 3: Escalation
        results_3 = demo_3_escalation()
        all_results.append(results_3)
        
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Demo interrupted by user{RESET}")
    
    # Summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
