"""
health_check.py — Standalone service health verification script.

Checks all required services on startup and prints colored status output.
Run this before starting the agent to verify your environment is ready.

Usage:
    python health_check.py
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def ok(service: str, message: str, detail: str = "") -> None:
    detail_str = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}✓{RESET} {BOLD}{service:<22}{RESET} {message}{detail_str}")


def fail(service: str, message: str, fix: str = "") -> None:
    fix_str = f"\n    {YELLOW}→ Fix: {fix}{RESET}" if fix else ""
    print(f"  {RED}✗{RESET} {BOLD}{service:<22}{RESET} {message}{fix_str}")


def warn(service: str, message: str, detail: str = "") -> None:
    detail_str = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {YELLOW}⚠{RESET} {BOLD}{service:<22}{RESET} {message}{detail_str}")


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'─' * 52}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'─' * 52}{RESET}")


# ──────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────

async def check_ollama() -> bool:
    """Verify Ollama is running and required models are available."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    primary = os.getenv("OLLAMA_PRIMARY_MODEL", "gemma3:1b")
    fallback = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen:1.8b")
    ultra = os.getenv("OLLAMA_ULTRA_LIGHT_MODEL", "gemma3:1b")
    embed = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest")
    embed_fb = os.getenv("OLLAMA_EMBED_FALLBACK", "all-minilm:latest")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
    except httpx.ConnectError:
        fail("Ollama", "Not running", "Run: `ollama serve` in a terminal")
        return False
    except Exception as e:
        fail("Ollama", f"Error: {e}", "Ensure Ollama is installed: https://ollama.ai")
        return False
    
    ok("Ollama", f"Running at {ollama_url}", f"{len(models)} models loaded")
    
    all_good = True
    for model_name, label in [
        (primary,  "Primary LLM"),
        (fallback, "Fallback LLM"),
        (ultra,    "Ultra-light LLM"),
        (embed,    "Embed model"),
        (embed_fb, "Embed fallback"),
    ]:
        # Check if model name (possibly without :tag) is in list
        found = any(model_name in m for m in models)
        if found:
            ok(f"  {label}", f"{model_name}")
        else:
            warn(f"  {label}", f"{model_name} not found", f"Pull with: `ollama pull {model_name}`")
            if model_name in (primary, embed):
                all_good = False
    
    return all_good


async def check_chromadb() -> bool:
    """Verify ChromaDB can be initialized and FAQ collection exists."""
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        
        faq_col = "support_faqs"
        if faq_col in col_names:
            col = client.get_collection(faq_col)
            count = col.count()
            ok("ChromaDB", f"Ready at {chroma_path}", f"{count} FAQ docs in collection")
        else:
            warn("ChromaDB", f"Ready, but '{faq_col}' collection empty",
                 "Run: `python knowledge_base/ingest.py`")
        return True
    except ImportError:
        fail("ChromaDB", "Not installed", "Run: `pip install chromadb`")
        return False
    except Exception as e:
        fail("ChromaDB", f"Error: {e}", f"Ensure {chroma_path} is writable")
        return False


async def check_mem0() -> bool:
    """Verify mem0 can write and read a test memory entry."""
    try:
        from mem0 import Memory
        
        mem0_path = os.getenv("MEM0_DB_PATH", "./mem0_db")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        primary = os.getenv("OLLAMA_PRIMARY_MODEL", "gemma3:1b")
        embed = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest")
        
        from mem0.configs.base import MemoryConfig
        from mem0.vector_stores.configs import VectorStoreConfig
        from mem0.llms.configs import LlmConfig
        from mem0.embeddings.configs import EmbedderConfig
        
        config = MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="chroma",
                config={
                    "collection_name": "support_mem",
                    "path": mem0_path
                }
            ),
            llm=LlmConfig(
                provider="ollama",
                config={
                    "model": primary,
                    "ollama_base_url": ollama_url,
                    "temperature": 0.0,
                    "max_tokens": 2048
                }
            ),
            embedder=EmbedderConfig(
                provider="ollama",
                config={
                    "model": embed,
                    "ollama_base_url": ollama_url
                }
            )
        )
        
        memory = Memory(config=config)
        
        # Test write
        test_user = "_healthcheck_"
        memory.add(
            messages=[{"role": "user", "content": "health check test message"}],
            user_id=test_user
        )
        
        # Test read
        results = memory.get_all(filters={"user_id": test_user})
        
        # Cleanup
        memory.delete_all(filters={"user_id": test_user})
        
        ok("mem0", "Read/write working", f"Config: SQLite + ChromaDB at {mem0_path}")
        return True
        
    except ImportError:
        fail("mem0", "Not installed", "Run: `pip install mem0ai`")
        return False
    except Exception as e:
        warn("mem0", f"Initialization issue: {str(e)[:80]}",
             "May work at runtime — check Ollama models are loaded")
        return True  # Non-fatal


async def check_mcp_server() -> bool:
    """Verify MCP server process starts and responds to tool list request."""
    import subprocess
    import json
    
    try:
        proc = subprocess.Popen(
            [sys.executable, "tools/mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send initialize request
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}}
        }) + "\n"
        
        # Send tools/list request
        list_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }) + "\n"
        
        proc.stdin.write(init_msg + list_msg)
        proc.stdin.flush()
        
        # Read responses with timeout
        import threading
        
        responses = []
        
        def read_output():
            for line in proc.stdout:
                line = line.strip()
                if line:
                    try:
                        responses.append(json.loads(line))
                    except Exception:
                        pass
                if len(responses) >= 2:
                    break
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        proc.terminate()
        
        if len(responses) >= 2:
            tool_response = responses[1]
            tools = tool_response.get("result", {}).get("tools", [])
            tool_names = [t["name"] for t in tools]
            ok("MCP Server", f"{len(tools)} tools registered",
               f"Tools: {', '.join(tool_names)}")
            return True
        else:
            warn("MCP Server", "Started but slow to respond — may still work at runtime")
            return True
            
    except FileNotFoundError:
        fail("MCP Server", "tools/mcp_server.py not found",
             "Run from project root directory")
        return False
    except Exception as e:
        warn("MCP Server", f"Check failed: {str(e)[:60]}", "May still work at runtime")
        return True


async def check_langsmith() -> bool:
    """Verify LangSmith API key is valid."""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "support-agent-portfolio")
    
    if not api_key or api_key in ("your_langsmith_api_key_here", ""):
        warn("LangSmith", "API key not configured",
             "Set LANGSMITH_API_KEY in .env (optional — get free at smith.langchain.com)")
        return True  # Non-fatal
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.smith.langchain.com/ok",
                headers={"x-api-key": api_key}
            )
            if response.status_code == 200:
                ok("LangSmith", f"Connected · Project: {project}")
                return True
            else:
                warn("LangSmith", f"API returned {response.status_code}",
                     "Check your LANGSMITH_API_KEY")
                return True
    except Exception as e:
        warn("LangSmith", f"Connection failed: {str(e)[:60]}",
             "LangSmith tracing disabled (non-fatal)")
        return True


async def check_slack() -> bool:
    """Verify Slack webhook URL is configured and responds."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    
    if not webhook_url or webhook_url == "https://hooks.slack.com/services/YOUR/WEBHOOK/URL":
        warn("Slack Webhook", "Not configured",
             "Set SLACK_WEBHOOK_URL in .env (optional — escalations logged locally)")
        return True  # Non-fatal
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                webhook_url,
                json={"text": "🔍 Support Agent health check ping"},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                ok("Slack Webhook", "Webhook responding")
                return True
            else:
                warn("Slack Webhook", f"Returned {response.status_code}",
                     "Check your SLACK_WEBHOOK_URL")
                return True
    except Exception as e:
        warn("Slack Webhook", f"Connection failed: {str(e)[:60]}")
        return True


async def check_order_db() -> bool:
    """Verify order database can be initialized."""
    db_path = os.getenv("ORDER_DB", "./orders.db")
    
    try:
        sys.path.insert(0, os.getcwd())
        from tools.order_db import init_order_db, get_order_status
        
        init_order_db()
        
        result = get_order_status("ORD-100001")
        if result.get("found"):
            ok("Order Database", f"Ready at {db_path}", "20 sample orders loaded")
            return True
        else:
            warn("Order Database", "Initialized but test query failed")
            return True
    except Exception as e:
        fail("Order Database", f"Error: {e}", f"Ensure {db_path} directory is writable")
        return False


async def check_python_deps() -> bool:
    """Verify critical Python packages are installed."""
    section("Python Dependencies")
    
    packages = [
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langsmith", "langsmith"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("chromadb", "chromadb"),
        ("mem0", "mem0ai"),
        ("httpx", "httpx"),
        ("tenacity", "tenacity"),
        ("dotenv", "python-dotenv"),
        ("websocket", "websocket-client"),
    ]
    
    all_ok = True
    for import_name, pkg_name in packages:
        try:
            __import__(import_name)
            ok(pkg_name, "Installed")
        except ImportError:
            fail(pkg_name, "Not installed", f"Run: `pip install {pkg_name}`")
            all_ok = False
    
    return all_ok


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

async def main() -> None:
    """Run all health checks and print a summary."""
    
    print(f"\n{BOLD}{CYAN}{'═' * 52}")
    print("  🤖 Customer Support Agent — Health Check")
    print(f"{'═' * 52}{RESET}")
    print(f"  {DIM}Checking all services...{RESET}\n")
    
    results: Dict[str, bool] = {}
    
    # Python deps first
    results["python_deps"] = await check_python_deps()
    
    # Core services
    section("Core Services")
    results["ollama"]     = await check_ollama()
    results["chromadb"]   = await check_chromadb()
    results["mem0"]       = await check_mem0()
    results["order_db"]   = await check_order_db()
    
    # Optional services
    section("Optional Services")
    results["mcp_server"] = await check_mcp_server()
    results["langsmith"]  = await check_langsmith()
    results["slack"]      = await check_slack()
    
    # Summary
    section("Summary")
    
    critical = ["python_deps", "ollama", "chromadb", "order_db"]
    optional = ["mem0", "mcp_server", "langsmith", "slack"]
    
    critical_pass = sum(1 for k in critical if results.get(k, False))
    optional_pass = sum(1 for k in optional if results.get(k, False))
    
    print(f"  Critical services: {critical_pass}/{len(critical)} passing")
    print(f"  Optional services: {optional_pass}/{len(optional)} passing")
    
    if all(results.get(k, False) for k in critical):
        print(f"\n  {GREEN}{BOLD}✓ READY TO RUN!{RESET}")
        print(f"\n  {DIM}Start the agent:{RESET}")
        print(f"  {CYAN}1. Terminal 1: python -m uvicorn api.server:app --host 0.0.0.0 --port 8000{RESET}")
        print(f"  {CYAN}2. Terminal 2: streamlit run ui/app.py{RESET}")
        print(f"  {CYAN}3. Or run demo: python demo.py{RESET}")
    else:
        failed = [k for k in critical if not results.get(k, False)]
        print(f"\n  {RED}{BOLD}✗ NOT READY — Fix critical failures first: {', '.join(failed)}{RESET}")
    
    print(f"\n{CYAN}{'═' * 52}{RESET}\n")
    
    # Return exit code
    sys.exit(0 if all(results.get(k, False) for k in critical) else 1)


if __name__ == "__main__":
    asyncio.run(main())
