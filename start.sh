#!/bin/bash
# start.sh — Convenience script to start all components

set -e

GREEN='\033[0;32m'
CYAN='\033[0;96m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  🤖  Customer Support Agent — Startup Script     ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# Load env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "  ${GREEN}✓${RESET} Loaded .env"
else
    echo -e "  ${YELLOW}⚠${RESET} No .env found — using defaults (copy .env.example to .env)"
fi

PORT=${FASTAPI_PORT:-8000}

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "  ${GREEN}✓${RESET} Virtual environment activated"
fi

# Check Ollama
echo -e "\n  ${CYAN}Checking Ollama...${RESET}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "  ${YELLOW}Starting Ollama in background...${RESET}"
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi
echo -e "  ${GREEN}✓${RESET} Ollama running"

# Initialize databases
echo -e "\n  ${CYAN}Initializing databases...${RESET}"
python -c "from tools.order_db import init_order_db; init_order_db()" 2>/dev/null && \
    echo -e "  ${GREEN}✓${RESET} Order database ready"

python knowledge_base/ingest.py 2>/dev/null && \
    echo -e "  ${GREEN}✓${RESET} FAQ knowledge base ready"

# Start FastAPI
echo -e "\n  ${CYAN}Starting FastAPI backend on port ${PORT}...${RESET}"
python -m uvicorn api.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level warning &
FASTAPI_PID=$!
sleep 2

if kill -0 $FASTAPI_PID 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} FastAPI running at http://localhost:${PORT}"
else
    echo -e "  ${RED}✗${RESET} FastAPI failed to start — check logs"
    exit 1
fi

# Start Streamlit
echo -e "\n  ${CYAN}Starting Streamlit dashboard...${RESET}"
echo -e "  ${GREEN}✓${RESET} Opening http://localhost:8501"
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop all services${RESET}"
echo ""

# Run streamlit in foreground
FASTAPI_URL="http://localhost:${PORT}" \
streamlit run ui/app.py \
    --server.port 8501 \
    --server.headless true \
    --theme.base dark

# Cleanup on exit
trap "kill $FASTAPI_PID 2>/dev/null; echo 'Services stopped.'" EXIT
