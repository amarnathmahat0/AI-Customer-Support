"""
mcp_server.py — MCP (Model Context Protocol) stdio server.

Exposes two tools to the LangGraph agent:
  1. get_order_status(order_id) — queries SQLite mock DB
  2. escalate_to_human(user_id, summary) — sends Slack webhook notification

Transport: stdio (standard input/output)
"""

import asyncio
import json
import logging
import sys
import os
import re

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.order_db import get_order_status as _get_order_status, init_order_db
from tools.slack_tool import escalate_to_human as _escalate_to_human

logging.basicConfig(
    level=logging.WARNING,  # Keep quiet on stdio
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("mcp_server.log")]
)
logger = logging.getLogger(__name__)

# MCP Protocol Constants
JSONRPC_VERSION = "2.0"
MCP_VERSION = "2024-11-05"

ORDER_ID_PATTERN = re.compile(r'^ORD-\d{6}$')


def make_response(request_id, result) -> dict:
    """Create a JSON-RPC success response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result
    }


def make_error(request_id, code: int, message: str) -> dict:
    """Create a JSON-RPC error response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message}
    }


def send_response(response: dict) -> None:
    """Write a JSON-RPC response to stdout."""
    line = json.dumps(response) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


async def handle_initialize(request_id, params: dict) -> None:
    """Handle MCP initialize handshake."""
    send_response(make_response(request_id, {
        "protocolVersion": MCP_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "support-agent-mcp",
            "version": "1.0.0"
        }
    }))


async def handle_tools_list(request_id) -> None:
    """Return the list of available tools with JSON schemas."""
    tools = [
        {
            "name": "get_order_status",
            "description": (
                "Retrieve the current status and details of a customer order "
                "from the order management system. Returns order status, "
                "estimated delivery, tracking information, and product details."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID in format ORD-XXXXXX (e.g., ORD-100001)",
                        "pattern": "^ORD-[0-9]{6}$"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        },
        {
            "name": "escalate_to_human",
            "description": (
                "Escalate the current conversation to a human support agent. "
                "Sends a notification to the #support-escalations Slack channel "
                "with conversation summary. Use when: confidence is low, "
                "user explicitly requests human, or issue is complex."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user identifier for the conversation being escalated"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of why escalation is needed and what the user needs",
                        "maxLength": 500
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Current session ID for context (optional)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Classifier confidence score that triggered escalation (optional)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["user_id", "summary"],
                "additionalProperties": False
            }
        }
    ]
    
    send_response(make_response(request_id, {"tools": tools}))


async def handle_tools_call(request_id, params: dict) -> None:
    """Handle a tool invocation request."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    
    logger.info(f"Tool call: {tool_name} with args {arguments}")
    
    try:
        if tool_name == "get_order_status":
            order_id = arguments.get("order_id", "")
            
            # Validate order ID format
            if not ORDER_ID_PATTERN.match(order_id):
                send_response(make_response(request_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": f"Invalid order ID format: '{order_id}'. Expected format: ORD-XXXXXX",
                            "found": False
                        })
                    }],
                    "isError": True
                }))
                return
            
            # Query database
            result = _get_order_status(order_id)
            
            send_response(make_response(request_id, {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }],
                "isError": not result.get("found", False)
            }))
            
        elif tool_name == "escalate_to_human":
            user_id = arguments.get("user_id", "")
            summary = arguments.get("summary", "")
            session_id = arguments.get("session_id")
            confidence = arguments.get("confidence")
            
            if not user_id or not summary:
                send_response(make_error(
                    request_id, -32602,
                    "Missing required parameters: user_id and summary"
                ))
                return
            
            # Truncate summary if too long
            summary = summary[:500]
            
            result = await _escalate_to_human(
                user_id=user_id,
                summary=summary,
                session_id=session_id,
                confidence=confidence
            )
            
            send_response(make_response(request_id, {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }],
                "isError": not result.get("success", False)
            }))
            
        else:
            send_response(make_error(
                request_id, -32601,
                f"Unknown tool: {tool_name}"
            ))
            
    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {e}")
        send_response(make_response(request_id, {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": f"Tool execution failed: {str(e)}"})
            }],
            "isError": True
        }))


async def process_message(line: str) -> None:
    """Parse and dispatch a single JSON-RPC message."""
    try:
        message = json.loads(line.strip())
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        send_response(make_error(None, -32700, "Parse error"))
        return
    
    request_id = message.get("id")
    method = message.get("method", "")
    params = message.get("params", {})
    
    logger.debug(f"Received method: {method}")
    
    if method == "initialize":
        await handle_initialize(request_id, params)
    elif method == "initialized":
        pass  # Notification, no response needed
    elif method == "tools/list":
        await handle_tools_list(request_id)
    elif method == "tools/call":
        await handle_tools_call(request_id, params)
    elif method == "ping":
        send_response(make_response(request_id, {}))
    else:
        if request_id is not None:
            send_response(make_error(request_id, -32601, f"Method not found: {method}"))


async def main() -> None:
    """
    Main event loop for the MCP stdio server.
    
    Reads JSON-RPC messages from stdin line by line and dispatches them
    to the appropriate handler functions.
    """
    logger.info("MCP server starting on stdio transport")
    
    # Initialize the order database on startup
    try:
        init_order_db()
        logger.info("Order database ready")
    except Exception as e:
        logger.error(f"Failed to initialize order DB: {e}")
    
    # Read from stdin line by line
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    
    logger.info("MCP server ready — waiting for messages")
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                logger.info("Stdin closed — shutting down")
                break
            
            line_str = line.decode("utf-8").strip()
            if line_str:
                await process_message(line_str)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
