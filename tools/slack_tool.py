"""
slack_tool.py — Slack webhook integration for human escalation.

Sends escalation notifications to the #support-escalations Slack channel
when an agent determines a human agent should take over the conversation.
"""

import os
import json
import logging
import httpx
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def get_webhook_url() -> Optional[str]:
    """Return the Slack webhook URL from environment variables."""
    return os.getenv("SLACK_WEBHOOK_URL")


async def escalate_to_human(
    user_id: str,
    summary: str,
    session_id: Optional[str] = None,
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    conversation_snippet: Optional[str] = None
) -> dict:
    """
    Send an escalation notification to the Slack #support-escalations channel.
    
    Args:
        user_id: The user identifier requiring escalation
        summary: Brief summary of why escalation is needed
        session_id: Current conversation session ID
        intent: The detected intent that triggered escalation
        confidence: Confidence score that fell below threshold
        conversation_snippet: Last few turns of conversation for context
        
    Returns:
        Dictionary with success status and message.
    """
    webhook_url = get_webhook_url()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Build rich Slack message with Block Kit
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 Customer Support Escalation",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*User ID:*\n`{user_id}`"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Session ID:*\n`{session_id or 'N/A'}`"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Timestamp:*\n{timestamp}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence Score:*\n{confidence:.2f if confidence else 'N/A'}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary:*\n{summary}"
            }
        }
    ]
    
    if conversation_snippet:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Last Conversation:*\n```{conversation_snippet[:500]}```"
            }
        })
    
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Take Over Conversation",
                    "emoji": True
                },
                "style": "primary",
                "value": f"takeover_{user_id}_{session_id}"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Full History",
                    "emoji": True
                },
                "value": f"history_{user_id}"
            }
        ]
    })
    
    blocks.append({"type": "divider"})
    
    payload = {
        "text": f"🚨 Escalation needed for user `{user_id}` — {summary[:100]}",
        "blocks": blocks
    }
    
    if not webhook_url or webhook_url == "https://hooks.slack.com/services/YOUR/WEBHOOK/URL":
        logger.warning("Slack webhook URL not configured — logging escalation locally")
        logger.info(f"ESCALATION: user={user_id}, summary={summary}")
        # Return success for demo purposes when webhook not configured
        return {
            "success": True,
            "message": "Escalation logged (Slack webhook not configured — running in demo mode)",
            "demo_mode": True,
            "payload": payload
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Escalation sent to Slack for user {user_id}")
                return {
                    "success": True,
                    "message": "Human agent notified via Slack",
                    "status_code": response.status_code
                }
            else:
                logger.error(f"Slack webhook returned {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "message": f"Failed to notify Slack (status {response.status_code})",
                    "status_code": response.status_code
                }
                
    except httpx.TimeoutException:
        logger.error("Slack webhook timed out")
        return {
            "success": False,
            "message": "Slack notification timed out — escalation logged locally"
        }
    except Exception as e:
        logger.error(f"Slack escalation error: {e}")
        return {
            "success": False,
            "message": f"Escalation error: {str(e)}"
        }


async def send_test_ping() -> bool:
    """
    Send a test ping to verify Slack webhook is working.
    
    Returns:
        True if webhook is reachable and returns 200, False otherwise.
    """
    webhook_url = get_webhook_url()
    
    if not webhook_url or webhook_url == "https://hooks.slack.com/services/YOUR/WEBHOOK/URL":
        logger.warning("Slack webhook not configured")
        return False
    
    try:
        payload = {
            "text": "✅ Support Agent health check ping — webhook is working!"
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(webhook_url, json=payload)
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Slack ping failed: {e}")
        return False
