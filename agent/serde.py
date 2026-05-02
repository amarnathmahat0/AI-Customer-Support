"""
Custom serializer for LangGraph state that excludes non-serializable fields.

This module provides a custom serializer that filters out transient fields
like asyncio.Queue before checkpointing, allowing AsyncSqliteSaver to
properly serialize the agent state.
"""

import asyncio
import logging
from typing import Any, Dict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class FilteredJsonPlusSerializer(JsonPlusSerializer):
    """
    Custom serializer that filters out non-serializable fields before encoding.
    
    Transient fields like event_queue, start_time, node_times, and langsmith_run_id
    are set per-request and should not be persisted to the checkpoint.
    """
    
    EXCLUDED_FIELDS = {
        "event_queue",      # asyncio.Queue - not serializable
        "start_time",       # float - temporary, recomputed per request
        "node_times",       # Dict - temporary execution tracking
        "langsmith_run_id"  # Optional[str] - temporary trace ID
    }
    
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """
        Serialize an object, filtering out non-serializable transient fields.
        
        Args:
            obj: The object to serialize (typically the agent state dict)
            
        Returns:
            Tuple of (type, serialized_bytes)
        """
        # Deep filter the object to remove non-serializable fields
        filtered_obj = self._deep_filter(obj)
        logger.debug(f"Filtered state for serialization: excluded {self.EXCLUDED_FIELDS}")
        return super().dumps_typed(filtered_obj)
    
    def _deep_filter(self, obj: Any) -> Any:
        """
        Recursively filter out non-serializable fields from nested structures.
        """
        if isinstance(obj, dict):
            return {
                k: self._deep_filter(v) 
                for k, v in obj.items() 
                if k not in self.EXCLUDED_FIELDS
            }
        elif isinstance(obj, list):
            return [self._deep_filter(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._deep_filter(item) for item in obj)
        elif isinstance(obj, set):
            return {self._deep_filter(item) for item in obj}
        else:
            return obj
        
        return super().dumps_typed(obj)
