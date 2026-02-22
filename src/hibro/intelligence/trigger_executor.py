#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trigger Executor Module
Executes smart triggers based on query analysis with mixed execution strategy
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.constants import (
    TRIGGER_HIGH_CONFIDENCE,
    TRIGGER_MEDIUM_CONFIDENCE,
    TRIGGER_LOW_CONFIDENCE
)
from .query_analyzer import QueryAnalyzer


class TriggerExecutor:
    """
    Smart trigger executor

    Executes tools based on query analysis with mixed execution strategy:
    - High confidence (>=0.8): Silent execution
    - Medium confidence (0.5-0.8): Return suggestions
    - Low confidence (<0.5): No trigger
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = TRIGGER_HIGH_CONFIDENCE      # 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = TRIGGER_MEDIUM_CONFIDENCE  # 0.5
    LOW_CONFIDENCE_THRESHOLD = TRIGGER_LOW_CONFIDENCE        # 0.3

    def __init__(self, mcp_server):
        """
        Initialize trigger executor

        Args:
            mcp_server: MCP server instance for tool execution
        """
        self.server = mcp_server
        self.logger = logging.getLogger(__name__)

        # Initialize query analyzer
        self._query_analyzer: Optional[QueryAnalyzer] = None

    @property
    def query_analyzer(self) -> QueryAnalyzer:
        """Lazy initialization of query analyzer"""
        if self._query_analyzer is None:
            memory_engine = getattr(self.server, 'memory_engine', None)
            project_scanner = getattr(self.server, 'project_scanner', None)
            self._query_analyzer = QueryAnalyzer(memory_engine, project_scanner)
        return self._query_analyzer

    async def process_query(self, query: str, project_path: str) -> Dict[str, Any]:
        """
        Process user query and decide execution strategy based on confidence

        Args:
            query: User's query text
            project_path: Current project path

        Returns:
            {
                "action": "silent_execute" | "suggest" | "none",
                "executed_tools": list,      # Executed tools
                "execution_results": dict,   # Execution results
                "suggestions": list,         # Suggestions (for medium confidence)
                "analysis": dict,            # Query analysis result
            }
        """
        result = {
            "action": "none",
            "executed_tools": [],
            "execution_results": {},
            "suggestions": [],
            "analysis": {}
        }

        try:
            # Analyze query
            analysis = self.query_analyzer.analyze(query, project_path)
            result["analysis"] = analysis

            confidence = analysis["confidence"]

            if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                # High confidence: silent execution
                result["action"] = "silent_execute"
                self.logger.info(
                    f"High confidence trigger ({confidence:.2f}): "
                    f"{analysis['suggested_tools']}"
                )

                for tool_name in analysis["suggested_tools"]:
                    tool_result = await self._execute_tool(tool_name, project_path)
                    result["executed_tools"].append(tool_name)
                    result["execution_results"][tool_name] = tool_result

            elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                # Medium confidence: return suggestions
                result["action"] = "suggest"
                result["suggestions"] = self._build_suggestions(analysis)
                self.logger.info(
                    f"Medium confidence trigger ({confidence:.2f}): "
                    f"suggesting {analysis['suggested_tools']}"
                )

            # Low confidence: no trigger
            return result

        except Exception as e:
            self.logger.error(f"Trigger processing failed: {e}")
            result["error"] = str(e)
            return result

    async def _execute_tool(self, tool_name: str, project_path: str) -> Dict[str, Any]:
        """
        Execute specified tool

        Args:
            tool_name: Name of tool to execute
            project_path: Project path

        Returns:
            Tool execution result
        """
        # Tool name to method mapping
        tool_map = {
            "get_project_progress": "_tool_get_project_progress",
            "scan_project": "_tool_scan_project",
            "get_quick_context": "_tool_get_quick_context",
            "search_semantic": "_tool_search_semantic",
            "search_memories": "_tool_search_memories",
        }

        method_name = tool_map.get(tool_name)
        if not method_name:
            return {"error": f"Unknown tool: {tool_name}"}

        method = getattr(self.server, method_name, None)
        if not method:
            return {"error": f"Tool method not found: {method_name}"}

        try:
            # Build args based on tool
            args = {"project_path": project_path}

            # Execute tool
            result = await method(args)
            self.logger.info(f"Tool executed: {tool_name}")
            return result

        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name}, {e}")
            return {"error": str(e)}

    def _build_suggestions(self, analysis: Dict) -> List[Dict]:
        """
        Build suggestion messages for medium confidence triggers

        Args:
            analysis: Query analysis result

        Returns:
            List of suggestion dicts
        """
        suggestions = []
        confidence = analysis.get("confidence", 0)

        # Tool descriptions
        tool_descriptions = {
            "get_project_progress": "Get project progress and status",
            "scan_project": "Scan project structure and tech stack",
            "remember": "Store this information to memory",
            "search_semantic": "Search memories semantically",
            "search_memories": "Search memories by keywords",
        }

        for tool_name in analysis.get("suggested_tools", []):
            description = tool_descriptions.get(tool_name, tool_name)
            suggestions.append({
                "tool": tool_name,
                "description": description,
                "confidence": round(confidence, 2),
                "matched_categories": analysis.get("matched_keywords", [])
            })

        return suggestions

    def get_trigger_status(self) -> Dict[str, Any]:
        """
        Get trigger system status

        Returns:
            Status information dict
        """
        return {
            "enabled": True,
            "high_confidence_threshold": self.HIGH_CONFIDENCE_THRESHOLD,
            "medium_confidence_threshold": self.MEDIUM_CONFIDENCE_THRESHOLD,
            "low_confidence_threshold": self.LOW_CONFIDENCE_THRESHOLD,
            "analyzer_ready": self._query_analyzer is not None
        }
