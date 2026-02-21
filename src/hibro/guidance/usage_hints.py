#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage Hint Provider
Provides context-relevant tool usage hints and guidance at appropriate times
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..utils.config import Config


class HintTrigger(Enum):
    """Hint trigger condition"""
    FIRST_USE = "first_use"              # First use
    ERROR_OCCURRED = "error_occurred"    # Error occurred
    INEFFICIENT_USE = "inefficient_use"  # Inefficient use
    MISSED_OPPORTUNITY = "missed_opportunity"  # Missed opportunity
    BEST_PRACTICE = "best_practice"      # Best practice
    FEATURE_DISCOVERY = "feature_discovery"  # Feature discovery


class HintType(Enum):
    """Hint type"""
    USAGE_TIP = "usage_tip"              # Usage tip
    PARAMETER_HINT = "parameter_hint"    # Parameter hint
    WORKFLOW_SUGGESTION = "workflow_suggestion"  # Workflow suggestion
    ERROR_PREVENTION = "error_prevention"  # Error prevention
    OPTIMIZATION = "optimization"        # Optimization suggestion
    FEATURE_HIGHLIGHT = "feature_highlight"  # Feature highlight


@dataclass
class UsageHint:
    """Usage hint"""
    hint_id: str
    hint_type: HintType
    trigger: HintTrigger
    tool_name: str
    title: str
    message: str
    example: Optional[str] = None
    priority: int = 5  # 1-10, 10 is highest
    show_once: bool = True  # Whether to show only once
    conditions: Dict[str, Any] = None  # Display conditions


class UsageHintProvider:
    """Usage hint provider"""

    def __init__(self, config: Config):
        """
        Initialize usage hint provider

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.usage_hint_provider')

        # Hint definitions
        self.hint_definitions = self._define_usage_hints()

        # User hint history
        self.user_hint_history: Dict[str, Set[str]] = {}  # session_id -> shown_hint_ids

        # Tool usage statistics
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}

        self.logger.info("Usage hint provider initialized")

    def _define_usage_hints(self) -> Dict[str, UsageHint]:
        """Define usage hints"""
        hints = {}

        # Core tool hints
        hints["get_quick_context_first_use"] = UsageHint(
            hint_id="get_quick_context_first_use",
            hint_type=HintType.USAGE_TIP,
            trigger=HintTrigger.FIRST_USE,
            tool_name="get_quick_context",
            title="Best Practice for Session Start",
            message="It's recommended to call this tool at the start of each new session to get personalized user context information.",
            example="Call before starting programming tasks to ensure understanding of user's code style preferences.",
            priority=10
        )

        hints["get_preferences_before_coding"] = UsageHint(
            hint_id="get_preferences_before_coding",
            hint_type=HintType.WORKFLOW_SUGGESTION,
            trigger=HintTrigger.BEST_PRACTICE,
            tool_name="get_preferences",
            title="Get Preferences Before Coding",
            message="Call this tool before writing or modifying code to ensure the generated code matches your programming habits.",
            example="Get code comment language preference to ensure comments are in English.",
            priority=9
        )

        # Analysis tool hints
        hints["analyze_vs_question"] = UsageHint(
            hint_id="analyze_vs_question",
            hint_type=HintType.FEATURE_HIGHLIGHT,
            trigger=HintTrigger.FEATURE_DISCOVERY,
            tool_name="analyze_project_deeply",
            title="Deep Analysis vs Quick Q&A",
            message="analyze_project_deeply is suitable for comprehensive analysis, answer_specific_question is suitable for quick answers to specific questions.",
            example="Use deep analysis for project decisions, use quick Q&A for error diagnosis.",
            priority=8
        )

        hints["question_context_hint"] = UsageHint(
            hint_id="question_context_hint",
            hint_type=HintType.PARAMETER_HINT,
            trigger=HintTrigger.INEFFICIENT_USE,
            tool_name="answer_specific_question",
            title="Provide Context Information",
            message="Provide relevant background information in the context parameter to get more accurate analysis results.",
            example="Describe the scenario where the error occurred, related code, or operation steps.",
            priority=7
        )

        # Intelligent assistant tool hints
        hints["suggestions_project_path"] = UsageHint(
            hint_id="suggestions_project_path",
            hint_type=HintType.PARAMETER_HINT,
            trigger=HintTrigger.INEFFICIENT_USE,
            tool_name="get_smart_suggestions",
            title="Specify Project Path for Better Recommendations",
            message="Providing the project_path parameter can get more precise project-related recommendations.",
            example="After specifying the project path, recommendations will be optimized based on project type and structure.",
            priority=8
        )

        hints["comprehensive_vs_smart"] = UsageHint(
            hint_id="comprehensive_vs_smart",
            hint_type=HintType.FEATURE_HIGHLIGHT,
            trigger=HintTrigger.FEATURE_DISCOVERY,
            tool_name="get_comprehensive_assistance",
            title="Comprehensive Assistance vs Smart Suggestions",
            message="get_comprehensive_assistance provides all-around assistance, get_smart_suggestions focuses on recommendations.",
            example="Use comprehensive assistance for complex problems, use smart suggestions for daily optimization.",
            priority=7
        )

        # Security tool hints
        hints["security_regular_check"] = UsageHint(
            hint_id="security_regular_check",
            hint_type=HintType.WORKFLOW_SUGGESTION,
            trigger=HintTrigger.MISSED_OPPORTUNITY,
            tool_name="check_security_status",
            title="Regular Security Checks",
            message="It's recommended to regularly check system security status to promptly detect and handle security risks.",
            example="Check security status weekly or after important changes.",
            priority=6
        )

        hints["backup_before_changes"] = UsageHint(
            hint_id="backup_before_changes",
            hint_type=HintType.ERROR_PREVENTION,
            trigger=HintTrigger.BEST_PRACTICE,
            tool_name="create_backup",
            title="Backup Before Important Changes",
            message="It's recommended to create a backup before making important system changes or configuration modifications.",
            example="Create backup before system upgrades, configuration changes, or important data modifications.",
            priority=9
        )

        # Search tool hints
        hints["semantic_vs_keyword"] = UsageHint(
            hint_id="semantic_vs_keyword",
            hint_type=HintType.FEATURE_HIGHLIGHT,
            trigger=HintTrigger.FEATURE_DISCOVERY,
            tool_name="search_semantic",
            title="Semantic Search vs Keyword Search",
            message="search_semantic is suitable for conceptual searching, search_memories is suitable for precise keyword searching.",
            example="Use semantic search to find related concepts, use keyword search to find specific terms.",
            priority=7
        )

        # Workflow tool hints
        hints["workflow_automation"] = UsageHint(
            hint_id="workflow_automation",
            hint_type=HintType.OPTIMIZATION,
            trigger=HintTrigger.MISSED_OPPORTUNITY,
            tool_name="detect_workflow_patterns",
            title="Discover Automation Opportunities",
            message="Regularly use this tool to detect repetitive workflows and discover automation optimization opportunities.",
            example="Detect repetitive command sequences and operation patterns during project development.",
            priority=6
        )

        # Parameter optimization hints
        hints["analysis_depth_control"] = UsageHint(
            hint_id="analysis_depth_control",
            hint_type=HintType.PARAMETER_HINT,
            trigger=HintTrigger.BEST_PRACTICE,
            tool_name="analyze_project_deeply",
            title="Control Analysis Depth",
            message="Use the analysis_depth parameter to control analysis detail level: quick (fast), standard (standard), deep (deep).",
            example="Use quick for daily checks, use deep for important decisions.",
            priority=6
        )

        hints["backup_priority_setting"] = UsageHint(
            hint_id="backup_priority_setting",
            hint_type=HintType.PARAMETER_HINT,
            trigger=HintTrigger.BEST_PRACTICE,
            tool_name="create_backup",
            title="Set Backup Priority",
            message="Use the priority parameter to set backup priority, urgent priority backups will be processed first.",
            example="Set to urgent in emergencies, use normal for daily backups.",
            priority=5
        )

        return hints

    def get_contextual_hints(self, session_id: str, context: Dict[str, Any]) -> List[UsageHint]:
        """
        Get context-relevant usage hints

        Args:
            session_id: Session ID
            context: Context information

        Returns:
            List[UsageHint]: List of hints
        """
        hints = []
        current_tool = context.get("current_tool")
        user_level = context.get("user_level", "beginner")
        recent_tools = context.get("recent_tools", [])
        error_occurred = context.get("error_occurred", False)

        # Initialize user hint history
        if session_id not in self.user_hint_history:
            self.user_hint_history[session_id] = set()

        shown_hints = self.user_hint_history[session_id]

        # 1. First use hints
        if current_tool:
            first_use_hint_id = f"{current_tool}_first_use"
            if (first_use_hint_id in self.hint_definitions and
                first_use_hint_id not in shown_hints and
                current_tool not in recent_tools):
                hints.append(self.hint_definitions[first_use_hint_id])

        # 2. Best practice hints
        hints.extend(self._get_best_practice_hints(context, shown_hints))

        # 3. Feature discovery hints
        hints.extend(self._get_feature_discovery_hints(context, shown_hints, user_level))

        # 4. Parameter optimization hints
        hints.extend(self._get_parameter_hints(context, shown_hints))

        # 5. Error prevention hints
        if error_occurred:
            hints.extend(self._get_error_prevention_hints(context, shown_hints))

        # 6. Workflow optimization hints
        hints.extend(self._get_workflow_hints(context, shown_hints))

        # Sort by priority and limit quantity
        hints.sort(key=lambda x: x.priority, reverse=True)
        return hints[:3]  # Return at most 3 hints

    def _get_best_practice_hints(self, context: Dict[str, Any], shown_hints: Set[str]) -> List[UsageHint]:
        """Get best practice hints"""
        hints = []
        current_tool = context.get("current_tool")
        recent_tools = context.get("recent_tools", [])

        # Get preferences before coding hint
        if (current_tool in ["remember", "search_memories"] and
            "get_preferences" not in recent_tools and
            "get_preferences_before_coding" not in shown_hints):
            hints.append(self.hint_definitions["get_preferences_before_coding"])

        # Security check hint
        if (len(recent_tools) > 5 and
            "check_security_status" not in recent_tools and
            "security_regular_check" not in shown_hints):
            hints.append(self.hint_definitions["security_regular_check"])

        # Backup hint
        if (current_tool in ["apply_security_policy", "set_project_context"] and
            "create_backup" not in recent_tools and
            "backup_before_changes" not in shown_hints):
            hints.append(self.hint_definitions["backup_before_changes"])

        return hints

    def _get_feature_discovery_hints(self, context: Dict[str, Any], shown_hints: Set[str],
                                   user_level: str) -> List[UsageHint]:
        """Get feature discovery hints"""
        hints = []
        current_tool = context.get("current_tool")
        recent_tools = context.get("recent_tools", [])

        # Analysis tool differentiation hint
        if (current_tool == "answer_specific_question" and
            "analyze_project_deeply" not in recent_tools and
            "analyze_vs_question" not in shown_hints and
            user_level in ["intermediate", "advanced"]):
            hints.append(self.hint_definitions["analyze_vs_question"])

        # Comprehensive assistant hint
        if (current_tool == "get_smart_suggestions" and
            "get_comprehensive_assistance" not in recent_tools and
            "comprehensive_vs_smart" not in shown_hints and
            user_level in ["advanced", "expert"]):
            hints.append(self.hint_definitions["comprehensive_vs_smart"])

        # Search tool differentiation hint
        if (current_tool == "search_memories" and
            "search_semantic" not in recent_tools and
            "semantic_vs_keyword" not in shown_hints):
            hints.append(self.hint_definitions["semantic_vs_keyword"])

        return hints

    def _get_parameter_hints(self, context: Dict[str, Any], shown_hints: Set[str]) -> List[UsageHint]:
        """Get parameter hints"""
        hints = []
        current_tool = context.get("current_tool")
        tool_params = context.get("tool_params", {})

        # Question analysis context hint
        if (current_tool == "answer_specific_question" and
            not tool_params.get("context") and
            "question_context_hint" not in shown_hints):
            hints.append(self.hint_definitions["question_context_hint"])

        # Smart suggestions project path hint
        if (current_tool == "get_smart_suggestions" and
            not tool_params.get("project_path") and
            "suggestions_project_path" not in shown_hints):
            hints.append(self.hint_definitions["suggestions_project_path"])

        # Analysis depth control hint
        if (current_tool == "analyze_project_deeply" and
            not tool_params.get("analysis_depth") and
            "analysis_depth_control" not in shown_hints):
            hints.append(self.hint_definitions["analysis_depth_control"])

        # Backup priority hint
        if (current_tool == "create_backup" and
            not tool_params.get("priority") and
            "backup_priority_setting" not in shown_hints):
            hints.append(self.hint_definitions["backup_priority_setting"])

        return hints

    def _get_error_prevention_hints(self, context: Dict[str, Any], shown_hints: Set[str]) -> List[UsageHint]:
        """Get error prevention hints"""
        hints = []
        error_message = context.get("error_message", "")

        # Provide corresponding hints based on error type
        if "backup_before_changes" not in shown_hints:
            hints.append(self.hint_definitions["backup_before_changes"])

        return hints

    def _get_workflow_hints(self, context: Dict[str, Any], shown_hints: Set[str]) -> List[UsageHint]:
        """Get workflow hints"""
        hints = []
        recent_tools = context.get("recent_tools", [])

        # Workflow automation hint
        if (len(recent_tools) > 8 and  # Used many tools
            "detect_workflow_patterns" not in recent_tools and
            "workflow_automation" not in shown_hints):
            hints.append(self.hint_definitions["workflow_automation"])

        return hints

    def mark_hint_shown(self, session_id: str, hint_id: str):
        """
        Mark hint as shown

        Args:
            session_id: Session ID
            hint_id: Hint ID
        """
        if session_id not in self.user_hint_history:
            self.user_hint_history[session_id] = set()

        self.user_hint_history[session_id].add(hint_id)
        self.logger.debug(f"Marked hint as shown: {session_id}, {hint_id}")

    def get_tool_specific_hints(self, tool_name: str, user_level: str = "beginner") -> List[UsageHint]:
        """
        Get usage hints for a specific tool

        Args:
            tool_name: Tool name
            user_level: User level

        Returns:
            List[UsageHint]: List of tool-related hints
        """
        hints = []

        for hint in self.hint_definitions.values():
            if hint.tool_name == tool_name:
                # Filter hints by user level
                if user_level == "beginner" and hint.hint_type in [HintType.USAGE_TIP, HintType.PARAMETER_HINT]:
                    hints.append(hint)
                elif user_level in ["intermediate", "advanced"] and hint.hint_type in [
                    HintType.OPTIMIZATION, HintType.FEATURE_HIGHLIGHT, HintType.WORKFLOW_SUGGESTION
                ]:
                    hints.append(hint)
                elif user_level == "expert":
                    hints.append(hint)

        return sorted(hints, key=lambda x: x.priority, reverse=True)

    def get_contextual_tips(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Get context-relevant quick tips

        Args:
            context: Context information

        Returns:
            Dict[str, str]: Tips {category: tip}
        """
        tips = {}
        current_tool = context.get("current_tool")
        user_level = context.get("user_level", "beginner")

        # Tool usage tips
        if current_tool:
            tool_tips = {
                "get_quick_context": "Call at the start of each session to get personalized context",
                "get_preferences": "Call before programming to ensure consistent code style",
                "answer_specific_question": "Provide detailed problem descriptions and context information",
                "analyze_project_deeply": "Suitable for complex decisions, can control analysis depth",
                "get_smart_suggestions": "Specify project path for more precise recommendations",
                "check_security_status": "Check regularly to promptly detect security risks",
                "create_backup": "Backup before important changes, set appropriate priority",
                "search_semantic": "Suitable for conceptual searching, find related ideas and topics"
            }

            if current_tool in tool_tips:
                tips["usage"] = tool_tips[current_tool]

        # User level related tips
        if user_level == "beginner":
            tips["level"] = "Recommend starting with basic tools: get_quick_context -> get_preferences -> remember"
        elif user_level == "intermediate":
            tips["level"] = "Can try intelligent analysis tools: analyze_project_deeply, get_smart_suggestions"
        elif user_level in ["advanced", "expert"]:
            tips["level"] = "Explore advanced features: get_comprehensive_assistance, adaptive learning tools"

        return tips

    def update_tool_usage_stats(self, session_id: str, tool_name: str,
                              params_used: Dict[str, Any], success: bool):
        """
        Update tool usage statistics

        Args:
            session_id: Session ID
            tool_name: Tool name
            params_used: Parameters used
            success: Whether successful
        """
        if session_id not in self.tool_usage_stats:
            self.tool_usage_stats[session_id] = {}

        if tool_name not in self.tool_usage_stats[session_id]:
            self.tool_usage_stats[session_id][tool_name] = {
                "usage_count": 0,
                "success_count": 0,
                "common_params": {},
                "last_used": None
            }

        stats = self.tool_usage_stats[session_id][tool_name]
        stats["usage_count"] += 1
        if success:
            stats["success_count"] += 1
        stats["last_used"] = datetime.now()

        # Count common parameters
        for param, value in params_used.items():
            if param not in stats["common_params"]:
                stats["common_params"][param] = {}

            value_str = str(value)
            if value_str not in stats["common_params"][param]:
                stats["common_params"][param][value_str] = 0
            stats["common_params"][param][value_str] += 1

    def get_usage_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics

        Args:
            session_id: Session ID, if None returns global statistics

        Returns:
            Dict[str, Any]: Statistics
        """
        if session_id and session_id in self.tool_usage_stats:
            return {
                "session_id": session_id,
                "tool_stats": self.tool_usage_stats[session_id],
                "hints_shown": len(self.user_hint_history.get(session_id, set())),
                "available_hints": len(self.hint_definitions)
            }
        else:
            # Global statistics
            total_sessions = len(self.tool_usage_stats)
            total_hints_shown = sum(len(hints) for hints in self.user_hint_history.values())

            return {
                "total_sessions": total_sessions,
                "total_hints_shown": total_hints_shown,
                "total_hint_definitions": len(self.hint_definitions),
                "active_sessions": list(self.tool_usage_stats.keys())
            }

    def cleanup_old_data(self, max_age_days: int = 7):
        """
        Clean up old data

        Args:
            max_age_days: Maximum retention days
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        sessions_to_remove = []

        for session_id, tool_stats in self.tool_usage_stats.items():
            # Check last usage time
            last_activity = None
            for tool_data in tool_stats.values():
                if tool_data.get("last_used"):
                    if not last_activity or tool_data["last_used"] > last_activity:
                        last_activity = tool_data["last_used"]

            if last_activity and last_activity < cutoff_date:
                sessions_to_remove.append(session_id)

        # Clean up expired sessions
        for session_id in sessions_to_remove:
            if session_id in self.tool_usage_stats:
                del self.tool_usage_stats[session_id]
            if session_id in self.user_hint_history:
                del self.user_hint_history[session_id]

        if sessions_to_remove:
            self.logger.info(f"Cleaned up data for {len(sessions_to_remove)} expired sessions")
