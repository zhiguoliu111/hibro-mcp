#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Recommender
Provides intelligent tool recommendations based on user behavior, project context, and tool relationships
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math

from ..utils.config import Config


class RecommendationReason(Enum):
    """Recommendation reason"""
    RELATED_TOOL = "related_tool"          # Related tool
    USER_PATTERN = "user_pattern"          # User usage pattern
    PROJECT_CONTEXT = "project_context"    # Project context
    ERROR_RECOVERY = "error_recovery"      # Error recovery
    WORKFLOW_OPTIMIZATION = "workflow_optimization"  # Workflow optimization
    LEARNING_PATH = "learning_path"        # Learning path
    POPULAR_CHOICE = "popular_choice"      # Popular choice


@dataclass
class ToolRecommendation:
    """Tool recommendation"""
    tool_name: str
    confidence_score: float  # 0.0-1.0
    reason: RecommendationReason
    explanation: str
    usage_hint: str
    prerequisites: List[str]
    estimated_value: float  # Estimated value score


class ToolRecommender:
    """Tool recommender"""

    def __init__(self, config: Config):
        """
        Initialize tool recommender

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.tool_recommender')

        # Tool categories and features
        self.tool_categories = self._define_tool_categories()
        self.tool_features = self._define_tool_features()

        # Usage patterns and statistics
        self.usage_patterns: Dict[str, Dict[str, float]] = {}
        self.tool_success_rates: Dict[str, float] = {}

        # Recommendation cache
        self.recommendation_cache: Dict[str, Tuple[List[ToolRecommendation], datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)

        self.logger.info("Tool recommender initialized")

    def _define_tool_categories(self) -> Dict[str, List[str]]:
        """Define tool categories"""
        return {
            "Beginner Essentials": [
                "get_quick_context", "get_preferences", "remember", "search_memories"
            ],
            "Information Retrieval": [
                "search_semantic", "get_project_context", "get_recent_decisions", "get_important_facts"
            ],
            "Intelligent Analysis": [
                "answer_specific_question", "analyze_project_deeply", "build_knowledge_graph",
                "analyze_causal_relations", "predict_next_needs"
            ],
            "Intelligent Assistant": [
                "get_smart_suggestions", "get_comprehensive_assistance", "detect_workflow_patterns",
                "get_workflow_recommendations", "execute_workflow", "get_intelligent_reminders"
            ],
            "Adaptive Learning": [
                "track_user_behavior", "get_personalized_recommendations", "analyze_user_patterns",
                "adaptive_importance_scoring", "get_learning_insights"
            ],
            "Security Monitoring": [
                "check_security_status", "apply_security_policy", "get_security_events",
                "resolve_security_event", "perform_security_scan"
            ],
            "Data Protection": [
                "create_backup", "restore_backup", "get_backup_statistics",
                "register_sync_device", "start_device_migration"
            ],
            "System Management": [
                "get_system_health", "get_status", "update_memory", "forget",
                "set_project_context", "set_active_task", "complete_active_task"
            ]
        }

    def _define_tool_features(self) -> Dict[str, Dict[str, Any]]:
        """Define tool features"""
        return {
            # Beginner tool features
            "get_quick_context": {
                "complexity": 1,  # 1-5, 1 is simplest
                "frequency": "high",  # high, medium, low
                "prerequisites": [],
                "use_cases": ["session_start", "context_needed"],
                "output_type": "structured_data"
            },
            "get_preferences": {
                "complexity": 1,
                "frequency": "high",
                "prerequisites": [],
                "use_cases": ["coding_start", "preference_check"],
                "output_type": "structured_data"
            },

            # Analysis tool features
            "answer_specific_question": {
                "complexity": 2,
                "frequency": "high",
                "prerequisites": [],
                "use_cases": ["problem_solving", "quick_analysis"],
                "output_type": "analysis_result"
            },
            "analyze_project_deeply": {
                "complexity": 4,
                "frequency": "medium",
                "prerequisites": ["get_quick_context"],
                "use_cases": ["deep_analysis", "decision_making"],
                "output_type": "comprehensive_report"
            },

            # Intelligent assistant tool features
            "get_smart_suggestions": {
                "complexity": 2,
                "frequency": "high",
                "prerequisites": ["get_quick_context"],
                "use_cases": ["optimization", "best_practices"],
                "output_type": "suggestions_list"
            },
            "get_comprehensive_assistance": {
                "complexity": 4,
                "frequency": "medium",
                "prerequisites": ["get_quick_context", "get_smart_suggestions"],
                "use_cases": ["complex_problems", "holistic_guidance"],
                "output_type": "comprehensive_assistance"
            },

            # Security tool features
            "check_security_status": {
                "complexity": 2,
                "frequency": "medium",
                "prerequisites": [],
                "use_cases": ["security_audit", "health_check"],
                "output_type": "status_report"
            },
            "create_backup": {
                "complexity": 2,
                "frequency": "low",
                "prerequisites": [],
                "use_cases": ["data_protection", "before_changes"],
                "output_type": "operation_result"
            }
        }

    def get_recommendations(self, user_context: Dict[str, Any],
                          max_recommendations: int = 5) -> List[ToolRecommendation]:
        """
        Get tool recommendations

        Args:
            user_context: User context information
            max_recommendations: Maximum number of recommendations

        Returns:
            List[ToolRecommendation]: List of recommendations
        """
        # Generate cache key
        cache_key = self._generate_cache_key(user_context)

        # Check cache
        if cache_key in self.recommendation_cache:
            cached_recommendations, cache_time = self.recommendation_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_recommendations[:max_recommendations]

        # Generate recommendations
        recommendations = []

        # 1. Recommendations based on user level
        recommendations.extend(self._get_level_based_recommendations(user_context))

        # 2. Related tool recommendations based on current tool
        recommendations.extend(self._get_related_tool_recommendations(user_context))

        # 3. Recommendations based on project context
        recommendations.extend(self._get_context_based_recommendations(user_context))

        # 4. Recommendations based on usage patterns
        recommendations.extend(self._get_pattern_based_recommendations(user_context))

        # 5. Recommendations based on error history
        recommendations.extend(self._get_error_based_recommendations(user_context))

        # Deduplicate and rank
        recommendations = self._deduplicate_and_rank(recommendations)

        # Cache results
        self.recommendation_cache[cache_key] = (recommendations, datetime.now())

        return recommendations[:max_recommendations]

    def _generate_cache_key(self, user_context: Dict[str, Any]) -> str:
        """Generate cache key"""
        key_parts = [
            user_context.get("user_level", "beginner"),
            user_context.get("current_tool", ""),
            user_context.get("project_type", ""),
            str(len(user_context.get("recent_tools", []))),
            str(len(user_context.get("error_history", [])))
        ]
        return "_".join(key_parts)

    def _get_level_based_recommendations(self, user_context: Dict[str, Any]) -> List[ToolRecommendation]:
        """Recommendations based on user level"""
        recommendations = []
        user_level = user_context.get("user_level", "beginner")
        recent_tools = set(user_context.get("recent_tools", []))

        if user_level == "beginner":
            # Recommend beginner tools for beginners
            beginner_tools = self.tool_categories["Beginner Essentials"]
            for tool in beginner_tools:
                if tool not in recent_tools:
                    recommendations.append(ToolRecommendation(
                        tool_name=tool,
                        confidence_score=0.9,
                        reason=RecommendationReason.LEARNING_PATH,
                        explanation=f"Essential tool recommended for beginner users to master",
                        usage_hint=self._get_tool_usage_hint(tool),
                        prerequisites=[],
                        estimated_value=0.8
                    ))

        elif user_level == "intermediate":
            # Recommend intelligent analysis tools for intermediate users
            intermediate_tools = self.tool_categories["Intelligent Analysis"]
            for tool in intermediate_tools:
                if tool not in recent_tools:
                    recommendations.append(ToolRecommendation(
                        tool_name=tool,
                        confidence_score=0.8,
                        reason=RecommendationReason.LEARNING_PATH,
                        explanation=f"Intelligent analysis tool suitable for intermediate users",
                        usage_hint=self._get_tool_usage_hint(tool),
                        prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                        estimated_value=0.7
                    ))

        elif user_level in ["advanced", "expert"]:
            # Recommend complex tools for advanced users
            advanced_tools = self.tool_categories["Adaptive Learning"] + self.tool_categories["Intelligent Assistant"]
            for tool in advanced_tools:
                if tool not in recent_tools:
                    tool_features = self.tool_features.get(tool, {})
                    if tool_features.get("complexity", 1) >= 3:  # Complex tools
                        recommendations.append(ToolRecommendation(
                            tool_name=tool,
                            confidence_score=0.7,
                            reason=RecommendationReason.LEARNING_PATH,
                            explanation=f"Complex feature tool suitable for advanced users",
                            usage_hint=self._get_tool_usage_hint(tool),
                            prerequisites=tool_features.get("prerequisites", []),
                            estimated_value=0.9
                        ))

        return recommendations

    def _get_related_tool_recommendations(self, user_context: Dict[str, Any]) -> List[ToolRecommendation]:
        """Related tool recommendations based on current tool"""
        recommendations = []
        current_tool = user_context.get("current_tool")
        recent_tools = set(user_context.get("recent_tools", []))

        if not current_tool:
            return recommendations

        # Tool relationships (copied from GuidanceManager)
        tool_relationships = {
            "get_quick_context": {
                "get_preferences": 0.9,
                "get_project_context": 0.8,
                "get_smart_suggestions": 0.7
            },
            "answer_specific_question": {
                "analyze_project_deeply": 0.8,
                "search_semantic": 0.7,
                "get_smart_suggestions": 0.6
            },
            "get_smart_suggestions": {
                "detect_workflow_patterns": 0.8,
                "get_workflow_recommendations": 0.9,
                "get_intelligent_reminders": 0.7
            },
            "check_security_status": {
                "apply_security_policy": 0.9,
                "get_security_events": 0.8,
                "create_backup": 0.7
            }
        }

        related_tools = tool_relationships.get(current_tool, {})
        for tool, weight in related_tools.items():
            if tool not in recent_tools:
                recommendations.append(ToolRecommendation(
                    tool_name=tool,
                    confidence_score=weight,
                    reason=RecommendationReason.RELATED_TOOL,
                    explanation=f"Tool related to {current_tool}",
                    usage_hint=self._get_tool_usage_hint(tool),
                    prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                    estimated_value=weight * 0.8
                ))

        return recommendations

    def _get_context_based_recommendations(self, user_context: Dict[str, Any]) -> List[ToolRecommendation]:
        """Recommendations based on project context"""
        recommendations = []
        project_type = user_context.get("project_type", "")
        current_phase = user_context.get("current_phase", "")
        recent_tools = set(user_context.get("recent_tools", []))

        # Recommendations based on project type
        if project_type == "web":
            web_tools = ["check_security_status", "get_smart_suggestions", "detect_workflow_patterns"]
            for tool in web_tools:
                if tool not in recent_tools:
                    recommendations.append(ToolRecommendation(
                        tool_name=tool,
                        confidence_score=0.7,
                        reason=RecommendationReason.PROJECT_CONTEXT,
                        explanation=f"Tool suitable for {project_type} projects",
                        usage_hint=self._get_tool_usage_hint(tool),
                        prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                        estimated_value=0.6
                    ))

        # Recommendations based on development phase
        if current_phase == "development":
            dev_tools = ["get_smart_suggestions", "detect_workflow_patterns", "answer_specific_question"]
            for tool in dev_tools:
                if tool not in recent_tools:
                    recommendations.append(ToolRecommendation(
                        tool_name=tool,
                        confidence_score=0.8,
                        reason=RecommendationReason.PROJECT_CONTEXT,
                        explanation=f"Tool suitable for {current_phase} phase",
                        usage_hint=self._get_tool_usage_hint(tool),
                        prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                        estimated_value=0.7
                    ))

        elif current_phase == "testing":
            test_tools = ["check_security_status", "analyze_project_deeply", "create_backup"]
            for tool in test_tools:
                if tool not in recent_tools:
                    recommendations.append(ToolRecommendation(
                        tool_name=tool,
                        confidence_score=0.8,
                        reason=RecommendationReason.PROJECT_CONTEXT,
                        explanation=f"Tool suitable for {current_phase} phase",
                        usage_hint=self._get_tool_usage_hint(tool),
                        prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                        estimated_value=0.7
                    ))

        return recommendations

    def _get_pattern_based_recommendations(self, user_context: Dict[str, Any]) -> List[ToolRecommendation]:
        """Recommendations based on usage patterns"""
        recommendations = []
        session_id = user_context.get("session_id", "")
        recent_tools = set(user_context.get("recent_tools", []))

        if session_id not in self.usage_patterns:
            return recommendations

        user_patterns = self.usage_patterns[session_id]

        # Find tools frequently used by user but not recently used
        for tool, usage_frequency in user_patterns.items():
            if tool not in recent_tools and usage_frequency > 0.3:  # Usage frequency threshold
                recommendations.append(ToolRecommendation(
                    tool_name=tool,
                    confidence_score=usage_frequency,
                    reason=RecommendationReason.USER_PATTERN,
                    explanation=f"Recommended based on your usage habits",
                    usage_hint=self._get_tool_usage_hint(tool),
                    prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                    estimated_value=usage_frequency * 0.9
                ))

        return recommendations

    def _get_error_based_recommendations(self, user_context: Dict[str, Any]) -> List[ToolRecommendation]:
        """Recommendations based on error history"""
        recommendations = []
        error_history = user_context.get("error_history", [])
        recent_tools = set(user_context.get("recent_tools", []))

        if not error_history:
            return recommendations

        # If there's error history, recommend analysis and diagnostic tools
        error_recovery_tools = [
            "answer_specific_question",
            "analyze_project_deeply",
            "get_smart_suggestions",
            "check_security_status"
        ]

        for tool in error_recovery_tools:
            if tool not in recent_tools:
                recommendations.append(ToolRecommendation(
                    tool_name=tool,
                    confidence_score=0.8,
                    reason=RecommendationReason.ERROR_RECOVERY,
                    explanation=f"Help analyze and solve encountered problems",
                    usage_hint=self._get_tool_usage_hint(tool),
                    prerequisites=self.tool_features.get(tool, {}).get("prerequisites", []),
                    estimated_value=0.8
                ))

        return recommendations

    def _get_tool_usage_hint(self, tool_name: str) -> str:
        """Get tool usage hint"""
        hints = {
            "get_quick_context": "Recommend calling at the start of each session to get personalized context",
            "get_preferences": "Call before programming to ensure consistent code style",
            "answer_specific_question": "Suitable for quick answers to specific questions, such as error analysis",
            "analyze_project_deeply": "Suitable for comprehensive analysis of complex projects and decision support",
            "get_smart_suggestions": "Get intelligent recommendations based on project status",
            "check_security_status": "Regularly check system security status",
            "create_backup": "Recommend creating backup before important changes",
            "search_semantic": "Use semantic search to find related concepts and information",
            "detect_workflow_patterns": "Identify repetitive workflows that can be automated"
        }
        return hints.get(tool_name, "Refer to tool description for detailed usage")

    def _deduplicate_and_rank(self, recommendations: List[ToolRecommendation]) -> List[ToolRecommendation]:
        """Deduplicate and rank recommendations"""
        # Deduplicate by tool name, keep highest confidence
        tool_recommendations = {}
        for rec in recommendations:
            if rec.tool_name not in tool_recommendations or rec.confidence_score > tool_recommendations[rec.tool_name].confidence_score:
                tool_recommendations[rec.tool_name] = rec

        # Sort by composite score
        sorted_recommendations = sorted(
            tool_recommendations.values(),
            key=lambda x: x.confidence_score * x.estimated_value,
            reverse=True
        )

        return sorted_recommendations

    def update_usage_patterns(self, session_id: str, tool_name: str, success: bool = True):
        """
        Update usage patterns

        Args:
            session_id: Session ID
            tool_name: Tool name
            success: Whether used successfully
        """
        if session_id not in self.usage_patterns:
            self.usage_patterns[session_id] = {}

        # Update usage frequency (simple exponential moving average)
        current_frequency = self.usage_patterns[session_id].get(tool_name, 0.0)
        self.usage_patterns[session_id][tool_name] = current_frequency * 0.9 + 0.1

        # Update success rate
        if tool_name not in self.tool_success_rates:
            self.tool_success_rates[tool_name] = 0.5  # Initial value

        current_success_rate = self.tool_success_rates[tool_name]
        success_value = 1.0 if success else 0.0
        self.tool_success_rates[tool_name] = current_success_rate * 0.95 + success_value * 0.05

        self.logger.debug(f"Updated usage patterns: {session_id}, {tool_name}, success: {success}")

    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Get recommendation statistics"""
        return {
            "total_sessions": len(self.usage_patterns),
            "cache_size": len(self.recommendation_cache),
            "tool_success_rates": dict(sorted(
                self.tool_success_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "most_recommended_tools": self._get_most_recommended_tools()
        }

    def _get_most_recommended_tools(self) -> List[Tuple[str, int]]:
        """Get most recommended tools"""
        tool_recommendation_counts = {}

        for recommendations, _ in self.recommendation_cache.values():
            for rec in recommendations:
                tool_recommendation_counts[rec.tool_name] = tool_recommendation_counts.get(rec.tool_name, 0) + 1

        return sorted(tool_recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
