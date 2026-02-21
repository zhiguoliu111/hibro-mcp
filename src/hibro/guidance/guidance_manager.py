#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Guidance Manager
Main manager that coordinates tool recommendations, usage hints, and learning paths
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import Config


class UserLevel(Enum):
    """User proficiency level"""
    BEGINNER = "beginner"      # Beginner user
    INTERMEDIATE = "intermediate"  # Intermediate user
    ADVANCED = "advanced"      # Advanced user
    EXPERT = "expert"         # Expert user


class GuidanceContext(Enum):
    """Guidance context type"""
    PROJECT_START = "project_start"      # Project start
    ERROR_OCCURRED = "error_occurred"    # Error occurred
    TASK_TRANSITION = "task_transition"  # Task transition
    TOOL_DISCOVERY = "tool_discovery"    # Tool discovery
    LEARNING_PATH = "learning_path"      # Learning path


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_level: UserLevel
    project_path: Optional[str] = None
    current_task: Optional[str] = None
    recent_tools: List[str] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)
    learning_progress: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class GuidanceRecommendation:
    """Guidance recommendation"""
    recommendation_id: str
    recommendation_type: str  # tool, hint, learning_path
    title: str
    description: str
    priority: int  # 1-10, 10 is highest
    context: GuidanceContext
    target_tools: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_value: float = 0.0  # Estimated value score


class GuidanceManager:
    """Intelligent guidance manager"""

    def __init__(self, config: Config):
        """
        Initialize guidance manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.guidance_manager')

        # User session management
        self.active_sessions: Dict[str, UserSession] = {}

        # Tool relationship network
        self.tool_relationships = self._build_tool_relationships()

        # Learning path definitions
        self.learning_paths = self._define_learning_paths()

        # Usage statistics
        self.tool_usage_stats: Dict[str, Dict[str, int]] = {}

        self.logger.info("Intelligent guidance manager initialized")

    def _build_tool_relationships(self) -> Dict[str, Dict[str, float]]:
        """Build tool relationship network"""
        # Define relationships and weights between tools
        relationships = {
            # Core entry tools
            "get_quick_context": {
                "get_preferences": 0.9,
                "get_project_context": 0.8,
                "get_smart_suggestions": 0.7,
                "check_security_status": 0.6
            },

            # Reasoning and analysis tools
            "answer_specific_question": {
                "analyze_project_deeply": 0.8,
                "search_semantic": 0.7,
                "get_smart_suggestions": 0.6
            },

            "analyze_project_deeply": {
                "answer_specific_question": 0.7,
                "build_knowledge_graph": 0.8,
                "get_comprehensive_assistance": 0.9
            },

            # Intelligent assistant tools
            "get_smart_suggestions": {
                "detect_workflow_patterns": 0.8,
                "get_workflow_recommendations": 0.9,
                "get_intelligent_reminders": 0.7
            },

            "get_comprehensive_assistance": {
                "get_smart_suggestions": 0.8,
                "detect_workflow_patterns": 0.7,
                "get_intelligent_reminders": 0.6
            },

            # Security monitoring tools
            "check_security_status": {
                "apply_security_policy": 0.9,
                "get_security_events": 0.8,
                "create_backup": 0.7
            },

            "create_backup": {
                "restore_backup": 0.8,
                "get_backup_statistics": 0.7,
                "check_security_status": 0.6
            },

            # Memory management tools
            "search_memories": {
                "search_semantic": 0.8,
                "remember": 0.7,
                "analyze_conversation": 0.6
            },

            "remember": {
                "analyze_conversation": 0.9,
                "get_preferences": 0.7,
                "search_memories": 0.6
            }
        }

        return relationships

    def _define_learning_paths(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define learning paths"""
        return {
            "beginner_path": [
                {
                    "stage": "Basic Introduction",
                    "tools": ["get_quick_context", "get_preferences", "remember"],
                    "description": "Learn basic context retrieval and preference management",
                    "estimated_time": "10-15 minutes"
                },
                {
                    "stage": "Information Retrieval",
                    "tools": ["search_memories", "search_semantic", "get_project_context"],
                    "description": "Master information search and project context retrieval",
                    "estimated_time": "15-20 minutes"
                },
                {
                    "stage": "Intelligent Analysis",
                    "tools": ["answer_specific_question", "get_smart_suggestions"],
                    "description": "Use intelligent analysis and recommendation features",
                    "estimated_time": "20-25 minutes"
                }
            ],

            "intermediate_path": [
                {
                    "stage": "Advanced Analysis",
                    "tools": ["analyze_project_deeply", "build_knowledge_graph"],
                    "description": "Deep project analysis and knowledge graph construction",
                    "estimated_time": "25-30 minutes"
                },
                {
                    "stage": "Workflow Optimization",
                    "tools": ["detect_workflow_patterns", "get_workflow_recommendations", "execute_workflow"],
                    "description": "Workflow detection and automation",
                    "estimated_time": "30-40 minutes"
                },
                {
                    "stage": "Security Management",
                    "tools": ["check_security_status", "apply_security_policy", "create_backup"],
                    "description": "System security and data protection",
                    "estimated_time": "20-30 minutes"
                }
            ],

            "advanced_path": [
                {
                    "stage": "Comprehensive Coordination",
                    "tools": ["get_comprehensive_assistance", "process_intelligent_assistant"],
                    "description": "Use comprehensive intelligent assistant features",
                    "estimated_time": "40-50 minutes"
                },
                {
                    "stage": "Adaptive Learning",
                    "tools": ["track_user_behavior", "get_personalized_recommendations", "adaptive_importance_scoring"],
                    "description": "Personalized recommendations and adaptive learning",
                    "estimated_time": "30-45 minutes"
                },
                {
                    "stage": "System Optimization",
                    "tools": ["get_assistant_statistics", "get_learning_insights", "perform_security_scan"],
                    "description": "System performance analysis and optimization",
                    "estimated_time": "35-45 minutes"
                }
            ]
        }

    def create_user_session(self, session_id: str, user_level: UserLevel = UserLevel.BEGINNER,
                          project_path: Optional[str] = None) -> UserSession:
        """
        Create user session

        Args:
            session_id: Session ID
            user_level: User proficiency level
            project_path: Project path

        Returns:
            UserSession: User session object
        """
        session = UserSession(
            session_id=session_id,
            user_level=user_level,
            project_path=project_path
        )

        self.active_sessions[session_id] = session
        self.logger.info(f"Created user session: {session_id}, level: {user_level.value}")

        return session

    def get_contextual_recommendations(self, session_id: str, context: GuidanceContext,
                                     current_tool: Optional[str] = None,
                                     error_message: Optional[str] = None) -> List[GuidanceRecommendation]:
        """
        Get context-relevant recommendations

        Args:
            session_id: Session ID
            context: Guidance context
            current_tool: Currently used tool
            error_message: Error message

        Returns:
            List[GuidanceRecommendation]: List of recommendations
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return []

        recommendations = []

        if context == GuidanceContext.PROJECT_START:
            recommendations.extend(self._get_project_start_recommendations(session))
        elif context == GuidanceContext.ERROR_OCCURRED:
            recommendations.extend(self._get_error_recovery_recommendations(session, error_message))
        elif context == GuidanceContext.TOOL_DISCOVERY:
            recommendations.extend(self._get_tool_discovery_recommendations(session, current_tool))
        elif context == GuidanceContext.TASK_TRANSITION:
            recommendations.extend(self._get_task_transition_recommendations(session))
        elif context == GuidanceContext.LEARNING_PATH:
            recommendations.extend(self._get_learning_path_recommendations(session))

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)

        return recommendations[:5]  # Return top 5 recommendations

    def _get_project_start_recommendations(self, session: UserSession) -> List[GuidanceRecommendation]:
        """Get project start recommendations"""
        recommendations = []

        # Basic recommendation: Get context
        recommendations.append(GuidanceRecommendation(
            recommendation_id="project_start_context",
            recommendation_type="tool",
            title="Get Project Context",
            description="First call get_quick_context to get user preferences and project information",
            priority=10,
            context=GuidanceContext.PROJECT_START,
            target_tools=["get_quick_context"]
        ))

        # If beginner user, recommend learning path
        if session.user_level == UserLevel.BEGINNER:
            recommendations.append(GuidanceRecommendation(
                recommendation_id="beginner_learning_path",
                recommendation_type="learning_path",
                title="Beginner Learning Path",
                description="Recommend following the beginner learning path to gradually master core features",
                priority=9,
                context=GuidanceContext.PROJECT_START,
                target_tools=["get_quick_context", "get_preferences", "remember"]
            ))

        # Project analysis recommendation
        if session.project_path:
            recommendations.append(GuidanceRecommendation(
                recommendation_id="project_analysis",
                recommendation_type="tool",
                title="Project Intelligent Analysis",
                description="Use get_smart_suggestions to get project-related intelligent recommendations",
                priority=8,
                context=GuidanceContext.PROJECT_START,
                target_tools=["get_smart_suggestions"]
            ))

        return recommendations

    def _get_error_recovery_recommendations(self, session: UserSession,
                                          error_message: Optional[str]) -> List[GuidanceRecommendation]:
        """Get error recovery recommendations"""
        recommendations = []

        # Record error
        if error_message:
            session.error_history.append(error_message)

        # Problem analysis recommendation
        recommendations.append(GuidanceRecommendation(
            recommendation_id="error_analysis",
            recommendation_type="tool",
            title="Error Analysis",
            description="Use answer_specific_question to analyze error causes and solutions",
            priority=10,
            context=GuidanceContext.ERROR_OCCURRED,
            target_tools=["answer_specific_question"]
        ))

        # If errors are frequent, recommend deep analysis
        if len(session.error_history) > 3:
            recommendations.append(GuidanceRecommendation(
                recommendation_id="deep_analysis",
                recommendation_type="tool",
                title="Deep Project Analysis",
                description="Frequent errors may require deep analysis, recommend using analyze_project_deeply",
                priority=9,
                context=GuidanceContext.ERROR_OCCURRED,
                target_tools=["analyze_project_deeply"]
            ))

        return recommendations

    def _get_tool_discovery_recommendations(self, session: UserSession,
                                          current_tool: Optional[str]) -> List[GuidanceRecommendation]:
        """Get tool discovery recommendations"""
        recommendations = []

        if not current_tool:
            return recommendations

        # Recommend related tools based on tool associations
        related_tools = self.tool_relationships.get(current_tool, {})

        for tool, weight in sorted(related_tools.items(), key=lambda x: x[1], reverse=True)[:3]:
            if tool not in session.recent_tools:
                recommendations.append(GuidanceRecommendation(
                    recommendation_id=f"related_tool_{tool}",
                    recommendation_type="tool",
                    title=f"Related Tool Recommendation: {tool}",
                    description=f"Based on usage of {current_tool}, recommend trying {tool}",
                    priority=int(weight * 10),
                    context=GuidanceContext.TOOL_DISCOVERY,
                    target_tools=[tool]
                ))

        return recommendations

    def _get_task_transition_recommendations(self, session: UserSession) -> List[GuidanceRecommendation]:
        """Get task transition recommendations"""
        recommendations = []

        # Workflow detection recommendation
        recommendations.append(GuidanceRecommendation(
            recommendation_id="workflow_detection",
            recommendation_type="tool",
            title="Workflow Pattern Detection",
            description="Use detect_workflow_patterns to detect automatable workflows",
            priority=8,
            context=GuidanceContext.TASK_TRANSITION,
            target_tools=["detect_workflow_patterns"]
        ))

        # Intelligent reminders recommendation
        recommendations.append(GuidanceRecommendation(
            recommendation_id="intelligent_reminders",
            recommendation_type="tool",
            title="Intelligent Reminders",
            description="Get intelligent reminders and recommendations based on project status",
            priority=7,
            context=GuidanceContext.TASK_TRANSITION,
            target_tools=["get_intelligent_reminders"]
        ))

        return recommendations

    def _get_learning_path_recommendations(self, session: UserSession) -> List[GuidanceRecommendation]:
        """Get learning path recommendations"""
        recommendations = []

        # Recommend learning path based on user level
        path_key = f"{session.user_level.value}_path"
        learning_path = self.learning_paths.get(path_key, [])

        for i, stage in enumerate(learning_path):
            # Check if this stage is completed
            stage_progress = session.learning_progress.get(f"stage_{i}", 0.0)

            if stage_progress < 1.0:  # Uncompleted stage
                recommendations.append(GuidanceRecommendation(
                    recommendation_id=f"learning_stage_{i}",
                    recommendation_type="learning_path",
                    title=f"Learning Stage: {stage['stage']}",
                    description=stage['description'],
                    priority=10 - i,  # Decrease priority by order
                    context=GuidanceContext.LEARNING_PATH,
                    target_tools=stage['tools']
                ))
                break  # Only recommend next uncompleted stage

        return recommendations

    def update_user_activity(self, session_id: str, tool_used: str, success: bool = True):
        """
        Update user activity

        Args:
            session_id: Session ID
            tool_used: Tool used
            success: Whether successful
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return

        # Update recently used tools
        if tool_used not in session.recent_tools:
            session.recent_tools.append(tool_used)
            if len(session.recent_tools) > 10:  # Keep last 10 tools
                session.recent_tools.pop(0)

        # Update activity time
        session.last_activity = datetime.now()

        # Update usage statistics
        if session_id not in self.tool_usage_stats:
            self.tool_usage_stats[session_id] = {}

        if tool_used not in self.tool_usage_stats[session_id]:
            self.tool_usage_stats[session_id][tool_used] = 0

        self.tool_usage_stats[session_id][tool_used] += 1

        self.logger.debug(f"Updated user activity: {session_id}, tool: {tool_used}, success: {success}")

    def get_usage_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics

        Args:
            session_id: Session ID, if None returns global statistics

        Returns:
            Dict[str, Any]: Statistics
        """
        if session_id:
            session = self.active_sessions.get(session_id)
            if not session:
                return {}

            return {
                "session_id": session_id,
                "user_level": session.user_level.value,
                "recent_tools": session.recent_tools,
                "tool_usage": self.tool_usage_stats.get(session_id, {}),
                "learning_progress": session.learning_progress,
                "error_count": len(session.error_history)
            }
        else:
            # Global statistics
            total_sessions = len(self.active_sessions)
            total_tools_used = sum(len(stats) for stats in self.tool_usage_stats.values())

            # Most popular tools
            tool_popularity = {}
            for session_stats in self.tool_usage_stats.values():
                for tool, count in session_stats.items():
                    tool_popularity[tool] = tool_popularity.get(tool, 0) + count

            popular_tools = sorted(tool_popularity.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "total_sessions": total_sessions,
                "total_tools_used": total_tools_used,
                "popular_tools": popular_tools,
                "active_sessions": list(self.active_sessions.keys())
            }

    def cleanup_inactive_sessions(self, max_inactive_hours: int = 24):
        """
        Clean up inactive sessions

        Args:
            max_inactive_hours: Maximum inactive hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_inactive_hours)
        inactive_sessions = []

        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time:
                inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            del self.active_sessions[session_id]
            if session_id in self.tool_usage_stats:
                del self.tool_usage_stats[session_id]

        if inactive_sessions:
            self.logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
