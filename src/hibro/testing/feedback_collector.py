#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Feedback Collector
Collects user experience feedback and analyzes improvement suggestions
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import Config


class FeedbackType(Enum):
    """Feedback type"""
    BUG_REPORT = "bug_report"              # Bug report
    FEATURE_REQUEST = "feature_request"    # Feature request
    USABILITY_ISSUE = "usability_issue"    # Usability issue
    PERFORMANCE_ISSUE = "performance_issue" # Performance issue
    DOCUMENTATION_ISSUE = "documentation_issue" # Documentation issue
    GENERAL_FEEDBACK = "general_feedback"   # General feedback


class FeedbackPriority(Enum):
    """Feedback priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(Enum):
    """Feedback status"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class UserFeedback:
    """User feedback"""
    feedback_id: str
    user_session: str
    feedback_type: FeedbackType
    priority: FeedbackPriority
    status: FeedbackStatus
    title: str
    description: str
    tool_name: Optional[str] = None
    error_message: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolution_notes: Optional[str] = None


@dataclass
class FeedbackAnalysis:
    """Feedback analysis result"""
    analysis_id: str
    feedback_ids: List[str]
    common_issues: List[str]
    improvement_suggestions: List[str]
    priority_areas: List[str]
    affected_tools: List[str]
    user_impact_score: float
    implementation_complexity: str
    created_at: datetime = field(default_factory=datetime.now)


class FeedbackCollector:
    """User feedback collector"""

    def __init__(self, config: Config):
        """
        Initialize feedback collector

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.feedback_collector')

        # Feedback storage
        self.feedbacks: List[UserFeedback] = []
        self.analyses: List[FeedbackAnalysis] = []

        # Feedback statistics
        self.feedback_stats = {
            "total_feedbacks": 0,
            "by_type": {},
            "by_priority": {},
            "by_status": {},
            "resolution_rate": 0.0
        }

        self.logger.info("User feedback collector initialized")

    def collect_feedback(self, user_session: str, feedback_type: str, title: str,
                        description: str, tool_name: Optional[str] = None,
                        error_message: Optional[str] = None,
                        user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect user feedback

        Args:
            user_session: User session ID
            feedback_type: Feedback type
            title: Feedback title
            description: Feedback description
            tool_name: Related tool name
            error_message: Error message
            user_context: User context

        Returns:
            Feedback ID
        """
        # Generate feedback ID
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedbacks)}"

        # Determine priority
        priority = self._determine_priority(feedback_type, error_message)

        # Create feedback object
        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_session=user_session,
            feedback_type=FeedbackType(feedback_type),
            priority=priority,
            status=FeedbackStatus.NEW,
            title=title,
            description=description,
            tool_name=tool_name,
            error_message=error_message,
            user_context=user_context or {}
        )

        # Store feedback
        self.feedbacks.append(feedback)
        self._update_stats()

        self.logger.info(f"Collected user feedback: {feedback_id}, type: {feedback_type}, priority: {priority.value}")
        return feedback_id

    def _determine_priority(self, feedback_type: str, error_message: Optional[str]) -> FeedbackPriority:
        """Determine feedback priority"""
        if feedback_type == "bug_report":
            if error_message and any(keyword in error_message.lower()
                                   for keyword in ["crash", "exception", "error", "fail"]):
                return FeedbackPriority.HIGH
            return FeedbackPriority.MEDIUM
        elif feedback_type == "performance_issue":
            return FeedbackPriority.HIGH
        elif feedback_type == "usability_issue":
            return FeedbackPriority.MEDIUM
        elif feedback_type == "feature_request":
            return FeedbackPriority.LOW
        else:
            return FeedbackPriority.MEDIUM

    def analyze_feedback_patterns(self, time_window_days: int = 30) -> FeedbackAnalysis:
        """
        Analyze feedback patterns

        Args:
            time_window_days: Analysis time window (days)

        Returns:
            Feedback analysis result
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_feedbacks = [f for f in self.feedbacks if f.created_at >= cutoff_date]

        if not recent_feedbacks:
            return FeedbackAnalysis(
                analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                feedback_ids=[],
                common_issues=[],
                improvement_suggestions=[],
                priority_areas=[],
                affected_tools=[],
                user_impact_score=0.0,
                implementation_complexity="low"
            )

        # Analyze common issues
        common_issues = self._identify_common_issues(recent_feedbacks)

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(recent_feedbacks)

        # Identify priority improvement areas
        priority_areas = self._identify_priority_areas(recent_feedbacks)

        # Count affected tools
        affected_tools = list(set(f.tool_name for f in recent_feedbacks if f.tool_name))

        # Calculate user impact score
        user_impact_score = self._calculate_user_impact_score(recent_feedbacks)

        # Assess implementation complexity
        implementation_complexity = self._assess_implementation_complexity(recent_feedbacks)

        analysis = FeedbackAnalysis(
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            feedback_ids=[f.feedback_id for f in recent_feedbacks],
            common_issues=common_issues,
            improvement_suggestions=improvement_suggestions,
            priority_areas=priority_areas,
            affected_tools=affected_tools,
            user_impact_score=user_impact_score,
            implementation_complexity=implementation_complexity
        )

        self.analyses.append(analysis)
        self.logger.info(f"Completed feedback analysis: {analysis.analysis_id}")
        return analysis

    def _identify_common_issues(self, feedbacks: List[UserFeedback]) -> List[str]:
        """Identify common issues"""
        issue_patterns = {}

        for feedback in feedbacks:
            # Group by tool name
            if feedback.tool_name:
                key = f"Tool {feedback.tool_name} related issues"
                issue_patterns[key] = issue_patterns.get(key, 0) + 1

            # Group by feedback type
            type_key = f"{feedback.feedback_type.value} type issues"
            issue_patterns[type_key] = issue_patterns.get(type_key, 0) + 1

            # Group by keywords
            keywords = ["parameter", "error", "performance", "interface", "documentation", "recommendation", "search"]
            for keyword in keywords:
                if keyword in feedback.description or keyword in feedback.title:
                    key = f"{keyword} related issues"
                    issue_patterns[key] = issue_patterns.get(key, 0) + 1

        # Return most frequent issues
        sorted_issues = sorted(issue_patterns.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:5] if count >= 2]

    def _generate_improvement_suggestions(self, feedbacks: List[UserFeedback]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Generate suggestions based on feedback type
        type_counts = {}
        for feedback in feedbacks:
            type_counts[feedback.feedback_type] = type_counts.get(feedback.feedback_type, 0) + 1

        for feedback_type, count in type_counts.items():
            if count >= 3:  # If a certain type of feedback is frequent
                if feedback_type == FeedbackType.USABILITY_ISSUE:
                    suggestions.append("Improve user interface design and simplify operation flow")
                elif feedback_type == FeedbackType.PERFORMANCE_ISSUE:
                    suggestions.append("Optimize system performance and reduce response time")
                elif feedback_type == FeedbackType.DOCUMENTATION_ISSUE:
                    suggestions.append("Improve documentation content and add usage examples")
                elif feedback_type == FeedbackType.BUG_REPORT:
                    suggestions.append("Strengthen quality assurance and reduce software defects")

        # Generate suggestions based on tool feedback
        tool_issues = {}
        for feedback in feedbacks:
            if feedback.tool_name:
                tool_issues[feedback.tool_name] = tool_issues.get(feedback.tool_name, 0) + 1

        for tool, count in tool_issues.items():
            if count >= 2:
                suggestions.append(f"Focus on optimizing the user experience of {tool} tool")

        return suggestions[:10]  # Return top 10 suggestions

    def _identify_priority_areas(self, feedbacks: List[UserFeedback]) -> List[str]:
        """Identify priority improvement areas"""
        priority_areas = []

        # Count high-priority feedback areas
        high_priority_feedbacks = [f for f in feedbacks if f.priority in [FeedbackPriority.HIGH, FeedbackPriority.CRITICAL]]

        if len(high_priority_feedbacks) > len(feedbacks) * 0.3:  # If high-priority feedback exceeds 30%
            priority_areas.append("System stability and error handling")

        # Count tool-related issues
        tool_feedback_count = len([f for f in feedbacks if f.tool_name])
        if tool_feedback_count > len(feedbacks) * 0.5:
            priority_areas.append("Tool functionality and usability")

        # Count performance-related issues
        performance_feedback_count = len([f for f in feedbacks if f.feedback_type == FeedbackType.PERFORMANCE_ISSUE])
        if performance_feedback_count > 2:
            priority_areas.append("System performance optimization")

        # Count documentation-related issues
        doc_feedback_count = len([f for f in feedbacks if f.feedback_type == FeedbackType.DOCUMENTATION_ISSUE])
        if doc_feedback_count > 1:
            priority_areas.append("Documentation and help system")

        return priority_areas

    def _calculate_user_impact_score(self, feedbacks: List[UserFeedback]) -> float:
        """Calculate user impact score"""
        if not feedbacks:
            return 0.0

        total_score = 0.0
        for feedback in feedbacks:
            # Calculate score based on priority
            if feedback.priority == FeedbackPriority.CRITICAL:
                score = 1.0
            elif feedback.priority == FeedbackPriority.HIGH:
                score = 0.8
            elif feedback.priority == FeedbackPriority.MEDIUM:
                score = 0.5
            else:
                score = 0.2

            # Adjust score based on feedback type
            if feedback.feedback_type in [FeedbackType.BUG_REPORT, FeedbackType.PERFORMANCE_ISSUE]:
                score *= 1.2
            elif feedback.feedback_type == FeedbackType.USABILITY_ISSUE:
                score *= 1.1

            total_score += score

        return min(total_score / len(feedbacks), 1.0)  # Normalize to 0-1

    def _assess_implementation_complexity(self, feedbacks: List[UserFeedback]) -> str:
        """Assess implementation complexity"""
        complexity_scores = []

        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.BUG_REPORT:
                complexity_scores.append(2)  # Medium complexity
            elif feedback.feedback_type == FeedbackType.PERFORMANCE_ISSUE:
                complexity_scores.append(3)  # High complexity
            elif feedback.feedback_type == FeedbackType.USABILITY_ISSUE:
                complexity_scores.append(2)  # Medium complexity
            elif feedback.feedback_type == FeedbackType.FEATURE_REQUEST:
                complexity_scores.append(3)  # High complexity
            elif feedback.feedback_type == FeedbackType.DOCUMENTATION_ISSUE:
                complexity_scores.append(1)  # Low complexity
            else:
                complexity_scores.append(2)  # Default medium complexity

        if not complexity_scores:
            return "low"

        avg_complexity = sum(complexity_scores) / len(complexity_scores)

        if avg_complexity <= 1.5:
            return "low"
        elif avg_complexity <= 2.5:
            return "medium"
        else:
            return "high"

    def update_feedback_status(self, feedback_id: str, status: str, resolution_notes: Optional[str] = None) -> bool:
        """
        Update feedback status

        Args:
            feedback_id: Feedback ID
            status: New status
            resolution_notes: Resolution notes

        Returns:
            Whether update was successful
        """
        feedback = next((f for f in self.feedbacks if f.feedback_id == feedback_id), None)
        if not feedback:
            return False

        feedback.status = FeedbackStatus(status)
        feedback.updated_at = datetime.now()
        if resolution_notes:
            feedback.resolution_notes = resolution_notes

        self._update_stats()
        self.logger.info(f"Feedback status updated: {feedback_id} -> {status}")
        return True

    def get_feedback_summary(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Get feedback summary

        Args:
            time_window_days: Time window (days)

        Returns:
            Feedback summary
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_feedbacks = [f for f in self.feedbacks if f.created_at >= cutoff_date]

        # Count feedbacks by type
        type_stats = {}
        priority_stats = {}
        status_stats = {}
        tool_stats = {}

        for feedback in recent_feedbacks:
            # Count by type
            type_key = feedback.feedback_type.value
            type_stats[type_key] = type_stats.get(type_key, 0) + 1

            # Count by priority
            priority_key = feedback.priority.value
            priority_stats[priority_key] = priority_stats.get(priority_key, 0) + 1

            # Count by status
            status_key = feedback.status.value
            status_stats[status_key] = status_stats.get(status_key, 0) + 1

            # Count by tool
            if feedback.tool_name:
                tool_stats[feedback.tool_name] = tool_stats.get(feedback.tool_name, 0) + 1

        # Calculate resolution rate
        resolved_count = status_stats.get("resolved", 0) + status_stats.get("closed", 0)
        resolution_rate = resolved_count / len(recent_feedbacks) if recent_feedbacks else 0

        return {
            "time_window_days": time_window_days,
            "total_feedbacks": len(recent_feedbacks),
            "by_type": type_stats,
            "by_priority": priority_stats,
            "by_status": status_stats,
            "by_tool": tool_stats,
            "resolution_rate": resolution_rate,
            "avg_resolution_time": self._calculate_avg_resolution_time(recent_feedbacks)
        }

    def _calculate_avg_resolution_time(self, feedbacks: List[UserFeedback]) -> float:
        """Calculate average resolution time (hours)"""
        resolved_feedbacks = [f for f in feedbacks
                            if f.status in [FeedbackStatus.RESOLVED, FeedbackStatus.CLOSED]]

        if not resolved_feedbacks:
            return 0.0

        total_time = 0.0
        for feedback in resolved_feedbacks:
            resolution_time = (feedback.updated_at - feedback.created_at).total_seconds() / 3600
            total_time += resolution_time

        return total_time / len(resolved_feedbacks)

    def _update_stats(self):
        """Update feedback statistics"""
        self.feedback_stats["total_feedbacks"] = len(self.feedbacks)

        # Count by type
        type_counts = {}
        priority_counts = {}
        status_counts = {}

        for feedback in self.feedbacks:
            type_counts[feedback.feedback_type.value] = type_counts.get(feedback.feedback_type.value, 0) + 1
            priority_counts[feedback.priority.value] = priority_counts.get(feedback.priority.value, 0) + 1
            status_counts[feedback.status.value] = status_counts.get(feedback.status.value, 0) + 1

        self.feedback_stats["by_type"] = type_counts
        self.feedback_stats["by_priority"] = priority_counts
        self.feedback_stats["by_status"] = status_counts

        # Calculate resolution rate
        resolved_count = status_counts.get("resolved", 0) + status_counts.get("closed", 0)
        self.feedback_stats["resolution_rate"] = resolved_count / len(self.feedbacks) if self.feedbacks else 0

    def export_feedback_data(self, format: str = "json") -> Dict[str, Any]:
        """
        Export feedback data

        Args:
            format: Export format

        Returns:
            Exported data
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_feedbacks": len(self.feedbacks),
            "feedbacks": [],
            "analyses": [],
            "statistics": self.feedback_stats
        }

        # Export feedback data
        for feedback in self.feedbacks:
            export_data["feedbacks"].append({
                "feedback_id": feedback.feedback_id,
                "user_session": feedback.user_session,
                "feedback_type": feedback.feedback_type.value,
                "priority": feedback.priority.value,
                "status": feedback.status.value,
                "title": feedback.title,
                "description": feedback.description,
                "tool_name": feedback.tool_name,
                "error_message": feedback.error_message,
                "user_context": feedback.user_context,
                "created_at": feedback.created_at.isoformat(),
                "updated_at": feedback.updated_at.isoformat(),
                "resolution_notes": feedback.resolution_notes
            })

        # Export analysis data
        for analysis in self.analyses:
            export_data["analyses"].append({
                "analysis_id": analysis.analysis_id,
                "feedback_ids": analysis.feedback_ids,
                "common_issues": analysis.common_issues,
                "improvement_suggestions": analysis.improvement_suggestions,
                "priority_areas": analysis.priority_areas,
                "affected_tools": analysis.affected_tools,
                "user_impact_score": analysis.user_impact_score,
                "implementation_complexity": analysis.implementation_complexity,
                "created_at": analysis.created_at.isoformat()
            })

        return export_data

    def cleanup_old_feedback(self, max_age_days: int = 90):
        """
        Clean up old feedback data

        Args:
            max_age_days: Maximum retention days
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Keep unresolved feedback and recent feedback
        self.feedbacks = [f for f in self.feedbacks
                         if f.created_at >= cutoff_date or
                         f.status not in [FeedbackStatus.RESOLVED, FeedbackStatus.CLOSED]]

        # Clean up old analyses
        self.analyses = [a for a in self.analyses if a.created_at >= cutoff_date]

        self._update_stats()
        self.logger.info(f"Cleaned up old feedback data from {max_age_days} days ago")