#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Assistant Unified Interface
Integrates proactive suggestions, workflow automation and intelligent reminder functions
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

from ..storage import MemoryRepository
from .proactive_advisor import ProactiveAdvisor, Suggestion, SuggestionMoment, Advice
from .workflow_automator import WorkflowAutomator, Pattern, WorkflowTemplate, ExecutionResult
from .reminder_system import ReminderSystem, Reminder, ReminderType, ReminderPriority


class AssistantMode(Enum):
    """Assistant mode enumeration"""
    PASSIVE = "passive"      # Passive mode: only respond to requests
    PROACTIVE = "proactive"  # Proactive mode: actively provide suggestions
    AUTOMATED = "automated"  # Automated mode: automatically execute workflows


class DecisionType(Enum):
    """Decision type enumeration"""
    SUGGESTION = "suggestion"    # Suggestion decision
    WORKFLOW = "workflow"        # Workflow decision
    REMINDER = "reminder"        # Reminder decision
    COORDINATION = "coordination" # Coordination decision


@dataclass
class AssistantContext:
    """Assistant context"""
    project_path: str
    user_query: Optional[str] = None
    recent_activities: List[Dict[str, Any]] = field(default_factory=list)
    project_metrics: Dict[str, Any] = field(default_factory=dict)
    error_context: Dict[str, Any] = field(default_factory=dict)
    time_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantDecision:
    """Assistant decision"""
    decision_type: DecisionType
    priority: int  # 1-10, 10 is highest priority
    confidence: float  # 0.0-1.0
    action_type: str
    action_data: Dict[str, Any]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AssistantResponse:
    """Assistant response"""
    suggestions: List[Suggestion] = field(default_factory=list)
    workflows: List[WorkflowTemplate] = field(default_factory=list)
    reminders: List[Reminder] = field(default_factory=list)
    decisions: List[AssistantDecision] = field(default_factory=list)
    coordination_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantConfig:
    """Assistant configuration"""
    mode: AssistantMode = AssistantMode.PROACTIVE
    max_suggestions: int = 5
    max_workflows: int = 3
    max_reminders: int = 10
    suggestion_threshold: float = 0.6
    workflow_threshold: float = 0.7
    reminder_threshold: float = 0.5
    auto_execute_workflows: bool = False
    proactive_interval: int = 300  # seconds
    enable_coordination: bool = True
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "suggestion": 0.3,
        "workflow": 0.4,
        "reminder": 0.3
    })


class IntelligentAssistant:
    """Intelligent Assistant Unified Interface"""

    def __init__(self, memory_repo: MemoryRepository, config: Optional[AssistantConfig] = None):
        """
        Initialize intelligent assistant

        Args:
            memory_repo: Memory repository instance
            config: Assistant configuration
        """
        self.memory_repo = memory_repo
        self.config = config or AssistantConfig()
        self.logger = logging.getLogger('hibro.intelligent_assistant')

        # Initialize sub-modules
        self.proactive_advisor = ProactiveAdvisor(memory_repo)
        self.workflow_automator = WorkflowAutomator(memory_repo)
        self.reminder_system = ReminderSystem(memory_repo)

        # Decision history
        self.decision_history: List[AssistantDecision] = []
        self.coordination_history: List[Dict[str, Any]] = []

        # Performance statistics
        self.performance_stats = {
            "total_requests": 0,
            "successful_suggestions": 0,
            "executed_workflows": 0,
            "active_reminders": 0,
            "coordination_actions": 0
        }

    def process_request(self, context: AssistantContext) -> AssistantResponse:
        """
        Process user request

        Args:
            context: Assistant context

        Returns:
            Assistant response
        """
        self.performance_stats["total_requests"] += 1

        try:
            # Build unified context
            unified_context = self._build_unified_context(context)

            # Execute decision engine
            decisions = self._make_decisions(unified_context)

            # Generate response
            response = self._generate_response(decisions, unified_context)

            # Execute coordination actions
            if self.config.enable_coordination:
                coordination_actions = self._coordinate_modules(response, unified_context)
                response.coordination_actions = coordination_actions

            # Record decision history
            self.decision_history.extend(decisions)

            self.logger.info(f"Request processing completed: {len(response.suggestions)} suggestions, "
                           f"{len(response.workflows)} workflows, {len(response.reminders)} reminders")

            return response

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            return AssistantResponse(metadata={"error": str(e)})

    def _build_unified_context(self, context: AssistantContext) -> Dict[str, Any]:
        """Build unified context"""
        unified_context = {
            "project_path": context.project_path,
            "user_query": context.user_query,
            "recent_activities": context.recent_activities,
            "project_metrics": context.project_metrics,
            "error_context": context.error_context,
            "time_context": context.time_context,
            "user_preferences": context.user_preferences,

            # Add assistant configuration
            "assistant_config": {
                "mode": self.config.mode.value,
                "max_suggestions": self.config.max_suggestions,
                "max_workflows": self.config.max_workflows,
                "max_reminders": self.config.max_reminders,
                "thresholds": {
                    "suggestion": self.config.suggestion_threshold,
                    "workflow": self.config.workflow_threshold,
                    "reminder": self.config.reminder_threshold
                }
            },

            # Add historical information
            "recent_decisions": self.decision_history[-10:] if self.decision_history else [],
            "performance_stats": self.performance_stats.copy()
        }

        return unified_context

    def _make_decisions(self, context: Dict[str, Any]) -> List[AssistantDecision]:
        """Execute decision engine"""
        decisions = []

        try:
            # Suggestion decisions
            suggestion_decisions = self._make_suggestion_decisions(context)
            decisions.extend(suggestion_decisions)

            # Workflow decisions
            workflow_decisions = self._make_workflow_decisions(context)
            decisions.extend(workflow_decisions)

            # Reminder decisions
            reminder_decisions = self._make_reminder_decisions(context)
            decisions.extend(reminder_decisions)

            # Coordination decisions
            if self.config.enable_coordination:
                coordination_decisions = self._make_coordination_decisions(context, decisions)
                decisions.extend(coordination_decisions)

            # Sort by priority
            decisions.sort(key=lambda d: (d.priority, d.confidence), reverse=True)

            self.logger.info(f"Generated {len(decisions)} decisions")

        except Exception as e:
            self.logger.error(f"Decision generation failed: {e}")

        return decisions

    def _make_suggestion_decisions(self, context: Dict[str, Any]) -> List[AssistantDecision]:
        """Generate suggestion decisions"""
        decisions = []

        try:
            # Detect suggestion moments
            suggestion_moments = self.proactive_advisor.detect_suggestion_moments(context)

            for moment in suggestion_moments:
                # Calculate decision priority and confidence
                priority = self._calculate_suggestion_priority(moment, context)
                confidence = self._calculate_suggestion_confidence(moment, context)

                if confidence >= self.config.suggestion_threshold:
                    decision = AssistantDecision(
                        decision_type=DecisionType.SUGGESTION,
                        priority=priority,
                        confidence=confidence,
                        action_type="generate_suggestion",
                        action_data={
                            "moment": moment,
                            "context": context
                        },
                        reasoning=f"Detected {moment.moment_type.value} moment, confidence {confidence:.2f}"
                    )
                    decisions.append(decision)

        except Exception as e:
            self.logger.error(f"Suggestion decision generation failed: {e}")

        return decisions

    def _make_workflow_decisions(self, context: Dict[str, Any]) -> List[AssistantDecision]:
        """Generate workflow decisions"""
        decisions = []

        try:
            # Detect workflow patterns
            patterns = self.workflow_automator.detect_patterns(context)

            for pattern in patterns:
                # Calculate decision priority and confidence
                priority = self._calculate_workflow_priority(pattern, context)
                confidence = self._calculate_workflow_confidence(pattern, context)

                if confidence >= self.config.workflow_threshold:
                    decision = AssistantDecision(
                        decision_type=DecisionType.WORKFLOW,
                        priority=priority,
                        confidence=confidence,
                        action_type="recommend_workflow",
                        action_data={
                            "pattern": pattern,
                            "context": context
                        },
                        reasoning=f"Detected {pattern.pattern_type.value} pattern, confidence {confidence:.2f}"
                    )
                    decisions.append(decision)

        except Exception as e:
            self.logger.error(f"Workflow decision generation failed: {e}")

        return decisions

    def _make_reminder_decisions(self, context: Dict[str, Any]) -> List[AssistantDecision]:
        """Generate reminder decisions"""
        decisions = []

        try:
            # Detect reminder moments
            reminder_types = self.reminder_system.detect_reminder_moments(context)

            for reminder_type in reminder_types:
                # Calculate decision priority and confidence
                priority = self._calculate_reminder_priority(reminder_type, context)
                confidence = self._calculate_reminder_confidence(reminder_type, context)

                if confidence >= self.config.reminder_threshold:
                    decision = AssistantDecision(
                        decision_type=DecisionType.REMINDER,
                        priority=priority,
                        confidence=confidence,
                        action_type="create_reminder",
                        action_data={
                            "reminder_type": reminder_type,
                            "context": context
                        },
                        reasoning=f"Detected {reminder_type.value} reminder moment, confidence {confidence:.2f}"
                    )
                    decisions.append(decision)

        except Exception as e:
            self.logger.error(f"Reminder decision generation failed: {e}")

        return decisions

    def _make_coordination_decisions(self, context: Dict[str, Any],
                                   existing_decisions: List[AssistantDecision]) -> List[AssistantDecision]:
        """Generate coordination decisions"""
        decisions = []

        try:
            # Analyze decision conflicts
            conflicts = self._detect_decision_conflicts(existing_decisions)

            for conflict in conflicts:
                decision = AssistantDecision(
                    decision_type=DecisionType.COORDINATION,
                    priority=8,  # Coordination decisions have higher priority
                    confidence=0.9,
                    action_type="resolve_conflict",
                    action_data={
                        "conflict": conflict,
                        "resolution_strategy": self._determine_resolution_strategy(conflict)
                    },
                    reasoning=f"Detected decision conflict: {conflict['type']}"
                )
                decisions.append(decision)

            # Analyze synergy opportunities
            synergies = self._detect_synergies(existing_decisions)

            for synergy in synergies:
                decision = AssistantDecision(
                    decision_type=DecisionType.COORDINATION,
                    priority=6,
                    confidence=0.8,
                    action_type="create_synergy",
                    action_data={
                        "synergy": synergy,
                        "enhancement_strategy": self._determine_enhancement_strategy(synergy)
                    },
                    reasoning=f"Detected synergy opportunity: {synergy['type']}"
                )
                decisions.append(decision)

        except Exception as e:
            self.logger.error(f"Coordination decision generation failed: {e}")

        return decisions

    def _generate_response(self, decisions: List[AssistantDecision],
                          context: Dict[str, Any]) -> AssistantResponse:
        """Generate assistant response"""
        response = AssistantResponse()

        try:
            # Execute decisions
            for decision in decisions:
                if decision.decision_type == DecisionType.SUGGESTION:
                    suggestions = self._execute_suggestion_decision(decision, context)
                    response.suggestions.extend(suggestions)

                elif decision.decision_type == DecisionType.WORKFLOW:
                    workflows = self._execute_workflow_decision(decision, context)
                    response.workflows.extend(workflows)

                elif decision.decision_type == DecisionType.REMINDER:
                    reminders = self._execute_reminder_decision(decision, context)
                    response.reminders.extend(reminders)

            # Apply quantity limits
            response.suggestions = response.suggestions[:self.config.max_suggestions]
            response.workflows = response.workflows[:self.config.max_workflows]
            response.reminders = response.reminders[:self.config.max_reminders]

            # Add decision information
            response.decisions = decisions
            response.metadata = {
                "total_decisions": len(decisions),
                "execution_time": datetime.now().isoformat(),
                "assistant_mode": self.config.mode.value
            }

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            response.metadata["error"] = str(e)

        return response

    def _coordinate_modules(self, response: AssistantResponse,
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate inter-module interactions"""
        coordination_actions = []

        try:
            # Coordinate suggestions and workflows
            suggestion_workflow_actions = self._coordinate_suggestions_workflows(
                response.suggestions, response.workflows, context
            )
            coordination_actions.extend(suggestion_workflow_actions)

            # Coordinate workflows and reminders
            workflow_reminder_actions = self._coordinate_workflows_reminders(
                response.workflows, response.reminders, context
            )
            coordination_actions.extend(workflow_reminder_actions)

            # Coordinate suggestions and reminders
            suggestion_reminder_actions = self._coordinate_suggestions_reminders(
                response.suggestions, response.reminders, context
            )
            coordination_actions.extend(suggestion_reminder_actions)

            self.performance_stats["coordination_actions"] += len(coordination_actions)

        except Exception as e:
            self.logger.error(f"Module coordination failed: {e}")

        return coordination_actions

    def _calculate_suggestion_priority(self, moment: SuggestionMoment, context: Dict[str, Any]) -> int:
        """Calculate suggestion priority"""
        base_priority = 5

        # Adjust based on moment type
        moment_priorities = {
            "project_initialization": 8,
            "phase_transition": 7,
            "problem_pattern": 9,
            "technical_decision": 6,
            "best_practice_opportunity": 5,
            "knowledge_gap": 4
        }

        priority = moment_priorities.get(moment.moment_type.value, base_priority)

        # Adjust based on error context
        if context.get("error_context", {}).get("has_errors", False):
            priority += 2

        return min(priority, 10)

    def _calculate_suggestion_confidence(self, moment: SuggestionMoment, context: Dict[str, Any]) -> float:
        """Calculate suggestion confidence"""
        base_confidence = 0.7

        # Adjust based on historical success rate
        historical_success = self._get_historical_success_rate("suggestion", moment.moment_type.value)
        confidence = base_confidence * (0.5 + 0.5 * historical_success)

        return min(confidence, 1.0)

    def _calculate_workflow_priority(self, pattern: Pattern, context: Dict[str, Any]) -> int:
        """Calculate workflow priority"""
        base_priority = 6

        # Adjust based on pattern frequency
        if pattern.frequency > 5:
            base_priority += 2
        elif pattern.frequency > 3:
            base_priority += 1

        return min(base_priority, 10)

    def _calculate_workflow_confidence(self, pattern: Pattern, context: Dict[str, Any]) -> float:
        """Calculate workflow confidence"""
        return min(pattern.confidence, 1.0)

    def _calculate_reminder_priority(self, reminder_type: ReminderType, context: Dict[str, Any]) -> int:
        """Calculate reminder priority"""
        priority_map = {
            ReminderType.SECURITY: 10,
            ReminderType.DEADLINE: 8,
            ReminderType.MILESTONE: 7,
            ReminderType.TECH_DEBT: 5,
            ReminderType.BEST_PRACTICE: 4,
            ReminderType.MAINTENANCE: 3
        }

        return priority_map.get(reminder_type, 5)

    def _calculate_reminder_confidence(self, reminder_type: ReminderType, context: Dict[str, Any]) -> float:
        """Calculate reminder confidence"""
        # Calculate confidence based on context data quality
        base_confidence = 0.8

        if reminder_type == ReminderType.SECURITY:
            security_issues = context.get("security_issues", [])
            if security_issues:
                base_confidence = 0.95

        elif reminder_type == ReminderType.MILESTONE:
            milestone_data = context.get("milestone_data", {})
            if milestone_data.get("target_date"):
                base_confidence = 0.9

        return base_confidence

    def _get_historical_success_rate(self, decision_type: str, subtype: str) -> float:
        """Get historical success rate"""
        # Simplified implementation, should actually get from database
        return 0.75

    def _detect_decision_conflicts(self, decisions: List[AssistantDecision]) -> List[Dict[str, Any]]:
        """Detect decision conflicts"""
        conflicts = []

        # Detect resource conflicts
        workflow_decisions = [d for d in decisions if d.decision_type == DecisionType.WORKFLOW]
        if len(workflow_decisions) > 1:
            conflicts.append({
                "type": "resource_conflict",
                "decisions": workflow_decisions,
                "description": "Multiple workflows may compete for the same resources"
            })

        return conflicts

    def _detect_synergies(self, decisions: List[AssistantDecision]) -> List[Dict[str, Any]]:
        """Detect synergy opportunities"""
        synergies = []

        suggestion_decisions = [d for d in decisions if d.decision_type == DecisionType.SUGGESTION]
        workflow_decisions = [d for d in decisions if d.decision_type == DecisionType.WORKFLOW]

        if suggestion_decisions and workflow_decisions:
            synergies.append({
                "type": "suggestion_workflow_synergy",
                "decisions": suggestion_decisions + workflow_decisions,
                "description": "Suggestions can be automated through workflows"
            })

        return synergies

    def _determine_resolution_strategy(self, conflict: Dict[str, Any]) -> str:
        """Determine conflict resolution strategy"""
        if conflict["type"] == "resource_conflict":
            return "prioritize_by_confidence"
        return "default"

    def _determine_enhancement_strategy(self, synergy: Dict[str, Any]) -> str:
        """Determine enhancement strategy"""
        if synergy["type"] == "suggestion_workflow_synergy":
            return "combine_suggestion_with_workflow"
        return "default"

    def _execute_suggestion_decision(self, decision: AssistantDecision,
                                   context: Dict[str, Any]) -> List[Suggestion]:
        """Execute suggestion decision"""
        try:
            moment = decision.action_data["moment"]
            suggestions = self.proactive_advisor.generate_contextual_advice(moment, context)
            self.performance_stats["successful_suggestions"] += len(suggestions)
            return suggestions
        except Exception as e:
            self.logger.error(f"Suggestion decision execution failed: {e}")
            return []

    def _execute_workflow_decision(self, decision: AssistantDecision,
                                 context: Dict[str, Any]) -> List[WorkflowTemplate]:
        """Execute workflow decision"""
        try:
            workflows = self.workflow_automator.recommend_workflows(context)
            return workflows
        except Exception as e:
            self.logger.error(f"Workflow decision execution failed: {e}")
            return []

    def _execute_reminder_decision(self, decision: AssistantDecision,
                                 context: Dict[str, Any]) -> List[Reminder]:
        """Execute reminder decision"""
        try:
            reminders = self.reminder_system.generate_contextual_reminders(context)
            self.performance_stats["active_reminders"] += len(reminders)
            return reminders
        except Exception as e:
            self.logger.error(f"Reminder decision execution failed: {e}")
            return []

    def _coordinate_suggestions_workflows(self, suggestions: List[Suggestion],
                                        workflows: List[WorkflowTemplate],
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate suggestions and workflows"""
        actions = []

        for suggestion in suggestions:
            for workflow in workflows:
                # Check if suggestion can be automated through workflow
                if self._can_automate_suggestion(suggestion, workflow):
                    actions.append({
                        "type": "automate_suggestion",
                        "suggestion_id": suggestion.id,
                        "workflow_id": workflow.id,
                        "description": f"Suggestion '{suggestion.advice.title}' can be automated through workflow '{workflow.name}'"
                    })

        return actions

    def _coordinate_workflows_reminders(self, workflows: List[WorkflowTemplate],
                                      reminders: List[Reminder],
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate workflows and reminders"""
        actions = []

        for workflow in workflows:
            for reminder in reminders:
                # Check if workflow can resolve reminder issue
                if self._can_resolve_reminder(workflow, reminder):
                    actions.append({
                        "type": "resolve_reminder_with_workflow",
                        "workflow_id": workflow.id,
                        "reminder_id": reminder.id,
                        "description": f"Workflow '{workflow.name}' can resolve reminder '{reminder.title}'"
                    })

        return actions

    def _coordinate_suggestions_reminders(self, suggestions: List[Suggestion],
                                        reminders: List[Reminder],
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate suggestions and reminders"""
        actions = []

        for suggestion in suggestions:
            for reminder in reminders:
                # Check if suggestion is related to reminder
                if self._is_suggestion_related_to_reminder(suggestion, reminder):
                    actions.append({
                        "type": "link_suggestion_reminder",
                        "suggestion_id": suggestion.id,
                        "reminder_id": reminder.id,
                        "description": f"Suggestion '{suggestion.advice.title}' is related to reminder '{reminder.title}'"
                    })

        return actions

    def _can_automate_suggestion(self, suggestion: Suggestion, workflow: WorkflowTemplate) -> bool:
        """Check if suggestion can be automated"""
        # Simplified implementation, should have more complex matching logic in practice
        suggestion_keywords = suggestion.advice.title.lower().split()
        workflow_keywords = workflow.name.lower().split()

        return len(set(suggestion_keywords) & set(workflow_keywords)) > 0

    def _can_resolve_reminder(self, workflow: WorkflowTemplate, reminder: Reminder) -> bool:
        """Check if workflow can resolve reminder"""
        # Simplified implementation
        return "test" in workflow.name.lower() and reminder.reminder_type == ReminderType.BEST_PRACTICE

    def _is_suggestion_related_to_reminder(self, suggestion: Suggestion, reminder: Reminder) -> bool:
        """Check if suggestion is related to reminder"""
        # Simplified implementation
        suggestion_keywords = suggestion.advice.title.lower().split()
        reminder_keywords = reminder.title.lower().split()

        return len(set(suggestion_keywords) & set(reminder_keywords)) > 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        try:
            # Get sub-module statistics
            suggestion_stats = self.proactive_advisor.get_statistics()
            workflow_stats = self.workflow_automator.get_statistics()
            reminder_stats = self.reminder_system.get_statistics()

            return {
                "performance": self.performance_stats.copy(),
                "configuration": {
                    "mode": self.config.mode.value,
                    "thresholds": {
                        "suggestion": self.config.suggestion_threshold,
                        "workflow": self.config.workflow_threshold,
                        "reminder": self.config.reminder_threshold
                    }
                },
                "modules": {
                    "suggestions": suggestion_stats,
                    "workflows": workflow_stats,
                    "reminders": reminder_stats
                },
                "decisions": {
                    "total_decisions": len(self.decision_history),
                    "decision_types": self._get_decision_type_distribution(),
                    "avg_confidence": self._get_average_confidence(),
                    "coordination_actions": len(self.coordination_history)
                }
            }

        except Exception as e:
            self.logger.error(f"Get statistics failed: {e}")
            return {}

    def _get_decision_type_distribution(self) -> Dict[str, int]:
        """Get decision type distribution"""
        distribution = {}
        for decision in self.decision_history:
            dtype = decision.decision_type.value
            distribution[dtype] = distribution.get(dtype, 0) + 1
        return distribution

    def _get_average_confidence(self) -> float:
        """Get average confidence"""
        if not self.decision_history:
            return 0.0

        total_confidence = sum(d.confidence for d in self.decision_history)
        return round(total_confidence / len(self.decision_history), 3)

    def update_config(self, new_config: AssistantConfig):
        """Update assistant configuration"""
        self.config = new_config
        self.logger.info("Assistant configuration updated")

    def reset_statistics(self):
        """Reset statistics"""
        self.performance_stats = {
            "total_requests": 0,
            "successful_suggestions": 0,
            "executed_workflows": 0,
            "active_reminders": 0,
            "coordination_actions": 0
        }
        self.decision_history.clear()
        self.coordination_history.clear()
        self.logger.info("Statistics reset")