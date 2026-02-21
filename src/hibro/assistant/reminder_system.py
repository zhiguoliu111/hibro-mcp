#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Reminder System
Intelligent reminder functionality based on project progress and historical experience
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

from ..storage import MemoryRepository


class ReminderType(Enum):
    """Reminder type enumeration"""
    MILESTONE = "milestone"          # Project milestone reminder
    TECH_DEBT = "tech_debt"         # Technical debt warning
    BEST_PRACTICE = "best_practice"  # Best practice recommendation
    DEADLINE = "deadline"           # Deadline reminder
    MAINTENANCE = "maintenance"     # Maintenance reminder
    SECURITY = "security"           # Security reminder


class ReminderPriority(Enum):
    """Reminder priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReminderStatus(Enum):
    """Reminder status enumeration"""
    PENDING = "pending"      # Pending
    ACTIVE = "active"        # Active
    DISMISSED = "dismissed"  # Dismissed
    COMPLETED = "completed"  # Completed


@dataclass
class MilestoneContext:
    """Milestone context"""
    project_path: str
    milestone_name: str
    target_date: datetime
    completion_percentage: float
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


@dataclass
class TechDebtContext:
    """Technical debt context"""
    project_path: str
    debt_type: str  # code_quality, performance, security, documentation
    severity: str   # low, medium, high, critical
    affected_files: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None
    impact_description: str = ""


@dataclass
class BestPracticeContext:
    """Best practice context"""
    project_path: str
    practice_category: str  # testing, documentation, code_review, deployment
    current_state: str
    recommended_action: str
    benefits: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)


@dataclass
class Reminder:
    """Reminder object"""
    id: str
    reminder_type: ReminderType
    priority: ReminderPriority
    title: str
    description: str
    context: Dict[str, Any]
    created_at: datetime
    scheduled_at: datetime
    status: ReminderStatus = ReminderStatus.PENDING
    project_path: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReminderRule:
    """Reminder rule"""
    id: str
    name: str
    reminder_type: ReminderType
    conditions: Dict[str, Any]  # Trigger conditions
    schedule_pattern: str       # Schedule pattern: once, daily, weekly, monthly
    priority: ReminderPriority
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class ReminderSystem:
    """Intelligent reminder system"""

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize reminder system

        Args:
            memory_repo: Memory repository instance
        """
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.reminder_system')

        # Reminder storage
        self.active_reminders: Dict[str, Reminder] = {}
        self.reminder_rules: Dict[str, ReminderRule] = {}
        self.reminder_history: List[Reminder] = []

        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default reminder rules"""
        default_rules = [
            # Project milestone rules
            ReminderRule(
                id="milestone_approaching",
                name="Milestone approaching reminder",
                reminder_type=ReminderType.MILESTONE,
                conditions={
                    "days_before_deadline": 7,
                    "completion_threshold": 0.8
                },
                schedule_pattern="daily",
                priority=ReminderPriority.HIGH
            ),

            # Technical debt rules
            ReminderRule(
                id="tech_debt_accumulation",
                name="Technical debt accumulation warning",
                reminder_type=ReminderType.TECH_DEBT,
                conditions={
                    "debt_score_threshold": 0.7,
                    "files_affected_threshold": 5
                },
                schedule_pattern="weekly",
                priority=ReminderPriority.MEDIUM
            ),

            # Best practice rules
            ReminderRule(
                id="testing_coverage",
                name="Testing coverage reminder",
                reminder_type=ReminderType.BEST_PRACTICE,
                conditions={
                    "coverage_threshold": 0.8,
                    "new_code_lines": 100
                },
                schedule_pattern="weekly",
                priority=ReminderPriority.MEDIUM
            ),

            # Security reminder rules
            ReminderRule(
                id="security_update",
                name="Security update reminder",
                reminder_type=ReminderType.SECURITY,
                conditions={
                    "vulnerability_severity": "high",
                    "days_since_discovery": 3
                },
                schedule_pattern="daily",
                priority=ReminderPriority.CRITICAL
            )
        ]

        for rule in default_rules:
            self.reminder_rules[rule.id] = rule

    def detect_reminder_moments(self, context: Dict[str, Any]) -> List[ReminderType]:
        """
        Detect reminder moments

        Args:
            context: Context information including:
                - project_path: Project path
                - project_metrics: Project metrics
                - recent_activities: Recent activities
                - milestone_data: Milestone data
                - code_quality_metrics: Code quality metrics

        Returns:
            List of reminder types to trigger
        """
        triggered_types = []

        try:
            project_path = context.get('project_path', '')
            project_metrics = context.get('project_metrics', {})
            milestone_data = context.get('milestone_data', {})
            code_quality = context.get('code_quality_metrics', {})

            # Detect milestone reminders
            if self._should_trigger_milestone_reminder(milestone_data):
                triggered_types.append(ReminderType.MILESTONE)

            # Detect technical debt reminders
            if self._should_trigger_tech_debt_reminder(code_quality):
                triggered_types.append(ReminderType.TECH_DEBT)

            # Detect best practice reminders
            if self._should_trigger_best_practice_reminder(project_metrics):
                triggered_types.append(ReminderType.BEST_PRACTICE)

            # Detect security reminders
            if self._should_trigger_security_reminder(context):
                triggered_types.append(ReminderType.SECURITY)

            self.logger.info(f"Detected {len(triggered_types)} reminder moments: {[t.value for t in triggered_types]}")

        except Exception as e:
            self.logger.error(f"Reminder moment detection failed: {e}")

        return triggered_types

    def _should_trigger_milestone_reminder(self, milestone_data: Dict[str, Any]) -> bool:
        """Check if milestone reminder should be triggered"""
        if not milestone_data:
            return False

        target_date = milestone_data.get('target_date')
        completion = milestone_data.get('completion_percentage', 0)

        if not target_date:
            return False

        # Parse target date
        if isinstance(target_date, str):
            try:
                target_date = datetime.fromisoformat(target_date)
            except ValueError:
                return False

        # Check if approaching deadline and completion is insufficient
        days_until_deadline = (target_date - datetime.now()).days
        return days_until_deadline <= 7 and completion < 0.8

    def _should_trigger_tech_debt_reminder(self, code_quality: Dict[str, Any]) -> bool:
        """Check if technical debt reminder should be triggered"""
        if not code_quality:
            return False

        debt_score = code_quality.get('debt_score', 0)
        affected_files = code_quality.get('affected_files', 0)

        return debt_score > 0.7 or affected_files > 5

    def _should_trigger_best_practice_reminder(self, project_metrics: Dict[str, Any]) -> bool:
        """Check if best practice reminder should be triggered"""
        if not project_metrics:
            return False

        test_coverage = project_metrics.get('test_coverage', 1.0)
        new_code_lines = project_metrics.get('new_code_lines', 0)

        return test_coverage < 0.8 and new_code_lines > 100

    def _should_trigger_security_reminder(self, context: Dict[str, Any]) -> bool:
        """Check if security reminder should be triggered"""
        security_issues = context.get('security_issues', [])

        for issue in security_issues:
            severity = issue.get('severity', 'low')
            days_since = issue.get('days_since_discovery', 0)

            if severity == 'critical' or (severity == 'high' and days_since > 3):
                return True

        return False

    def generate_contextual_reminders(self, context: Dict[str, Any]) -> List[Reminder]:
        """
        Generate contextual reminders

        Args:
            context: Context information

        Returns:
            List of generated reminders
        """
        reminders = []

        try:
            triggered_types = self.detect_reminder_moments(context)

            for reminder_type in triggered_types:
                reminder = self._create_reminder_for_type(reminder_type, context)
                if reminder:
                    reminders.append(reminder)

            # Sort by priority
            reminders.sort(key=lambda r: self._get_priority_score(r.priority), reverse=True)

            self.logger.info(f"Generated {len(reminders)} contextual reminders")

        except Exception as e:
            self.logger.error(f"Contextual reminder generation failed: {e}")

        return reminders

    def _create_reminder_for_type(self, reminder_type: ReminderType, context: Dict[str, Any]) -> Optional[Reminder]:
        """Create reminder for specific type"""
        project_path = context.get('project_path', '')
        now = datetime.now()

        if reminder_type == ReminderType.MILESTONE:
            return self._create_milestone_reminder(context, now)
        elif reminder_type == ReminderType.TECH_DEBT:
            return self._create_tech_debt_reminder(context, now)
        elif reminder_type == ReminderType.BEST_PRACTICE:
            return self._create_best_practice_reminder(context, now)
        elif reminder_type == ReminderType.SECURITY:
            return self._create_security_reminder(context, now)

        return None

    def _create_milestone_reminder(self, context: Dict[str, Any], now: datetime) -> Optional[Reminder]:
        """Create milestone reminder"""
        milestone_data = context.get('milestone_data', {})
        if not milestone_data:
            return None

        milestone_name = milestone_data.get('name', 'Unnamed milestone')
        target_date = milestone_data.get('target_date')
        completion = milestone_data.get('completion_percentage', 0)

        if isinstance(target_date, str):
            try:
                target_date = datetime.fromisoformat(target_date)
            except ValueError:
                target_date = now + timedelta(days=7)

        days_left = (target_date - now).days

        return Reminder(
            id=f"milestone_{milestone_name}_{now.timestamp()}",
            reminder_type=ReminderType.MILESTONE,
            priority=ReminderPriority.HIGH,
            title=f"Milestone '{milestone_name}' approaching deadline",
            description=f"{days_left} days until milestone deadline, current completion {completion*100:.1f}%",
            context={
                "milestone_name": milestone_name,
                "target_date": target_date.isoformat(),
                "completion_percentage": completion,
                "days_remaining": days_left
            },
            created_at=now,
            scheduled_at=now,
            project_path=context.get('project_path'),
            tags={"milestone", "deadline"}
        )

    def _create_tech_debt_reminder(self, context: Dict[str, Any], now: datetime) -> Optional[Reminder]:
        """Create technical debt reminder"""
        code_quality = context.get('code_quality_metrics', {})
        if not code_quality:
            return None

        debt_score = code_quality.get('debt_score', 0)
        affected_files = code_quality.get('affected_files', 0)

        return Reminder(
            id=f"tech_debt_{now.timestamp()}",
            reminder_type=ReminderType.TECH_DEBT,
            priority=ReminderPriority.MEDIUM,
            title="Technical debt accumulation warning",
            description=f"Project technical debt score {debt_score:.2f}, affecting {affected_files} files",
            context={
                "debt_score": debt_score,
                "affected_files": affected_files,
                "debt_details": code_quality.get('debt_details', [])
            },
            created_at=now,
            scheduled_at=now,
            project_path=context.get('project_path'),
            tags={"tech_debt", "code_quality"}
        )

    def _create_best_practice_reminder(self, context: Dict[str, Any], now: datetime) -> Optional[Reminder]:
        """Create best practice reminder"""
        project_metrics = context.get('project_metrics', {})
        if not project_metrics:
            return None

        test_coverage = project_metrics.get('test_coverage', 1.0)
        new_code_lines = project_metrics.get('new_code_lines', 0)

        return Reminder(
            id=f"best_practice_{now.timestamp()}",
            reminder_type=ReminderType.BEST_PRACTICE,
            priority=ReminderPriority.MEDIUM,
            title="Insufficient test coverage",
            description=f"Current test coverage {test_coverage*100:.1f}%, new code {new_code_lines} lines",
            context={
                "test_coverage": test_coverage,
                "new_code_lines": new_code_lines,
                "recommended_coverage": 0.8
            },
            created_at=now,
            scheduled_at=now,
            project_path=context.get('project_path'),
            tags={"best_practice", "testing"}
        )

    def _create_security_reminder(self, context: Dict[str, Any], now: datetime) -> Optional[Reminder]:
        """Create security reminder"""
        security_issues = context.get('security_issues', [])
        if not security_issues:
            return None

        critical_issues = [i for i in security_issues if i.get('severity') == 'critical']
        high_issues = [i for i in security_issues if i.get('severity') == 'high']

        if critical_issues:
            issue = critical_issues[0]
            priority = ReminderPriority.CRITICAL
            title = f"Critical security vulnerability: {issue.get('title', 'Unknown vulnerability')}"
        elif high_issues:
            issue = high_issues[0]
            priority = ReminderPriority.HIGH
            title = f"High severity security vulnerability: {issue.get('title', 'Unknown vulnerability')}"
        else:
            return None

        return Reminder(
            id=f"security_{issue.get('id', now.timestamp())}",
            reminder_type=ReminderType.SECURITY,
            priority=priority,
            title=title,
            description=issue.get('description', 'Please address security vulnerabilities promptly'),
            context={
                "issue_id": issue.get('id'),
                "severity": issue.get('severity'),
                "days_since_discovery": issue.get('days_since_discovery', 0)
            },
            created_at=now,
            scheduled_at=now,
            project_path=context.get('project_path'),
            tags={"security", "vulnerability"}
        )

    def _get_priority_score(self, priority: ReminderPriority) -> int:
        """Get priority score"""
        priority_scores = {
            ReminderPriority.LOW: 1,
            ReminderPriority.MEDIUM: 2,
            ReminderPriority.HIGH: 3,
            ReminderPriority.CRITICAL: 4
        }
        return priority_scores.get(priority, 1)

    def schedule_reminder(self, reminder: Reminder) -> bool:
        """
        Schedule reminder

        Args:
            reminder: Reminder object

        Returns:
            Whether scheduling was successful
        """
        try:
            # Check if identical reminder already exists
            if reminder.id in self.active_reminders:
                self.logger.warning(f"Reminder {reminder.id} already exists")
                return False

            # Add to active reminder list
            self.active_reminders[reminder.id] = reminder

            # Record to history
            self.reminder_history.append(reminder)

            self.logger.info(f"Reminder scheduled: {reminder.title}")
            return True

        except Exception as e:
            self.logger.error(f"Reminder scheduling failed: {e}")
            return False

    def dismiss_reminder(self, reminder_id: str) -> bool:
        """
        Dismiss reminder

        Args:
            reminder_id: Reminder ID

        Returns:
            Whether the operation was successful
        """
        try:
            if reminder_id in self.active_reminders:
                reminder = self.active_reminders[reminder_id]
                reminder.status = ReminderStatus.DISMISSED
                del self.active_reminders[reminder_id]

                self.logger.info(f"Reminder dismissed: {reminder.title}")
                return True
            else:
                self.logger.warning(f"Reminder {reminder_id} does not exist")
                return False

        except Exception as e:
            self.logger.error(f"Failed to dismiss reminder: {e}")
            return False

    def complete_reminder(self, reminder_id: str) -> bool:
        """
        Complete reminder

        Args:
            reminder_id: Reminder ID

        Returns:
            Whether the operation was successful
        """
        try:
            if reminder_id in self.active_reminders:
                reminder = self.active_reminders[reminder_id]
                reminder.status = ReminderStatus.COMPLETED
                del self.active_reminders[reminder_id]

                self.logger.info(f"Reminder completed: {reminder.title}")
                return True
            else:
                self.logger.warning(f"Reminder {reminder_id} does not exist")
                return False

        except Exception as e:
            self.logger.error(f"Failed to complete reminder: {e}")
            return False

    def get_active_reminders(self, project_path: Optional[str] = None,
                           reminder_type: Optional[ReminderType] = None,
                           priority: Optional[ReminderPriority] = None) -> List[Reminder]:
        """
        Get active reminders

        Args:
            project_path: Project path filter
            reminder_type: Reminder type filter
            priority: Priority filter

        Returns:
            Filtered reminder list
        """
        reminders = list(self.active_reminders.values())

        # Apply filter conditions
        if project_path:
            reminders = [r for r in reminders if r.project_path == project_path]

        if reminder_type:
            reminders = [r for r in reminders if r.reminder_type == reminder_type]

        if priority:
            reminders = [r for r in reminders if r.priority == priority]

        # Sort by priority and creation time
        reminders.sort(key=lambda r: (
            self._get_priority_score(r.priority),
            r.created_at.timestamp()
        ), reverse=True)

        return reminders

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get reminder system statistics

        Returns:
            Statistics dictionary
        """
        try:
            total_reminders = len(self.reminder_history)
            active_count = len(self.active_reminders)

            if total_reminders == 0:
                return {
                    "total_reminders": 0,
                    "active_reminders": 0,
                    "completion_rate": 0.0,
                    "reminder_types": {},
                    "priority_distribution": {}
                }

            # Calculate status distribution
            status_counts = {}
            type_counts = {}
            priority_counts = {}

            for reminder in self.reminder_history:
                # Status statistics
                status = reminder.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

                # Type statistics
                rtype = reminder.reminder_type.value
                type_counts[rtype] = type_counts.get(rtype, 0) + 1

                # Priority statistics
                priority = reminder.priority.value
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            completed_count = status_counts.get('completed', 0)
            completion_rate = completed_count / total_reminders if total_reminders > 0 else 0.0

            return {
                "total_reminders": total_reminders,
                "active_reminders": active_count,
                "completed_reminders": completed_count,
                "dismissed_reminders": status_counts.get('dismissed', 0),
                "completion_rate": round(completion_rate, 3),
                "reminder_types": type_counts,
                "priority_distribution": priority_counts,
                "status_distribution": status_counts
            }

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}