#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert Manager
Provides alert creation, management, and notification functionality for the system
"""

import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue

from ..utils.config import Config


class AlertSeverity(Enum):
    """Alert severity level"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: str
    source: str
    created_at: datetime
    updated_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    occurrences: int = 1
    last_occurrence: datetime = field(default_factory=datetime.now)


@dataclass
class AlertRule:
    """Alert rule"""
    rule_id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: str
    enabled: bool = True
    cooldown_minutes: int = 15
    auto_resolve: bool = False
    auto_resolve_condition: Optional[Callable[[Dict[str, Any]], bool]] = None


class AlertManager:
    """Alert manager"""

    def __init__(self, config: Config):
        """
        Initialize alert manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.alert_manager')

        # Alert configuration
        self.alert_config = {
            'max_active_alerts': 1000,
            'alert_history_retention_days': 30,
            'default_cooldown_minutes': 15,
            'enable_notifications': True,
            'notification_channels': ['log'],
            'suppress_duplicates': True,
            'auto_acknowledge_after_hours': 24
        }

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppressed_alerts: Dict[str, datetime] = {}

        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self._initialize_default_rules()

        # Notification handlers
        self.notification_handlers: Dict[str, Callable] = {
            'log': self._log_notification,
            'callback': None
        }

        # Alert queue and processing
        self.alert_queue: Queue = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.running = False

        # Callback functions
        self.on_alert_created: Optional[Callable] = None
        self.on_alert_resolved: Optional[Callable] = None

    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id='high_memory_usage',
                name='High Memory Usage',
                description='Memory usage exceeds threshold',
                condition=lambda data: data.get('memory_percent', 0) > 90,
                severity=AlertSeverity.WARNING,
                category='performance'
            ),
            AlertRule(
                rule_id='critical_memory_usage',
                name='Critical Memory Usage',
                description='Memory usage exceeds critical level',
                condition=lambda data: data.get('memory_percent', 0) > 95,
                severity=AlertSeverity.CRITICAL,
                category='performance'
            ),
            AlertRule(
                rule_id='high_disk_usage',
                name='High Disk Usage',
                description='Disk usage exceeds threshold',
                condition=lambda data: data.get('disk_percent', 0) > 85,
                severity=AlertSeverity.WARNING,
                category='storage'
            ),
            AlertRule(
                rule_id='critical_disk_usage',
                name='Critical Disk Usage',
                description='Disk usage exceeds critical level',
                condition=lambda data: data.get('disk_percent', 0) > 95,
                severity=AlertSeverity.CRITICAL,
                category='storage'
            ),
            AlertRule(
                rule_id='database_error',
                name='Database Error',
                description='Database operation failed',
                condition=lambda data: data.get('database_error', False),
                severity=AlertSeverity.ERROR,
                category='database'
            ),
            AlertRule(
                rule_id='backup_failure',
                name='Backup Failure',
                description='Automatic backup failed',
                condition=lambda data: data.get('backup_failed', False),
                severity=AlertSeverity.ERROR,
                category='backup'
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def start_processing(self):
        """Start alert processing"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Alert processor started")

    def stop_processing(self):
        """Stop alert processing"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        self.logger.info("Alert processor stopped")

    def _processing_loop(self):
        """Alert processing loop"""
        while self.running:
            try:
                # Process alerts in queue
                while not self.alert_queue.empty():
                    try:
                        alert_data = self.alert_queue.get_nowait()
                        self._process_alert_data(alert_data)
                    except Exception as e:
                        self.logger.error(f"Failed to process alert data: {e}")

                # Check auto-resolve
                self._check_auto_resolve()

                # Clean up expired alerts
                self._cleanup_expired_alerts()

                time.sleep(5)

            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
                time.sleep(10)

    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    category: str, source: str = 'system',
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create an alert

        Args:
            title: Alert title
            message: Alert message
            severity: Severity level
            category: Category
            source: Source
            metadata: Metadata

        Returns:
            Created alert
        """
        import uuid

        alert_id = f"alert_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

        # Check if should be suppressed
        if self._should_suppress_alert(alert):
            self.logger.debug(f"Alert suppressed: {alert_id}")
            return alert

        # Add to active alerts
        self.active_alerts[alert_id] = alert

        # Send notifications
        self._send_notifications(alert)

        # Trigger callback
        if self.on_alert_created:
            try:
                self.on_alert_created(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        self.logger.info(f"Alert created: [{severity.value}] {title}")
        return alert

    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Check duplicate alerts
        if self.alert_config['suppress_duplicates']:
            for active_alert in self.active_alerts.values():
                if (active_alert.title == alert.title and
                    active_alert.category == alert.category and
                    active_alert.status == AlertStatus.ACTIVE):
                    # Update existing alert occurrence count
                    active_alert.occurrences += 1
                    active_alert.last_occurrence = datetime.now()
                    active_alert.updated_at = datetime.now()
                    return True

        # Check suppression list
        suppress_key = f"{alert.category}:{alert.title}"
        if suppress_key in self.suppressed_alerts:
            suppress_until = self.suppressed_alerts[suppress_key]
            if datetime.now() < suppress_until:
                return True
            else:
                del self.suppressed_alerts[suppress_key]

        return False

    def _send_notifications(self, alert: Alert):
        """Send notifications"""
        if not self.alert_config['enable_notifications']:
            return

        for channel in self.alert_config['notification_channels']:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Failed to send notification [{channel}]: {e}")

    def _log_notification(self, alert: Alert):
        """Log notification"""
        log_levels = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }

        level = log_levels.get(alert.severity, logging.INFO)
        self.logger.log(level, f"[ALERT] {alert.title}: {alert.message}")

    def check_rules(self, data: Dict[str, Any]):
        """
        Check alert rules

        Args:
            data: Check data
        """
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            try:
                if rule.condition(data):
                    # Check cooldown period
                    if self._is_in_cooldown(rule):
                        continue

                    # Create alert
                    self.create_alert(
                        title=rule.name,
                        message=rule.description,
                        severity=rule.severity,
                        category=rule.category,
                        source='rule_check',
                        metadata={'rule_id': rule.rule_id, 'data': data}
                    )

            except Exception as e:
                self.logger.error(f"Rule check {rule.rule_id} failed: {e}")

    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown"""
        for alert in self.active_alerts.values():
            if (alert.metadata.get('rule_id') == rule.rule_id and
                alert.status == AlertStatus.ACTIVE):
                time_since_creation = datetime.now() - alert.created_at
                if time_since_creation < timedelta(minutes=rule.cooldown_minutes):
                    return True
        return False

    def _check_auto_resolve(self):
        """Check auto-resolve conditions"""
        for rule in self.alert_rules.values():
            if not rule.auto_resolve or not rule.auto_resolve_condition:
                continue

            for alert in list(self.active_alerts.values()):
                if (alert.metadata.get('rule_id') == rule.rule_id and
                    alert.status == AlertStatus.ACTIVE):
                    try:
                        if rule.auto_resolve_condition(alert.metadata.get('data', {})):
                            self.resolve_alert(alert.alert_id, "Auto-resolved")
                    except Exception as e:
                        self.logger.error(f"Auto-resolve check failed: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = 'system') -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID
            acknowledged_by: Acknowledger

        Returns:
            Whether acknowledgement was successful
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        alert.updated_at = datetime.now()

        self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = 'system') -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert ID
            resolved_by: Resolver

        Returns:
            Whether resolution was successful
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()

        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]

        # Trigger callback
        if self.on_alert_resolved:
            try:
                self.on_alert_resolved(alert)
            except Exception as e:
                self.logger.error(f"Alert resolved callback failed: {e}")

        self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True

    def suppress_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """
        Suppress an alert

        Args:
            alert_id: Alert ID
            duration_minutes: Suppression duration (minutes)

        Returns:
            Whether suppression was successful
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.updated_at = datetime.now()

        # Add to suppression list
        suppress_key = f"{alert.category}:{alert.title}"
        self.suppressed_alerts[suppress_key] = datetime.now() + timedelta(minutes=duration_minutes)

        self.logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes")
        return True

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         category: Optional[str] = None) -> List[Alert]:
        """
        Get active alerts

        Args:
            severity: Severity filter
            category: Category filter

        Returns:
            List of alerts
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_history(self, limit: int = 100,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get alert history

        Args:
            limit: Quantity limit
            severity: Severity filter

        Returns:
            List of alert history
        """
        history = self.alert_history[-limit:]

        if severity:
            history = [a for a in history if a.severity == severity]

        return sorted(history, key=lambda a: a.created_at, reverse=True)

    def _cleanup_expired_alerts(self):
        """Clean up expired alerts"""
        retention_days = self.alert_config['alert_history_retention_days']
        cutoff = datetime.now() - timedelta(days=retention_days)

        self.alert_history = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff
        ]

        # Clean up expired suppressions
        expired_suppressions = [
            key for key, until in self.suppressed_alerts.items()
            if datetime.now() >= until
        ]
        for key in expired_suppressions:
            del self.suppressed_alerts[key]

    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Alert rule added: {rule.rule_id}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Alert rule removed: {rule_id}")
            return True
        return False

    def register_notification_handler(self, channel: str, handler: Callable):
        """Register notification handler"""
        self.notification_handlers[channel] = handler
        if channel not in self.alert_config['notification_channels']:
            self.alert_config['notification_channels'].append(channel)
        self.logger.info(f"Notification handler registered: {channel}")

    def set_callbacks(self, on_alert_created: Optional[Callable] = None,
                     on_alert_resolved: Optional[Callable] = None):
        """Set callback functions"""
        self.on_alert_created = on_alert_created
        self.on_alert_resolved = on_alert_resolved

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        severity_counts = {s.value: 0 for s in AlertSeverity}
        category_counts = {}

        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
            category_counts[alert.category] = category_counts.get(alert.category, 0) + 1

        return {
            'active_alerts': len(self.active_alerts),
            'history_count': len(self.alert_history),
            'suppressed_count': len(self.suppressed_alerts),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'rules_count': len(self.alert_rules)
        }

    def update_alert_config(self, **kwargs) -> bool:
        """Update alert configuration"""
        try:
            for key, value in kwargs.items():
                if key in self.alert_config:
                    self.alert_config[key] = value
                    self.logger.info(f"Alert configuration updated: {key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update alert configuration: {e}")
            return False

    def queue_alert_data(self, data: Dict[str, Any]):
        """Queue alert data for processing"""
        self.alert_queue.put(data)