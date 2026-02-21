#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security monitoring manager
Enterprise-level unified management interface coordinating security, monitoring, and backup modules
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..utils.config import Config
from .encryption import EncryptionManager
from .access_control import AccessController
from ..monitoring.health_checker import HealthChecker
from ..backup.backup_manager import BackupManager, RestoreManager, MigrationManager
from ..storage import DatabaseManager


class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event"""
    event_id: str
    event_type: str
    severity: AlertSeverity
    source_module: str
    timestamp: datetime
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy"""
    policy_id: str
    policy_name: str
    security_level: SecurityLevel
    encryption_required: bool = True
    mfa_required: bool = False
    backup_frequency_hours: int = 24
    monitoring_interval_minutes: int = 5
    auto_recovery_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    retention_days: int = 90


class SecurityMonitoringManager:
    """Security monitoring manager - Enterprise-level unified interface"""

    def __init__(self, config: Config):
        """
        Initialize security monitoring manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.security_monitoring_manager')

        # Initialize modules
        self.encryption_manager = EncryptionManager(config)
        self.access_controller = AccessController(config)

        # Initialize database manager (HealthChecker requires it)
        self.db_manager = DatabaseManager(config)
        self.health_checker = HealthChecker(config, self.db_manager)

        self.backup_manager = BackupManager(config, self.encryption_manager)
        self.restore_manager = self.backup_manager.restore_manager
        self.migration_manager = self.backup_manager.migration_manager

        # Security events and policies
        self.security_events: List[SecurityEvent] = []
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self._load_security_data()

        # Coordination mechanism
        self._coordination_thread = None
        self._coordination_running = False
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Start coordination service
        self._start_coordination_service()

    def _load_security_data(self):
        """Load security data"""
        try:
            # Create default security policy
            default_policy = SecurityPolicy(
                policy_id="default",
                policy_name="Default Security Policy",
                security_level=SecurityLevel.MEDIUM,
                encryption_required=True,
                mfa_required=False,
                backup_frequency_hours=24,
                monitoring_interval_minutes=5,
                auto_recovery_enabled=True,
                alert_thresholds={
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'disk_usage': 90.0,
                    'failed_login_attempts': 5.0
                },
                retention_days=90
            )
            self.security_policies["default"] = default_policy

            self.logger.info("Security data loading completed")

        except Exception as e:
            self.logger.error(f"Failed to load security data: {e}")

    def _start_coordination_service(self):
        """Start coordination service"""
        try:
            self._coordination_running = True
            self._coordination_thread = threading.Thread(target=self._coordination_worker, daemon=True)
            self._coordination_thread.start()
            self.logger.info("Security monitoring coordination service started")

        except Exception as e:
            self.logger.error(f"Failed to start coordination service: {e}")

    def _coordination_worker(self):
        """Coordination service worker thread"""
        while self._coordination_running:
            try:
                # Execute coordination tasks
                self._coordinate_modules()
                self._process_security_events()
                self._check_security_policies()

                # Wait for next check
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Coordination service error: {e}")
                time.sleep(30)

    def _coordinate_modules(self):
        """Coordinate various modules"""
        try:
            # Sync encryption status
            if self.encryption_manager.is_unlocked():
                # Ensure backup encryption is enabled
                self.backup_manager.update_backup_config(encrypt_backups=True)

            # Sync monitoring status
            health_report = self.health_checker.run_health_check()
            overall_health = 1.0 if health_report.overall_status.name == 'HEALTHY' else 0.5
            if overall_health < 0.7:
                # Poor health status, trigger backup
                self._trigger_emergency_backup()

            # Sync access control
            failed_attempts = self.access_controller.get_security_statistics().get('failed_login_attempts', 0)
            if failed_attempts > 10:
                # Too many failed login attempts, enhance monitoring
                self.health_checker.update_check_config(monitoring_interval_seconds=30)

        except Exception as e:
            self.logger.error(f"Module coordination failed: {e}")

    def _process_security_events(self):
        """Process security events"""
        try:
            # Check unresolved security events
            unresolved_events = [e for e in self.security_events if not e.resolved]

            for event in unresolved_events:
                # Execute appropriate handling based on event type
                if event.event_type == "backup_failure":
                    self._handle_backup_failure_event(event)
                elif event.event_type == "security_breach":
                    self._handle_security_breach_event(event)
                elif event.event_type == "system_anomaly":
                    self._handle_system_anomaly_event(event)

        except Exception as e:
            self.logger.error(f"Failed to process security events: {e}")

    def _check_security_policies(self):
        """Check security policy compliance"""
        try:
            for policy_id, policy in self.security_policies.items():
                # Check backup frequency
                latest_backup = self.backup_manager.list_backups()
                if latest_backup:
                    latest_backup = max(latest_backup, key=lambda x: x.created_at)
                    hours_since_backup = (datetime.now() - latest_backup.created_at).total_seconds() / 3600

                    if hours_since_backup > policy.backup_frequency_hours:
                        self._create_security_event(
                            "policy_violation",
                            AlertSeverity.WARNING,
                            "backup_manager",
                            f"Backup frequency violates policy {policy_id}: {hours_since_backup:.1f} hours since last backup"
                        )

                # Check encryption requirements
                if policy.encryption_required and not self.encryption_manager.is_unlocked():
                    self._create_security_event(
                        "policy_violation",
                        AlertSeverity.ERROR,
                        "encryption_manager",
                        f"Encryption requirement violates policy {policy_id}: encryption manager not unlocked"
                    )

        except Exception as e:
            self.logger.error(f"Failed to check security policies: {e}")

    def _trigger_emergency_backup(self):
        """Trigger emergency backup"""
        try:
            self.logger.warning("System health status poor, triggering emergency backup")
            backup_info = self.backup_manager.create_full_backup(description="Emergency backup - system health abnormal")

            if backup_info:
                self._create_security_event(
                    "emergency_backup",
                    AlertSeverity.INFO,
                    "backup_manager",
                    f"Emergency backup created: {backup_info.backup_id}"
                )

        except Exception as e:
            self.logger.error(f"Failed to trigger emergency backup: {e}")

    def _create_security_event(self, event_type: str, severity: AlertSeverity,
                             source_module: str, description: str, details: Optional[Dict[str, Any]] = None):
        """Create security event"""
        try:
            event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.security_events)}"

            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                source_module=source_module,
                timestamp=datetime.now(),
                description=description,
                details=details or {}
            )

            self.security_events.append(event)
            self.logger.info(f"Security event created: {event_id} - {description}")

            # Trigger event handlers
            self._trigger_event_handlers(event_type, event)

        except Exception as e:
            self.logger.error(f"Failed to create security event: {e}")

    def _trigger_event_handlers(self, event_type: str, event: SecurityEvent):
        """Trigger event handlers"""
        try:
            handlers = self._event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler execution failed: {e}")

        except Exception as e:
            self.logger.error(f"Failed to trigger event handlers: {e}")

    def _handle_backup_failure_event(self, event: SecurityEvent):
        """Handle backup failure event"""
        try:
            # Attempt to retry backup
            backup_info = self.backup_manager.create_incremental_backup(description="Retry after backup failure")
            if backup_info:
                event.resolved = True
                event.resolution_notes = f"Retry backup successful: {backup_info.backup_id}"
                self.logger.info(f"Backup failure event resolved: {event.event_id}")

        except Exception as e:
            self.logger.error(f"Failed to handle backup failure event: {e}")

    def _handle_security_breach_event(self, event: SecurityEvent):
        """Handle security breach event"""
        try:
            # Force re-authentication
            self.access_controller.force_logout_all_sessions()

            # Create emergency backup
            self._trigger_emergency_backup()

            event.resolved = True
            event.resolution_notes = "Forced logout all sessions and created emergency backup"
            self.logger.warning(f"Security breach event handled: {event.event_id}")

        except Exception as e:
            self.logger.error(f"Failed to handle security breach event: {e}")

    def _handle_system_anomaly_event(self, event: SecurityEvent):
        """Handle system anomaly event"""
        try:
            # Perform system recovery
            recovery_result = self.health_checker.perform_auto_recovery()

            if recovery_result.get('success', False):
                event.resolved = True
                event.resolution_notes = f"Auto recovery successful: {recovery_result.get('actions_taken', [])}"
                self.logger.info(f"System anomaly event resolved: {event.event_id}")

        except Exception as e:
            self.logger.error(f"Failed to handle system anomaly event: {e}")

    def _convert_health_report_to_dict(self, health_report) -> Dict[str, Any]:
        """Convert health report to dictionary format"""
        try:
            return {
                'overall_health': 1.0 if health_report.overall_status.name == 'HEALTHY' else 0.5,
                'status': health_report.overall_status.name,
                'timestamp': health_report.timestamp.isoformat(),
                'checks_passed': len([c for c in health_report.checks if c.status.name == 'HEALTHY']),
                'total_checks': len(health_report.checks),
                'summary': health_report.summary,
                'recommendations': health_report.recommendations
            }
        except Exception as e:
            self.logger.error(f"Failed to convert health report: {e}")
            return {'overall_health': 0.0, 'status': 'UNKNOWN', 'error': str(e)}

    # Public interface methods
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        try:
            return {
                'encryption_status': {
                    'unlocked': self.encryption_manager.is_unlocked(),
                    'statistics': self.encryption_manager.get_encryption_statistics()
                },
                'access_control_status': {
                    'statistics': self.access_controller.get_security_statistics(),
                    'active_sessions': len(self.access_controller.active_sessions)
                },
                'monitoring_status': {
                    'health': self._convert_health_report_to_dict(self.health_checker.run_health_check()),
                    'alerts': self.health_checker.get_alert_history(limit=5)
                },
                'backup_status': {
                    'statistics': self.backup_manager.get_enhanced_backup_statistics(),
                    'health_score': self.backup_manager._calculate_backup_health_score()
                },
                'security_events': {
                    'total_events': len(self.security_events),
                    'unresolved_events': len([e for e in self.security_events if not e.resolved]),
                    'recent_events': [
                        {
                            'event_id': e.event_id,
                            'type': e.event_type,
                            'severity': e.severity.value,
                            'timestamp': e.timestamp.isoformat(),
                            'description': e.description
                        }
                        for e in sorted(self.security_events, key=lambda x: x.timestamp, reverse=True)[:5]
                    ]
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {}

    def apply_security_policy(self, policy_id: str) -> bool:
        """Apply security policy"""
        try:
            policy = self.security_policies.get(policy_id)
            if not policy:
                self.logger.error(f"Security policy does not exist: {policy_id}")
                return False

            # Apply encryption policy
            if policy.encryption_required and not self.encryption_manager.is_unlocked():
                self.logger.warning("Security policy requires encryption, but encryption manager is not unlocked")

            # Apply backup policy
            self.backup_manager.update_backup_config(
                auto_backup_interval_hours=policy.backup_frequency_hours,
                backup_retention_days=policy.retention_days
            )

            # Apply monitoring policy
            self.health_checker.update_monitoring_config(
                monitoring_interval_seconds=policy.monitoring_interval_minutes * 60
            )

            # Apply access control policy
            if policy.mfa_required:
                # MFA requirement can be enabled here
                pass

            self.logger.info(f"Security policy applied: {policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply security policy: {e}")
            return False

    def register_event_handler(self, event_type: str, handler: Callable[[SecurityEvent], None]):
        """Register event handler"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_security_events(self, event_type: Optional[str] = None,
                          resolved: Optional[bool] = None) -> List[SecurityEvent]:
        """Get security events"""
        events = self.security_events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if resolved is not None:
            events = [e for e in events if e.resolved == resolved]

        return sorted(events, key=lambda x: x.timestamp, reverse=True)

    def resolve_security_event(self, event_id: str, resolution_notes: str) -> bool:
        """Resolve security event"""
        try:
            event = next((e for e in self.security_events if e.event_id == event_id), None)
            if not event:
                return False

            event.resolved = True
            event.resolution_notes = resolution_notes
            self.logger.info(f"Security event resolved: {event_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to resolve security event: {e}")
            return False

    def stop_coordination_service(self):
        """Stop coordination service"""
        try:
            self._coordination_running = False
            if self._coordination_thread and self._coordination_thread.is_alive():
                self._coordination_thread.join(timeout=5)

            # Stop automated services for each module
            self.backup_manager.stop_automated_services()

            self.logger.info("Security monitoring coordination service stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop coordination service: {e}")