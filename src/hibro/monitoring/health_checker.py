#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Checker
Provides system health status checking and reporting functionality
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..utils.config import Config
from ..storage.database import DatabaseManager


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result"""
    check_name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


@dataclass
class SystemHealthReport:
    """System health report"""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]
    generated_at: datetime
    recommendations: List[str]


class HealthChecker:
    """Health checker"""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        """
        Initialize health checker

        Args:
            config: Configuration object
            db_manager: Database manager
        """
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.health_checker')

        # Check configuration
        self.check_config = {
            'check_database': True,
            'check_filesystem': True,
            'check_memory_usage': True,
            'check_disk_space': True,
            'check_cache': True,
            'check_security': True,
            'check_connectivity': True,
            'disk_space_warning_threshold': 80,
            'disk_space_critical_threshold': 95,
            'memory_warning_threshold': 80,
            'memory_critical_threshold': 95
        }

        # Custom check functions
        self.custom_checks: Dict[str, Callable] = {}

        # Check history
        self.check_history: List[SystemHealthReport] = []

    def register_custom_check(self, name: str, check_func: Callable):
        """
        Register custom check

        Args:
            name: Check name
            check_func: Check function
        """
        self.custom_checks[name] = check_func
        self.logger.info(f"Custom health check registered: {name}")

    def run_health_check(self) -> SystemHealthReport:
        """
        Run health check

        Returns:
            System health report
        """
        checks = []
        start_time = datetime.now()

        try:
            # Database check
            if self.check_config['check_database']:
                checks.append(self._check_database())

            # File system check
            if self.check_config['check_filesystem']:
                checks.append(self._check_filesystem())

            # Disk space check
            if self.check_config['check_disk_space']:
                checks.append(self._check_disk_space())

            # Memory usage check
            if self.check_config['check_memory_usage']:
                checks.append(self._check_memory_usage())

            # Cache check
            if self.check_config['check_cache']:
                checks.append(self._check_cache())

            # Security check
            if self.check_config['check_security']:
                checks.append(self._check_security())

            # Run custom checks
            for name, check_func in self.custom_checks.items():
                try:
                    result = check_func()
                    if isinstance(result, HealthCheckResult):
                        checks.append(result)
                except Exception as e:
                    checks.append(HealthCheckResult(
                        check_name=name,
                        status=HealthStatus.WARNING,
                        message=f"Check execution failed: {e}",
                        details={'error': str(e)},
                        timestamp=datetime.now(),
                        duration_ms=0
                    ))

            # Calculate overall status
            overall_status = self._calculate_overall_status(checks)

            # Generate summary and recommendations
            summary = self._generate_summary(checks)
            recommendations = self._generate_recommendations(checks)

            report = SystemHealthReport(
                overall_status=overall_status,
                checks=checks,
                summary=summary,
                generated_at=start_time,
                recommendations=recommendations
            )

            # Record history
            self.check_history.append(report)
            if len(self.check_history) > 100:
                self.check_history = self.check_history[-50:]

            return report

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealthReport(
                overall_status=HealthStatus.UNKNOWN,
                checks=[],
                summary={'error': str(e)},
                generated_at=start_time,
                recommendations=["Please check system logs for details"]
            )

    def _check_database(self) -> HealthCheckResult:
        """Check database health status"""
        start_time = datetime.now()
        details = {}

        try:
            # Check database connection
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Execute integrity check
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()
                details['integrity'] = integrity_result[0] if integrity_result else 'unknown'

                # Check database size
                db_size = self.db_manager.db_path.stat().st_size
                details['size_mb'] = db_size / (1024 * 1024)

                # Check memory count
                cursor.execute("SELECT COUNT(*) FROM memories")
                memory_count = cursor.fetchone()[0]
                details['memory_count'] = memory_count

                # Check most recent memory
                cursor.execute("SELECT created_at FROM memories ORDER BY created_at DESC LIMIT 1")
                last_memory = cursor.fetchone()
                details['last_memory_time'] = last_memory[0] if last_memory else None

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            if details['integrity'] == 'ok':
                status = HealthStatus.HEALTHY
                message = f"Database is healthy ({details['memory_count']} memories)"
            else:
                status = HealthStatus.CRITICAL
                message = f"Database integrity issue: {details['integrity']}"

            return HealthCheckResult(
                check_name="database",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_filesystem(self) -> HealthCheckResult:
        """Check filesystem health status"""
        start_time = datetime.now()
        details = {}

        try:
            data_dir = Path(self.config.data_directory)

            # Check data directory
            if not data_dir.exists():
                status = HealthStatus.CRITICAL
                message = "Data directory does not exist"
            else:
                # Check subdirectories
                subdirs = ['conversations', 'contexts', 'backups', 'logs']
                dir_status = {}

                for subdir in subdirs:
                    subdir_path = data_dir / subdir
                    dir_status[subdir] = {
                        'exists': subdir_path.exists(),
                        'writable': os.access(subdir_path, os.W_OK) if subdir_path.exists() else False
                    }

                details['directories'] = dir_status

                # Check write permissions
                if os.access(data_dir, os.W_OK):
                    status = HealthStatus.HEALTHY
                    message = "Filesystem is healthy"
                else:
                    status = HealthStatus.WARNING
                    message = "Data directory has restricted write permissions"

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                check_name="filesystem",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="filesystem",
                status=HealthStatus.WARNING,
                message=f"Filesystem check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space"""
        start_time = datetime.now()
        details = {}

        try:
            import shutil

            data_dir = Path(self.config.data_directory)
            disk_usage = shutil.disk_usage(data_dir)

            total_gb = disk_usage.total / (1024 ** 3)
            used_gb = disk_usage.used / (1024 ** 3)
            free_gb = disk_usage.free / (1024 ** 3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            details = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'used_percent': round(used_percent, 2)
            }

            if used_percent >= self.check_config['disk_space_critical_threshold']:
                status = HealthStatus.CRITICAL
                message = f"Disk space critically low ({used_percent:.1f}% used)"
            elif used_percent >= self.check_config['disk_space_warning_threshold']:
                status = HealthStatus.WARNING
                message = f"Disk space is low ({used_percent:.1f}% used)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space is sufficient ({free_gb:.1f}GB available)"

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                check_name="disk_space",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk space check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage"""
        start_time = datetime.now()
        details = {}

        try:
            import psutil

            memory = psutil.virtual_memory()
            details = {
                'total_mb': round(memory.total / (1024 ** 2), 2),
                'available_mb': round(memory.available / (1024 ** 2), 2),
                'used_percent': round(memory.percent, 2)
            }

            if memory.percent >= self.check_config['memory_critical_threshold']:
                status = HealthStatus.CRITICAL
                message = f"Memory usage is critically high ({memory.percent:.1f}%)"
            elif memory.percent >= self.check_config['memory_warning_threshold']:
                status = HealthStatus.WARNING
                message = f"Memory usage is high ({memory.percent:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage is normal ({memory.percent:.1f}%)"

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                check_name="memory_usage",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except ImportError:
            return HealthCheckResult(
                check_name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message="psutil library not installed, cannot check memory",
                details={},
                timestamp=datetime.now(),
                duration_ms=0
            )
        except Exception as e:
            return HealthCheckResult(
                check_name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_cache(self) -> HealthCheckResult:
        """Check cache status"""
        start_time = datetime.now()
        details = {}

        try:
            # Simplified cache check
            cache_dir = Path(self.config.data_directory) / 'cache'

            if cache_dir.exists():
                cache_files = list(cache_dir.iterdir())
                total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

                details = {
                    'cache_directory_exists': True,
                    'cache_file_count': len(cache_files),
                    'cache_size_mb': round(total_size / (1024 ** 2), 2)
                }

                status = HealthStatus.HEALTHY
                message = f"Cache directory is healthy ({len(cache_files)} files)"
            else:
                details = {'cache_directory_exists': False}
                status = HealthStatus.WARNING
                message = "Cache directory does not exist"

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                check_name="cache",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="cache",
                status=HealthStatus.WARNING,
                message=f"Cache check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_security(self) -> HealthCheckResult:
        """Check security status"""
        start_time = datetime.now()
        details = {}
        warnings = []

        try:
            data_dir = Path(self.config.data_directory)

            # Check file permissions
            db_path = data_dir / 'hibro.db'
            if db_path.exists():
                db_stat = db_path.stat()
                # Check if too open (Unix permissions)
                db_mode = oct(db_stat.st_mode)[-3:]
                details['database_permissions'] = db_mode

                # On Unix systems, check if readable by other users
                if db_mode[-1] != '0':  # Others have permissions
                    warnings.append("Database file permissions may be too open")

            # Check key files
            keys_dir = data_dir / 'keys'
            if keys_dir.exists():
                details['keys_directory_exists'] = True
                key_files = list(keys_dir.glob('*.key'))
                details['key_file_count'] = len(key_files)
            else:
                details['keys_directory_exists'] = False

            # Check backup encryption
            backup_dir = data_dir / 'backups'
            if backup_dir.exists():
                encrypted_backups = list(backup_dir.glob('*.enc'))
                all_backups = list(backup_dir.glob('*.tar*'))
                details['encrypted_backup_ratio'] = (
                    len(encrypted_backups) / len(all_backups) if all_backups else 0
                )

            if warnings:
                status = HealthStatus.WARNING
                message = "; ".join(warnings)
            else:
                status = HealthStatus.HEALTHY
                message = "Security check passed"

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                check_name="security",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="security",
                status=HealthStatus.WARNING,
                message=f"Security check failed: {e}",
                details={'error': str(e)},
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _calculate_overall_status(self, checks: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall status"""
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in checks]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _generate_summary(self, checks: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate summary"""
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        for check in checks:
            status_counts[check.status] += 1

        return {
            'total_checks': len(checks),
            'healthy_count': status_counts[HealthStatus.HEALTHY],
            'warning_count': status_counts[HealthStatus.WARNING],
            'critical_count': status_counts[HealthStatus.CRITICAL],
            'unknown_count': status_counts[HealthStatus.UNKNOWN]
        }

    def _generate_recommendations(self, checks: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        for check in checks:
            if check.status == HealthStatus.CRITICAL:
                if check.check_name == "database":
                    recommendations.append("Immediately check database integrity and repair")
                elif check.check_name == "disk_space":
                    recommendations.append("Clean up disk space or expand storage capacity immediately")
                elif check.check_name == "memory_usage":
                    recommendations.append("Restart the application or increase system memory")

            elif check.status == HealthStatus.WARNING:
                if check.check_name == "disk_space":
                    recommendations.append("Monitor disk usage and consider cleaning up old files")
                elif check.check_name == "memory_usage":
                    recommendations.append("Monitor memory usage and consider optimizing the application")
                elif check.check_name == "security":
                    recommendations.append("Check and fix security configuration issues")

        if not recommendations:
            recommendations.append("System is running normally, continue monitoring")

        return recommendations

    # ==================== Stage 4 Enterprise Monitoring Enhancement Features ====================

    def start_continuous_monitoring(self, interval_seconds: int = 300) -> bool:
        """
        Start continuous monitoring

        Args:
            interval_seconds: Monitoring interval (seconds)

        Returns:
            Whether startup was successful
        """
        try:
            import threading
            import time

            if hasattr(self, '_monitoring_thread') and self._monitoring_thread.is_alive():
                self.logger.warning("Continuous monitoring is already running")
                return False

            self._monitoring_active = True
            self._monitoring_interval = interval_seconds

            def monitoring_loop():
                while self._monitoring_active:
                    try:
                        report = self.run_health_check()
                        self._process_monitoring_report(report)

                        # Check if auto-recovery is needed
                        self._attempt_auto_recovery(report)

                        time.sleep(self._monitoring_interval)
                    except Exception as e:
                        self.logger.error(f"Monitoring loop error: {e}")
                        time.sleep(60)  # Wait 1 minute on error

            self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self._monitoring_thread.start()

            self.logger.info(f"Continuous monitoring started, interval {interval_seconds} seconds")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start continuous monitoring: {e}")
            return False

    def stop_continuous_monitoring(self) -> bool:
        """
        Stop continuous monitoring

        Returns:
            Whether stop was successful
        """
        try:
            if hasattr(self, '_monitoring_active'):
                self._monitoring_active = False

            if hasattr(self, '_monitoring_thread') and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5)

            self.logger.info("Continuous monitoring stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop continuous monitoring: {e}")
            return False

    def _process_monitoring_report(self, report: SystemHealthReport):
        """Process monitoring report"""
        try:
            # Detect anomaly patterns
            anomalies = self._detect_anomalies(report)

            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} anomalies")
                for anomaly in anomalies:
                    self.logger.warning(f"Anomaly: {anomaly}")

            # Trigger alerts
            if report.overall_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                self._trigger_alerts(report, anomalies)

            # Record performance metrics
            self._record_performance_metrics(report)

        except Exception as e:
            self.logger.error(f"Failed to process monitoring report: {e}")

    def _detect_anomalies(self, report: SystemHealthReport) -> List[str]:
        """Detect anomaly patterns"""
        anomalies = []

        try:
            # Check historical trends
            if len(self.check_history) >= 3:
                recent_reports = self.check_history[-3:]

                # Detect performance degradation trends
                disk_usage_trend = []
                memory_usage_trend = []

                for hist_report in recent_reports:
                    for check in hist_report.checks:
                        if check.check_name == "disk_space" and "used_percent" in check.details:
                            disk_usage_trend.append(check.details["used_percent"])
                        elif check.check_name == "memory_usage" and "used_percent" in check.details:
                            memory_usage_trend.append(check.details["used_percent"])

                # Detect rapid disk usage growth
                if len(disk_usage_trend) >= 3:
                    recent_growth = disk_usage_trend[-1] - disk_usage_trend[-3]
                    if recent_growth > 10:  # 10% growth
                        anomalies.append(f"Rapid disk usage growth: {recent_growth:.1f}%")

                # Detect abnormal memory usage fluctuations
                if len(memory_usage_trend) >= 3:
                    avg_memory = sum(memory_usage_trend) / len(memory_usage_trend)
                    current_memory = memory_usage_trend[-1]
                    if abs(current_memory - avg_memory) > 20:  # 20% deviation
                        anomalies.append(f"Abnormal memory usage fluctuation: current {current_memory:.1f}%, average {avg_memory:.1f}%")

            # Detect abnormal check timing
            for check in report.checks:
                if check.duration_ms > 5000:  # Over 5 seconds
                    anomalies.append(f"{check.check_name} check took too long: {check.duration_ms:.0f}ms")

            # Detect status mutations
            if len(self.check_history) >= 1:
                last_report = self.check_history[-1]
                for current_check in report.checks:
                    for last_check in last_report.checks:
                        if (current_check.check_name == last_check.check_name and
                            current_check.status != last_check.status and
                            current_check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]):
                            anomalies.append(f"{current_check.check_name} status mutation: {last_check.status.value} -> {current_check.status.value}")

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")

        return anomalies

    def _trigger_alerts(self, report: SystemHealthReport, anomalies: List[str]):
        """Trigger alerts"""
        try:
            alert_data = {
                'timestamp': report.generated_at.isoformat(),
                'overall_status': report.overall_status.value,
                'critical_checks': [
                    check.check_name for check in report.checks
                    if check.status == HealthStatus.CRITICAL
                ],
                'warning_checks': [
                    check.check_name for check in report.checks
                    if check.status == HealthStatus.WARNING
                ],
                'anomalies': anomalies,
                'recommendations': report.recommendations
            }

            # Record alerts
            if not hasattr(self, '_alert_history'):
                self._alert_history = []

            self._alert_history.append(alert_data)

            # Limit alert history size
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-500:]

            # Here you can integrate external alert systems (email, SMS, Webhook, etc.)
            self.logger.warning(f"System alert: {report.overall_status.value}")

            if report.overall_status == HealthStatus.CRITICAL:
                self.logger.critical("System is in critical state, requires immediate attention")

        except Exception as e:
            self.logger.error(f"Failed to trigger alerts: {e}")

    def _record_performance_metrics(self, report: SystemHealthReport):
        """Record performance metrics"""
        try:
            if not hasattr(self, '_performance_metrics'):
                self._performance_metrics = []

            metrics = {
                'timestamp': report.generated_at.isoformat(),
                'overall_status': report.overall_status.value,
                'check_count': len(report.checks),
                'total_duration_ms': sum(check.duration_ms for check in report.checks)
            }

            # Extract specific metrics
            for check in report.checks:
                if check.check_name == "disk_space" and "used_percent" in check.details:
                    metrics['disk_usage_percent'] = check.details['used_percent']
                    metrics['disk_free_gb'] = check.details['free_gb']
                elif check.check_name == "memory_usage" and "used_percent" in check.details:
                    metrics['memory_usage_percent'] = check.details['used_percent']
                    metrics['memory_available_mb'] = check.details['available_mb']
                elif check.check_name == "database" and "memory_count" in check.details:
                    metrics['memory_count'] = check.details['memory_count']
                    metrics['database_size_mb'] = check.details['size_mb']

            self._performance_metrics.append(metrics)

            # Limit metrics history size
            if len(self._performance_metrics) > 10000:
                self._performance_metrics = self._performance_metrics[-5000:]

        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")

    def _attempt_auto_recovery(self, report: SystemHealthReport):
        """Attempt auto-recovery"""
        try:
            recovery_actions = []

            for check in report.checks:
                if check.status == HealthStatus.CRITICAL:
                    if check.check_name == "disk_space":
                        # Auto-clean temporary files
                        if self._auto_cleanup_temp_files():
                            recovery_actions.append("Cleaned temporary files")

                    elif check.check_name == "cache":
                        # Auto-clean cache
                        if self._auto_cleanup_cache():
                            recovery_actions.append("Cleaned cache")

                    elif check.check_name == "database":
                        # Attempt database repair
                        if self._auto_repair_database():
                            recovery_actions.append("Database repair")

            if recovery_actions:
                self.logger.info(f"Executed auto-recovery operations: {', '.join(recovery_actions)}")

        except Exception as e:
            self.logger.error(f"Auto-recovery failed: {e}")

    def _auto_cleanup_temp_files(self) -> bool:
        """Auto-clean temporary files"""
        try:
            import tempfile
            import shutil

            temp_dir = Path(tempfile.gettempdir())
            hibro_temp = temp_dir / 'hibro_temp'

            if hibro_temp.exists():
                shutil.rmtree(hibro_temp)
                self.logger.info("Temporary file cleanup completed")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Temporary file cleanup failed: {e}")
            return False

    def _auto_cleanup_cache(self) -> bool:
        """Auto-clean cache"""
        try:
            cache_dir = Path(self.config.data_directory) / 'cache'

            if cache_dir.exists():
                # Delete cache files older than 7 days
                import time
                current_time = time.time()
                cleaned_count = 0

                for cache_file in cache_dir.iterdir():
                    if cache_file.is_file():
                        file_age = current_time - cache_file.stat().st_mtime
                        if file_age > 7 * 24 * 3600:  # 7 days
                            cache_file.unlink()
                            cleaned_count += 1

                if cleaned_count > 0:
                    self.logger.info(f"Cleaned up {cleaned_count} expired cache files")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return False

    def _auto_repair_database(self) -> bool:
        """Auto-repair database"""
        try:
            # Simple database repair attempt
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Execute VACUUM cleanup
                cursor.execute("VACUUM")

                # Rebuild indexes
                cursor.execute("REINDEX")

                conn.commit()

            self.logger.info("Database auto-repair completed")
            return True

        except Exception as e:
            self.logger.error(f"Database auto-repair failed: {e}")
            return False

    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends

        Args:
            hours: Time range (hours)

        Returns:
            Performance trend data
        """
        try:
            if not hasattr(self, '_performance_metrics'):
                return {'error': 'No performance metrics data'}

            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Filter data within specified time range
            recent_metrics = [
                metric for metric in self._performance_metrics
                if datetime.fromisoformat(metric['timestamp']) >= cutoff_time
            ]

            if not recent_metrics:
                return {'error': 'No data in specified time range'}

            # Calculate trends
            trends = {
                'time_range_hours': hours,
                'data_points': len(recent_metrics),
                'trends': {}
            }

            # Disk usage trends
            disk_usage = [m.get('disk_usage_percent') for m in recent_metrics if 'disk_usage_percent' in m]
            if disk_usage:
                trends['trends']['disk_usage'] = {
                    'current': disk_usage[-1],
                    'average': sum(disk_usage) / len(disk_usage),
                    'min': min(disk_usage),
                    'max': max(disk_usage),
                    'trend': 'increasing' if disk_usage[-1] > disk_usage[0] else 'decreasing'
                }

            # Memory usage trends
            memory_usage = [m.get('memory_usage_percent') for m in recent_metrics if 'memory_usage_percent' in m]
            if memory_usage:
                trends['trends']['memory_usage'] = {
                    'current': memory_usage[-1],
                    'average': sum(memory_usage) / len(memory_usage),
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'trend': 'increasing' if memory_usage[-1] > memory_usage[0] else 'decreasing'
                }

            # Memory count trends
            memory_counts = [m.get('memory_count') for m in recent_metrics if 'memory_count' in m]
            if memory_counts:
                trends['trends']['memory_count'] = {
                    'current': memory_counts[-1],
                    'average': sum(memory_counts) / len(memory_counts),
                    'min': min(memory_counts),
                    'max': max(memory_counts),
                    'growth': memory_counts[-1] - memory_counts[0] if len(memory_counts) > 1 else 0
                }

            return trends

        except Exception as e:
            self.logger.error(f"Failed to get performance trends: {e}")
            return {'error': str(e)}

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get alert history

        Args:
            limit: Return quantity limit

        Returns:
            List of alert history
        """
        try:
            if not hasattr(self, '_alert_history'):
                return []

            return self._alert_history[-limit:]

        except Exception as e:
            self.logger.error(f"Failed to get alert history: {e}")
            return []

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics

        Returns:
            Monitoring statistics
        """
        try:
            stats = {
                'monitoring_active': hasattr(self, '_monitoring_active') and self._monitoring_active,
                'monitoring_interval': getattr(self, '_monitoring_interval', 0),
                'total_health_checks': len(self.check_history),
                'performance_metrics_count': len(getattr(self, '_performance_metrics', [])),
                'alert_count': len(getattr(self, '_alert_history', [])),
                'custom_checks_count': len(self.custom_checks)
            }

            # Last 24 hours statistics
            if hasattr(self, '_alert_history'):
                from datetime import datetime, timedelta
                last_24h = datetime.now() - timedelta(hours=24)

                recent_alerts = [
                    alert for alert in self._alert_history
                    if datetime.fromisoformat(alert['timestamp']) >= last_24h
                ]

                stats['alerts_last_24h'] = len(recent_alerts)
                stats['critical_alerts_last_24h'] = len([
                    alert for alert in recent_alerts
                    if alert['overall_status'] == 'critical'
                ])

            # System health trend
            if len(self.check_history) >= 10:
                recent_reports = self.check_history[-10:]
                healthy_count = len([r for r in recent_reports if r.overall_status == HealthStatus.HEALTHY])
                stats['health_trend'] = {
                    'healthy_percentage': (healthy_count / len(recent_reports)) * 100,
                    'recent_status': recent_reports[-1].overall_status.value
                }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get monitoring statistics: {e}")
            return {'error': str(e)}

    def export_monitoring_data(self, start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Export monitoring data

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Monitoring data
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'health_reports': [],
                'performance_metrics': [],
                'alert_history': []
            }

            # Filter health reports
            filtered_reports = self.check_history
            if start_date:
                filtered_reports = [r for r in filtered_reports if r.generated_at >= start_date]
            if end_date:
                filtered_reports = [r for r in filtered_reports if r.generated_at <= end_date]

            for report in filtered_reports:
                export_data['health_reports'].append({
                    'timestamp': report.generated_at.isoformat(),
                    'overall_status': report.overall_status.value,
                    'checks': [
                        {
                            'name': check.check_name,
                            'status': check.status.value,
                            'message': check.message,
                            'duration_ms': check.duration_ms,
                            'details': check.details
                        }
                        for check in report.checks
                    ],
                    'recommendations': report.recommendations
                })

            # Filter performance metrics
            if hasattr(self, '_performance_metrics'):
                filtered_metrics = self._performance_metrics
                if start_date:
                    filtered_metrics = [
                        m for m in filtered_metrics
                        if datetime.fromisoformat(m['timestamp']) >= start_date
                    ]
                if end_date:
                    filtered_metrics = [
                        m for m in filtered_metrics
                        if datetime.fromisoformat(m['timestamp']) <= end_date
                    ]
                export_data['performance_metrics'] = filtered_metrics

            # Filter alert history
            if hasattr(self, '_alert_history'):
                filtered_alerts = self._alert_history
                if start_date:
                    filtered_alerts = [
                        a for a in filtered_alerts
                        if datetime.fromisoformat(a['timestamp']) >= start_date
                    ]
                if end_date:
                    filtered_alerts = [
                        a for a in filtered_alerts
                        if datetime.fromisoformat(a['timestamp']) <= end_date
                    ]
                export_data['alert_history'] = filtered_alerts

            return export_data

        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, checks: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        for check in checks:
            if check.status == HealthStatus.CRITICAL:
                if check.check_name == 'database':
                    recommendations.append("Immediately fix database integrity issues, may need to restore from backup")
                elif check.check_name == 'disk_space':
                    recommendations.append("Immediately clean up disk space or expand storage capacity")
                elif check.check_name == 'memory_usage':
                    recommendations.append("Free up memory or increase system memory")

            elif check.status == HealthStatus.WARNING:
                if check.check_name == 'disk_space':
                    recommendations.append("Plan to clean up unnecessary files and free up disk space")
                elif check.check_name == 'memory_usage':
                    recommendations.append("Consider optimizing memory usage or increasing cache cleanup frequency")

        return list(set(recommendations))  # Remove duplicates

    def get_check_history(self, limit: int = 10) -> List[SystemHealthReport]:
        """Get check history"""
        return self.check_history[-limit:]

    def update_check_config(self, **kwargs) -> bool:
        """Update check configuration"""
        try:
            for key, value in kwargs.items():
                if key in self.check_config:
                    self.check_config[key] = value
                    self.logger.info(f"Health check configuration updated: {key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update health check configuration: {e}")
            return False


# Add missing imports
import os