#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance monitor
Provides system performance monitoring, analysis, and reporting functionality
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

from ..utils.config import Config


@dataclass
class PerformanceMetric:
    """Performance metric"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    query_count: int = 0
    query_avg_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0


@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str  # healthy, warning, critical
    cpu_status: str
    memory_status: str
    disk_status: str
    database_status: str
    cache_status: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """Performance monitor"""

    def __init__(self, config: Config):
        """
        Initialize performance monitor

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.performance_monitor')

        # Monitoring configuration
        self.monitor_config = {
            'collection_interval_seconds': 30,
            'retention_hours': 24,
            'cpu_warning_threshold': 80.0,
            'cpu_critical_threshold': 95.0,
            'memory_warning_threshold': 80.0,
            'memory_critical_threshold': 95.0,
            'disk_warning_threshold': 85.0,
            'disk_critical_threshold': 95.0,
            'query_time_warning_ms': 1000,
            'query_time_critical_ms': 5000,
            'enable_alerts': True,
            'alert_cooldown_minutes': 15
        }

        # Performance data storage
        max_samples = int(self.monitor_config['retention_hours'] * 3600 /
                         self.monitor_config['collection_interval_seconds'])
        self.metrics_history: deque = deque(maxlen=max_samples)

        # Query performance statistics
        self.query_stats = {
            'total_queries': 0,
            'total_time_ms': 0.0,
            'slow_queries': 0,
            'failed_queries': 0,
            'recent_queries': deque(maxlen=1000)
        }

        # Monitoring status
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_alert_time: Dict[str, datetime] = {}

        # Callback functions
        self.on_alert: Optional[Callable] = None
        self.on_health_change: Optional[Callable] = None

        # Baseline performance data
        self.baseline_metrics: Optional[PerformanceMetric] = None

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Performance monitoring is already running")
            return

        try:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()

            self.logger.info("Performance monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            self.monitoring_active = False
            raise

    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring_active:
            return

        try:
            self.monitoring_active = False

            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)

            self.logger.info("Performance monitoring stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop performance monitoring: {e}")

    def _monitoring_loop(self):
        """Monitoring loop"""
        self.logger.info("Performance monitoring loop started")

        while self.monitoring_active:
            try:
                # Collect performance metrics
                metric = self._collect_performance_metrics()
                self.metrics_history.append(metric)

                # Analyze system health status
                health = self._analyze_system_health(metric)

                # Check alert conditions
                self._check_alerts(metric, health)

                # Update baseline data
                self._update_baseline(metric)

                # Wait for next collection
                time.sleep(self.monitor_config['collection_interval_seconds'])

            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {e}")
                time.sleep(5)

    def _collect_performance_metrics(self) -> PerformanceMetric:
        """Collect performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

            # Query performance statistics
            query_count = len(self.query_stats['recent_queries'])
            query_avg_time_ms = (self.query_stats['total_time_ms'] /
                               max(1, self.query_stats['total_queries']))

            return PerformanceMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                query_count=query_count,
                query_avg_time_ms=query_avg_time_ms,
                cache_hit_rate=0.0,  # Needs to be obtained from cache manager
                active_connections=0  # Needs to be obtained from connection pool
            )

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetric(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0
            )

    def _analyze_system_health(self, metric: PerformanceMetric) -> SystemHealth:
        """Analyze system health status"""
        health = SystemHealth(
            overall_status="healthy",
            cpu_status="healthy",
            memory_status="healthy",
            disk_status="healthy",
            database_status="healthy",
            cache_status="healthy"
        )

        try:
            # CPU status analysis
            if metric.cpu_percent >= self.monitor_config['cpu_critical_threshold']:
                health.cpu_status = "critical"
                health.issues.append(f"CPU usage too high: {metric.cpu_percent:.1f}%")
                health.recommendations.append("Check CPU-intensive processes, consider optimizing algorithms or increasing compute resources")
            elif metric.cpu_percent >= self.monitor_config['cpu_warning_threshold']:
                health.cpu_status = "warning"
                health.issues.append(f"CPU usage relatively high: {metric.cpu_percent:.1f}%")
                health.recommendations.append("Monitor CPU usage trends, prepare optimization measures")

            # Memory status analysis
            if metric.memory_percent >= self.monitor_config['memory_critical_threshold']:
                health.memory_status = "critical"
                health.issues.append(f"Memory usage too high: {metric.memory_percent:.1f}%")
                health.recommendations.append("Clean memory cache, check for memory leaks, consider adding memory")
            elif metric.memory_percent >= self.monitor_config['memory_warning_threshold']:
                health.memory_status = "warning"
                health.issues.append(f"Memory usage relatively high: {metric.memory_percent:.1f}%")
                health.recommendations.append("Monitor memory usage trends, optimize cache strategy")

            # Disk status analysis
            try:
                disk_usage = psutil.disk_usage(self.config.data_directory)
                disk_percent = (disk_usage.used / disk_usage.total) * 100

                if disk_percent >= self.monitor_config['disk_critical_threshold']:
                    health.disk_status = "critical"
                    health.issues.append(f"Disk usage too high: {disk_percent:.1f}%")
                    health.recommendations.append("Clean old data, expand storage space")
                elif disk_percent >= self.monitor_config['disk_warning_threshold']:
                    health.disk_status = "warning"
                    health.issues.append(f"Disk usage relatively high: {disk_percent:.1f}%")
                    health.recommendations.append("Plan data cleanup and storage expansion")

            except Exception as e:
                health.disk_status = "unknown"
                health.issues.append(f"Unable to get disk usage: {e}")

            # Query performance analysis
            if metric.query_avg_time_ms >= self.monitor_config['query_time_critical_ms']:
                health.database_status = "critical"
                health.issues.append(f"Query response time too long: {metric.query_avg_time_ms:.1f}ms")
                health.recommendations.append("Optimize slow queries, check index configuration")
            elif metric.query_avg_time_ms >= self.monitor_config['query_time_warning_ms']:
                health.database_status = "warning"
                health.issues.append(f"Query response time relatively long: {metric.query_avg_time_ms:.1f}ms")
                health.recommendations.append("Analyze query performance, consider optimization")

            # Cache performance analysis
            if metric.cache_hit_rate < 0.5:  # Hit rate below 50%
                health.cache_status = "warning"
                health.issues.append(f"Cache hit rate low: {metric.cache_hit_rate:.1%}")
                health.recommendations.append("Adjust cache strategy, increase cache capacity")

            # Determine overall status
            statuses = [health.cpu_status, health.memory_status, health.disk_status,
                       health.database_status, health.cache_status]

            if "critical" in statuses:
                health.overall_status = "critical"
            elif "warning" in statuses:
                health.overall_status = "warning"
            else:
                health.overall_status = "healthy"

        except Exception as e:
            self.logger.error(f"Failed to analyze system health status: {e}")
            health.overall_status = "unknown"
            health.issues.append(f"Health status analysis failed: {e}")

        return health

    def _check_alerts(self, metric: PerformanceMetric, health: SystemHealth):
        """Check alert conditions"""
        if not self.monitor_config['enable_alerts']:
            return

        try:
            current_time = datetime.now()
            cooldown = timedelta(minutes=self.monitor_config['alert_cooldown_minutes'])

            # Check various alert conditions
            alerts_to_send = []

            # CPU alert
            if health.cpu_status == "critical":
                alert_key = "cpu_critical"
                if (alert_key not in self.last_alert_time or
                    current_time - self.last_alert_time[alert_key] > cooldown):
                    alerts_to_send.append({
                        'type': 'cpu_critical',
                        'message': f'CPU usage reached critical level: {metric.cpu_percent:.1f}%',
                        'metric': metric,
                        'health': health
                    })
                    self.last_alert_time[alert_key] = current_time

            # Memory alert
            if health.memory_status == "critical":
                alert_key = "memory_critical"
                if (alert_key not in self.last_alert_time or
                    current_time - self.last_alert_time[alert_key] > cooldown):
                    alerts_to_send.append({
                        'type': 'memory_critical',
                        'message': f'Memory usage reached critical level: {metric.memory_percent:.1f}%',
                        'metric': metric,
                        'health': health
                    })
                    self.last_alert_time[alert_key] = current_time

            # Send alerts
            for alert in alerts_to_send:
                self._send_alert(alert)

        except Exception as e:
            self.logger.error(f"Failed to check alert conditions: {e}")

    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert"""
        try:
            self.logger.warning(f"Performance alert: {alert['message']}")

            if self.on_alert:
                self.on_alert(alert)

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    def _update_baseline(self, metric: PerformanceMetric):
        """Update baseline performance data"""
        try:
            if self.baseline_metrics is None:
                self.baseline_metrics = metric
            else:
                # Update baseline using exponential moving average
                alpha = 0.1  # Smoothing factor
                self.baseline_metrics.cpu_percent = (
                    alpha * metric.cpu_percent +
                    (1 - alpha) * self.baseline_metrics.cpu_percent
                )
                self.baseline_metrics.memory_percent = (
                    alpha * metric.memory_percent +
                    (1 - alpha) * self.baseline_metrics.memory_percent
                )

        except Exception as e:
            self.logger.error(f"Failed to update baseline data: {e}")

    def record_query_performance(self, query: str, execution_time_ms: float, success: bool):
        """
        Record query performance

        Args:
            query: Query statement
            execution_time_ms: Execution time (milliseconds)
            success: Whether successful
        """
        try:
            self.query_stats['total_queries'] += 1
            self.query_stats['total_time_ms'] += execution_time_ms

            if not success:
                self.query_stats['failed_queries'] += 1

            if execution_time_ms >= self.monitor_config['query_time_warning_ms']:
                self.query_stats['slow_queries'] += 1

            # Record recent query
            self.query_stats['recent_queries'].append({
                'timestamp': datetime.now(),
                'query': query[:100],  # Truncate long queries
                'execution_time_ms': execution_time_ms,
                'success': success
            })

        except Exception as e:
            self.logger.error(f"Failed to record query performance: {e}")

    def get_current_metrics(self) -> Optional[PerformanceMetric]:
        """Get current performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_history(self, hours: int = 1) -> List[PerformanceMetric]:
        """
        Get historical performance metrics

        Args:
            hours: Get data from the last N hours

        Returns:
            List of performance metrics
        """
        if not self.metrics_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [metric for metric in self.metrics_history
                if metric.timestamp >= cutoff_time]

    def get_system_health(self) -> SystemHealth:
        """Get system health status"""
        current_metric = self.get_current_metrics()
        if current_metric:
            return self._analyze_system_health(current_metric)
        else:
            return SystemHealth(
                overall_status="unknown",
                cpu_status="unknown",
                memory_status="unknown",
                disk_status="unknown",
                database_status="unknown",
                cache_status="unknown",
                issues=["No performance data"],
                recommendations=["Start performance monitoring"]
            )

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate performance report

        Args:
            hours: Report time range (hours)

        Returns:
            Performance report
        """
        try:
            metrics = self.get_metrics_history(hours)
            if not metrics:
                return {'error': 'No performance data'}

            # Calculate statistics
            cpu_values = [m.cpu_percent for m in metrics]
            memory_values = [m.memory_percent for m in metrics]
            query_times = [m.query_avg_time_ms for m in metrics]

            report = {
                'time_range': {
                    'start': metrics[0].timestamp.isoformat(),
                    'end': metrics[-1].timestamp.isoformat(),
                    'duration_hours': hours
                },
                'cpu_stats': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory_stats': {
                    'avg': sum(memory_values) / len(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values)
                },
                'query_stats': {
                    'avg_time_ms': sum(query_times) / len(query_times),
                    'max_time_ms': max(query_times),
                    'total_queries': self.query_stats['total_queries'],
                    'slow_queries': self.query_stats['slow_queries'],
                    'failed_queries': self.query_stats['failed_queries']
                },
                'current_health': self.get_system_health().__dict__,
                'baseline_comparison': self._compare_with_baseline(metrics[-1]) if self.baseline_metrics else None
            }

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}

    def _compare_with_baseline(self, current_metric: PerformanceMetric) -> Dict[str, Any]:
        """Compare with baseline performance"""
        if not self.baseline_metrics:
            return {}

        try:
            return {
                'cpu_change_percent': current_metric.cpu_percent - self.baseline_metrics.cpu_percent,
                'memory_change_percent': current_metric.memory_percent - self.baseline_metrics.memory_percent,
                'query_time_change_ms': current_metric.query_avg_time_ms - self.baseline_metrics.query_avg_time_ms
            }

        except Exception as e:
            self.logger.error(f"Baseline comparison failed: {e}")
            return {}

    def optimize_performance(self) -> List[str]:
        """
        Performance optimization suggestions

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        try:
            health = self.get_system_health()
            current_metric = self.get_current_metrics()

            if not current_metric:
                return ["Unable to get current performance data"]

            # CPU optimization suggestions
            if current_metric.cpu_percent > 70:
                suggestions.append("CPU usage relatively high, recommend optimizing compute-intensive operations")
                suggestions.append("Consider using caching to reduce redundant computations")

            # Memory optimization suggestions
            if current_metric.memory_percent > 70:
                suggestions.append("Memory usage relatively high, recommend cleaning unnecessary caches")
                suggestions.append("Check for memory leaks")

            # Query optimization suggestions
            if current_metric.query_avg_time_ms > 500:
                suggestions.append("Query response time relatively long, recommend optimizing SQL queries")
                suggestions.append("Check database index configuration")

            # Cache optimization suggestions
            if current_metric.cache_hit_rate < 0.7:
                suggestions.append("Cache hit rate relatively low, recommend adjusting cache strategy")
                suggestions.append("Increase cache time for hot data")

            if not suggestions:
                suggestions.append("System performance is good, no special optimization needed")

        except Exception as e:
            self.logger.error(f"Failed to generate performance optimization suggestions: {e}")
            suggestions.append(f"Failed to generate optimization suggestions: {e}")

        return suggestions

    def set_callbacks(self, on_alert: Optional[Callable] = None,
                     on_health_change: Optional[Callable] = None):
        """
        Set callback functions

        Args:
            on_alert: Alert callback
            on_health_change: Health status change callback
        """
        self.on_alert = on_alert
        self.on_health_change = on_health_change

    def update_monitor_config(self, **kwargs) -> bool:
        """
        Update monitoring configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.monitor_config:
                    self.monitor_config[key] = value
                    self.logger.info(f"Performance monitoring configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update performance monitoring configuration: {e}")
            return False

    def get_monitor_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'metrics_count': len(self.metrics_history),
            'query_stats': self.query_stats.copy(),
            'config': self.monitor_config.copy(),
            'has_baseline': self.baseline_metrics is not None
        }