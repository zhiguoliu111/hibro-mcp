#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Manager
Provides system self-diagnostics and troubleshooting functionality
"""

import os
import sys
import platform
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..utils.config import Config
from ..storage.database import DatabaseManager


class DiagnosticLevel(Enum):
    """Diagnostic level"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticIssue:
    """Diagnostic issue"""
    issue_id: str
    level: DiagnosticLevel
    category: str
    description: str
    possible_causes: List[str]
    suggested_solutions: List[str]
    affected_components: List[str]
    auto_fixable: bool


@dataclass
class DiagnosticReport:
    """Diagnostic report"""
    report_id: str
    generated_at: datetime
    system_info: Dict[str, Any]
    issues: List[DiagnosticIssue]
    summary: Dict[str, int]
    auto_fix_results: List[Dict[str, Any]]


class DiagnosticManager:
    """Diagnostic manager"""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        """
        Initialize diagnostic manager

        Args:
            config: Configuration object
            db_manager: Database manager
        """
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.diagnostic_manager')

        # Diagnostic rules
        self.diagnostic_rules = self._initialize_diagnostic_rules()

        # Diagnostic history
        self.diagnostic_history: List[DiagnosticReport] = []

    def _initialize_diagnostic_rules(self) -> List[Dict[str, Any]]:
        """Initialize diagnostic rules"""
        return [
            {
                'id': 'db_corruption',
                'check': self._check_database_corruption,
                'category': 'database',
                'auto_fix': False
            },
            {
                'id': 'db_size_warning',
                'check': self._check_database_size,
                'category': 'database',
                'auto_fix': True,
                'fix': self._fix_database_size
            },
            {
                'id': 'missing_directories',
                'check': self._check_missing_directories,
                'category': 'filesystem',
                'auto_fix': True,
                'fix': self._fix_missing_directories
            },
            {
                'id': 'permission_issues',
                'check': self._check_permission_issues,
                'category': 'filesystem',
                'auto_fix': False
            },
            {
                'id': 'orphaned_files',
                'check': self._check_orphaned_files,
                'category': 'filesystem',
                'auto_fix': True,
                'fix': self._fix_orphaned_files
            },
            {
                'id': 'config_issues',
                'check': self._check_config_issues,
                'category': 'configuration',
                'auto_fix': False
            },
            {
                'id': 'memory_leak_hint',
                'check': self._check_memory_leak_hints,
                'category': 'performance',
                'auto_fix': False
            },
            {
                'id': 'old_backups',
                'check': self._check_old_backups,
                'category': 'backup',
                'auto_fix': True,
                'fix': self._fix_old_backups
            }
        ]

    def run_diagnostics(self, auto_fix: bool = False) -> DiagnosticReport:
        """
        Run system diagnostics

        Args:
            auto_fix: Whether to automatically fix fixable issues

        Returns:
            Diagnostic report
        """
        report_id = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting system diagnostics: {report_id}")

        # Collect system information
        system_info = self._collect_system_info()

        # Run diagnostic checks
        issues = []
        for rule in self.diagnostic_rules:
            try:
                result = rule['check']()
                if result:
                    result['category'] = rule['category']
                    result['auto_fixable'] = rule.get('auto_fix', False)
                    issues.append(self._create_diagnostic_issue(rule['id'], result))

            except Exception as e:
                self.logger.error(f"Diagnostic rule {rule['id']} execution failed: {e}")

        # Generate summary
        summary = self._generate_summary(issues)

        # Auto-fix
        auto_fix_results = []
        if auto_fix:
            auto_fix_results = self._apply_auto_fixes(issues)

        # Create report
        report = DiagnosticReport(
            report_id=report_id,
            generated_at=datetime.now(),
            system_info=system_info,
            issues=issues,
            summary=summary,
            auto_fix_results=auto_fix_results
        )

        # Save history
        self.diagnostic_history.append(report)
        if len(self.diagnostic_history) > 50:
            self.diagnostic_history = self.diagnostic_history[-25:]

        self.logger.info(f"Diagnostics completed: Found {len(issues)} issues")
        return report

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        try:
            return {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'python': {
                    'version': platform.python_version(),
                    'implementation': platform.python_implementation(),
                    'executable': sys.executable
                },
                'application': {
                    'data_directory': str(self.config.data_directory),
                    'database_path': str(self.db_manager.db_path),
                    'config_file': str(self.config.config_file) if hasattr(self.config, 'config_file') else None
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to collect system information: {e}")
            return {'error': str(e)}

    def _create_diagnostic_issue(self, issue_id: str, result: Dict[str, Any]) -> DiagnosticIssue:
        """Create diagnostic issue"""
        level_map = {
            'info': DiagnosticLevel.INFO,
            'warning': DiagnosticLevel.WARNING,
            'error': DiagnosticLevel.ERROR,
            'critical': DiagnosticLevel.CRITICAL
        }

        return DiagnosticIssue(
            issue_id=issue_id,
            level=level_map.get(result.get('level', 'info'), DiagnosticLevel.INFO),
            category=result.get('category', 'general'),
            description=result.get('description', ''),
            possible_causes=result.get('causes', []),
            suggested_solutions=result.get('solutions', []),
            affected_components=result.get('components', []),
            auto_fixable=result.get('auto_fixable', False)
        )

    def _generate_summary(self, issues: List[DiagnosticIssue]) -> Dict[str, int]:
        """Generate summary"""
        summary = {
            'total_issues': len(issues),
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'critical_count': 0,
            'auto_fixable_count': 0
        }

        for issue in issues:
            if issue.level == DiagnosticLevel.INFO:
                summary['info_count'] += 1
            elif issue.level == DiagnosticLevel.WARNING:
                summary['warning_count'] += 1
            elif issue.level == DiagnosticLevel.ERROR:
                summary['error_count'] += 1
            elif issue.level == DiagnosticLevel.CRITICAL:
                summary['critical_count'] += 1

            if issue.auto_fixable:
                summary['auto_fixable_count'] += 1

        return summary

    def _apply_auto_fixes(self, issues: List[DiagnosticIssue]) -> List[Dict[str, Any]]:
        """Apply auto-fixes"""
        results = []

        for issue in issues:
            if not issue.auto_fixable:
                continue

            # Find corresponding fix function
            for rule in self.diagnostic_rules:
                if rule['id'] == issue.issue_id and 'fix' in rule:
                    try:
                        fix_result = rule['fix']()
                        results.append({
                            'issue_id': issue.issue_id,
                            'success': fix_result.get('success', False),
                            'message': fix_result.get('message', '')
                        })
                        self.logger.info(f"Auto-fix {issue.issue_id}: {fix_result.get('message', '')}")
                    except Exception as e:
                        results.append({
                            'issue_id': issue.issue_id,
                            'success': False,
                            'message': str(e)
                        })
                        self.logger.error(f"Auto-fix failed {issue.issue_id}: {e}")

        return results

    # Diagnostic check implementations

    def _check_database_corruption(self) -> Optional[Dict[str, Any]]:
        """Check database corruption"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result[0] != 'ok':
                    return {
                        'level': 'critical',
                        'description': f"Database integrity check failed: {result[0]}",
                        'causes': [
                            'Database file corrupted',
                            'Disk error',
                            'Improper shutdown'
                        ],
                        'solutions': [
                            'Restore database from recent backup',
                            'Run database repair tool',
                            'Check disk health status'
                        ],
                        'components': ['database']
                    }

            return None

        except Exception as e:
            return {
                'level': 'error',
                'description': f"Cannot check database integrity: {e}",
                'causes': ['Database file inaccessible'],
                'solutions': ['Check file permissions', 'Verify database path is correct'],
                'components': ['database']
            }

    def _check_database_size(self) -> Optional[Dict[str, Any]]:
        """Check database size"""
        try:
            db_size = self.db_manager.db_path.stat().st_size
            size_mb = db_size / (1024 * 1024)

            if size_mb > 500:  # Over 500MB
                return {
                    'level': 'warning',
                    'description': f"Database file is large: {size_mb:.1f}MB",
                    'causes': [
                        'Large amount of stored memories',
                        'Lack of data cleanup',
                        'Database fragmentation'
                    ],
                    'solutions': [
                        'Run database VACUUM optimization',
                        'Clean up old memory data',
                        'Increase storage space'
                    ],
                    'components': ['database'],
                    'size_mb': size_mb
                }

            return None

        except Exception as e:
            return None

    def _fix_database_size(self) -> Dict[str, Any]:
        """Fix database size issue"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()

            return {
                'success': True,
                'message': 'Database VACUUM completed'
            }

        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }

    def _check_missing_directories(self) -> Optional[Dict[str, Any]]:
        """Check missing directories"""
        try:
            data_dir = Path(self.config.data_directory)
            required_dirs = ['conversations', 'contexts', 'backups', 'logs', 'cache']
            missing = []

            for dir_name in required_dirs:
                dir_path = data_dir / dir_name
                if not dir_path.exists():
                    missing.append(dir_name)

            if missing:
                return {
                    'level': 'warning',
                    'description': f"Missing required directories: {', '.join(missing)}",
                    'causes': [
                        'Incomplete first installation',
                        'Directories accidentally deleted'
                    ],
                    'solutions': [
                        'Create missing directories'
                    ],
                    'components': ['filesystem'],
                    'missing_dirs': missing
                }

            return None

        except Exception as e:
            return None

    def _fix_missing_directories(self) -> Dict[str, Any]:
        """Fix missing directories"""
        try:
            data_dir = Path(self.config.data_directory)
            required_dirs = ['conversations', 'contexts', 'backups', 'logs', 'cache']
            created = []

            for dir_name in required_dirs:
                dir_path = data_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(dir_name)

            return {
                'success': True,
                'message': f'Directories created: {", ".join(created)}'
            }

        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }

    def _check_permission_issues(self) -> Optional[Dict[str, Any]]:
        """Check permission issues"""
        try:
            data_dir = Path(self.config.data_directory)
            issues = []

            # Check data directory permissions
            if not os.access(data_dir, os.R_OK):
                issues.append(f"Data directory not readable: {data_dir}")
            if not os.access(data_dir, os.W_OK):
                issues.append(f"Data directory not writable: {data_dir}")

            # Check database file permissions
            db_path = self.db_manager.db_path
            if db_path.exists():
                if not os.access(db_path, os.R_OK):
                    issues.append(f"Database file not readable: {db_path}")
                if not os.access(db_path, os.W_OK):
                    issues.append(f"Database file not writable: {db_path}")

            if issues:
                return {
                    'level': 'error',
                    'description': f"Permission issues: {'; '.join(issues)}",
                    'causes': [
                        'Incorrect file permission settings',
                        'Insufficient user permissions'
                    ],
                    'solutions': [
                        'Check and fix file permissions',
                        'Run as administrator',
                        'Change data directory location'
                    ],
                    'components': ['filesystem']
                }

            return None

        except Exception as e:
            return None

    def _check_orphaned_files(self) -> Optional[Dict[str, Any]]:
        """Check orphaned files"""
        try:
            data_dir = Path(self.config.data_directory)
            orphaned_count = 0
            orphaned_size = 0

            # Check temporary files
            temp_files = list(data_dir.rglob('*.tmp')) + list(data_dir.rglob('*.temp'))
            orphaned_count += len(temp_files)
            orphaned_size += sum(f.stat().st_size for f in temp_files if f.exists())

            # Check old temporary directories
            temp_dirs = list(data_dir.glob('temp_*'))
            for temp_dir in temp_dirs:
                if temp_dir.is_dir():
                    for f in temp_dir.rglob('*'):
                        if f.is_file():
                            orphaned_count += 1
                            orphaned_size += f.stat().st_size

            if orphaned_count > 0:
                return {
                    'level': 'info',
                    'description': f"Found {orphaned_count} orphaned/temporary files ({orphaned_size / 1024:.1f}KB)",
                    'causes': [
                        'Interrupted operations',
                        'Incomplete cleanup tasks'
                    ],
                    'solutions': [
                        'Clean up orphaned files'
                    ],
                    'components': ['filesystem'],
                    'orphaned_count': orphaned_count,
                    'orphaned_size_kb': orphaned_size / 1024
                }

            return None

        except Exception as e:
            return None

    def _fix_orphaned_files(self) -> Dict[str, Any]:
        """Clean up orphaned files"""
        try:
            data_dir = Path(self.config.data_directory)
            removed_count = 0
            removed_size = 0

            # Clean up temporary files
            for pattern in ['*.tmp', '*.temp']:
                for f in data_dir.rglob(pattern):
                    try:
                        removed_size += f.stat().st_size
                        f.unlink()
                        removed_count += 1
                    except Exception:
                        pass

            # Clean up temporary directories
            for temp_dir in data_dir.glob('temp_*'):
                if temp_dir.is_dir():
                    try:
                        import shutil
                        for f in temp_dir.rglob('*'):
                            if f.is_file():
                                removed_size += f.stat().st_size
                                removed_count += 1
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

            return {
                'success': True,
                'message': f'Cleaned up {removed_count} files ({removed_size / 1024:.1f}KB)'
            }

        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }

    def _check_config_issues(self) -> Optional[Dict[str, Any]]:
        """Check configuration issues"""
        try:
            issues = []

            # Check key configurations
            if not self.config.data_directory:
                issues.append("Data directory not configured")

            # Check invalid configuration values
            if hasattr(self.config, 'max_memory_gb') and self.config.max_memory_gb < 0.1:
                issues.append("Memory limit configured too small")

            if issues:
                return {
                    'level': 'warning',
                    'description': f"Configuration issues: {'; '.join(issues)}",
                    'causes': [
                        'Configuration file corrupted',
                        'Manual editing error'
                    ],
                    'solutions': [
                        'Check and fix configuration file',
                        'Reset to default configuration'
                    ],
                    'components': ['configuration']
                }

            return None

        except Exception as e:
            return None

    def _check_memory_leak_hints(self) -> Optional[Dict[str, Any]]:
        """Check memory leak indicators"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # If memory usage exceeds 500MB, issue warning
            if memory_mb > 500:
                return {
                    'level': 'warning',
                    'description': f"High memory usage: {memory_mb:.1f}MB",
                    'causes': [
                        'Memory leak',
                        'Excessive caching',
                        'Processing large amounts of data'
                    ],
                    'solutions': [
                        'Restart application',
                        'Clear cache',
                        'Check memory usage patterns'
                    ],
                    'components': ['performance'],
                    'memory_mb': memory_mb
                }

            return None

        except ImportError:
            return None
        except Exception as e:
            return None

    def _check_old_backups(self) -> Optional[Dict[str, Any]]:
        """Check old backups"""
        try:
            backup_dir = Path(self.config.data_directory) / 'backups'
            if not backup_dir.exists():
                return None

            old_backups = []
            from datetime import timedelta
            old_threshold = datetime.now() - timedelta(days=90)

            for backup_file in backup_dir.glob('*.tar*'):
                try:
                    mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if mtime < old_threshold:
                        old_backups.append({
                            'file': backup_file.name,
                            'date': mtime.isoformat(),
                            'size_mb': backup_file.stat().st_size / (1024 * 1024)
                        })
                except Exception:
                    continue

            if old_backups:
                return {
                    'level': 'info',
                    'description': f"Found {len(old_backups)} backups older than 90 days",
                    'causes': [
                        'Backup cleanup policy not executed',
                        'Retention period set too long'
                    ],
                    'solutions': [
                        'Clean up old backups to free space',
                        'Adjust backup retention policy'
                    ],
                    'components': ['backup'],
                    'old_backups': old_backups
                }

            return None

        except Exception as e:
            return None

    def _fix_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups"""
        try:
            backup_dir = Path(self.config.data_directory) / 'backups'
            if not backup_dir.exists():
                return {'success': True, 'message': 'Backup directory does not exist'}

            from datetime import timedelta
            old_threshold = datetime.now() - timedelta(days=90)
            removed_count = 0
            removed_size = 0

            for backup_file in backup_dir.glob('*.tar*'):
                try:
                    mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if mtime < old_threshold:
                        removed_size += backup_file.stat().st_size
                        backup_file.unlink()
                        removed_count += 1
                except Exception:
                    continue

            return {
                'success': True,
                'message': f'Cleaned up {removed_count} old backups ({removed_size / (1024*1024):.1f}MB)'
            }

        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }

    def get_diagnostic_history(self, limit: int = 10) -> List[DiagnosticReport]:
        """Get diagnostic history"""
        return self.diagnostic_history[-limit:]

    def add_custom_diagnostic_rule(self, rule_id: str, check_func: callable,
                                  category: str = 'custom', auto_fix: bool = False,
                                  fix_func: Optional[callable] = None):
        """
        Add custom diagnostic rule

        Args:
            rule_id: Rule ID
            check_func: Check function
            category: Category
            auto_fix: Whether it can be auto-fixed
            fix_func: Fix function
        """
        rule = {
            'id': rule_id,
            'check': check_func,
            'category': category,
            'auto_fix': auto_fix
        }

        if auto_fix and fix_func:
            rule['fix'] = fix_func

        self.diagnostic_rules.append(rule)
        self.logger.info(f"Custom diagnostic rule added: {rule_id}")