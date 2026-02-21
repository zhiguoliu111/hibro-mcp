#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backup Manager
Provides automated backup functionality for database and filesystem
"""

import os
import shutil
import sqlite3
import tarfile
import gzip
import json
import logging
import threading
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.config import Config
from ..security.encryption import EncryptionManager


class BackupStrategy(Enum):
    """Backup strategy enumeration"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


class SyncStatus(Enum):
    """Synchronization status enumeration"""
    SYNCED = "synced"
    PENDING = "pending"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class BackupInfo:
    """Backup information"""
    backup_id: str
    backup_type: str  # full, incremental, differential
    created_at: datetime
    file_path: Path
    size_bytes: int
    compressed: bool
    encrypted: bool
    checksum: str
    metadata: Dict[str, Any]
    status: BackupStatus = BackupStatus.COMPLETED
    sync_status: SyncStatus = SyncStatus.SYNCED
    device_id: Optional[str] = None
    parent_backup_id: Optional[str] = None
    integrity_verified: bool = False
    last_verified_at: Optional[datetime] = None
    retention_policy: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class RestoreInfo:
    """Restore information"""
    restore_id: str
    backup_id: str
    restore_type: str  # full, selective, point_in_time
    target_path: Path
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"
    progress_percentage: float = 0.0
    restored_files_count: int = 0
    total_files_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationInfo:
    """Migration information"""
    migration_id: str
    source_device: str
    target_device: str
    migration_type: str  # full, selective, incremental
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"
    progress_percentage: float = 0.0
    transferred_size_bytes: int = 0
    total_size_bytes: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncDevice:
    """Synchronization device information"""
    device_id: str
    device_name: str
    device_type: str  # desktop, laptop, mobile, server
    last_sync_at: Optional[datetime] = None
    sync_enabled: bool = True
    sync_path: Optional[Path] = None
    connection_status: str = "offline"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupManager:
    """Backup manager"""

    def __init__(self, config: Config, encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize backup manager

        Args:
            config: Configuration object
            encryption_manager: Encryption manager
        """
        self.config = config
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger('hibro.backup_manager')

        # Backup configuration
        self.backup_config = {
            'backup_directory': Path(config.data_directory) / 'backups',
            'max_backups': 30,
            'compress_backups': True,
            'encrypt_backups': True,
            'auto_backup_enabled': True,
            'auto_backup_interval_hours': 24,
            'incremental_backup_enabled': True,
            'differential_backup_enabled': True,
            'backup_retention_days': 90,
            'verify_backups': True,
            'parallel_backup_threads': 4,
            'backup_integrity_check_interval_hours': 168,  # Check once per week
            'disaster_recovery_enabled': True,
            'cross_device_sync_enabled': False,
            'backup_compression_level': 6,
            'backup_notification_enabled': True
        }

        # Create backup directory
        self.backup_config['backup_directory'].mkdir(parents=True, exist_ok=True)

        # Backup history
        self.backup_history: List[BackupInfo] = []
        self._load_backup_history()

        # Enterprise-level enhanced features
        self.restore_manager = RestoreManager(config, encryption_manager)
        self.migration_manager = MigrationManager(config, encryption_manager)

        # Auto backup scheduler
        self._backup_scheduler_thread = None
        self._scheduler_running = False

        # Integrity check scheduler
        self._integrity_checker_thread = None
        self._integrity_checker_running = False

        # Start automated services
        self._start_automated_services()

    def _load_backup_history(self):
        """Load backup history"""
        try:
            history_file = self.backup_config['backup_directory'] / 'backup_history.json'
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                for item in history_data:
                    backup_info = BackupInfo(
                        backup_id=item['backup_id'],
                        backup_type=item['backup_type'],
                        created_at=datetime.fromisoformat(item['created_at']),
                        file_path=Path(item['file_path']),
                        size_bytes=item['size_bytes'],
                        compressed=item['compressed'],
                        encrypted=item['encrypted'],
                        checksum=item['checksum'],
                        metadata=item.get('metadata', {})
                    )
                    self.backup_history.append(backup_info)

                self.logger.info(f"Loaded {len(self.backup_history)} backup history records")

        except Exception as e:
            self.logger.warning(f"Failed to load backup history: {e}")

    def _start_automated_services(self):
        """Start automated services"""
        try:
            if self.backup_config['auto_backup_enabled']:
                self._start_backup_scheduler()

            if self.backup_config['backup_integrity_check_interval_hours'] > 0:
                self._start_integrity_checker()

            self.logger.info("Automated backup service started")

        except Exception as e:
            self.logger.error(f"Failed to start automated service: {e}")

    def _start_backup_scheduler(self):
        """Start backup scheduler"""
        if self._backup_scheduler_thread and self._backup_scheduler_thread.is_alive():
            return

        self._scheduler_running = True
        self._backup_scheduler_thread = threading.Thread(target=self._backup_scheduler_worker, daemon=True)
        self._backup_scheduler_thread.start()
        self.logger.info("Backup scheduler started")

    def _backup_scheduler_worker(self):
        """Backup scheduler worker thread"""
        while self._scheduler_running:
            try:
                # Check if backup needs to be executed
                if self._should_create_backup():
                    self.logger.info("Auto backup triggered")
                    if self.backup_config['incremental_backup_enabled']:
                        self.create_incremental_backup(description="Auto incremental backup")
                    else:
                        self.create_full_backup(description="Auto full backup")

                # Wait for next check
                time.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Backup scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes after error

    def _start_integrity_checker(self):
        """Start integrity checker"""
        if self._integrity_checker_thread and self._integrity_checker_thread.is_alive():
            return

        self._integrity_checker_running = True
        self._integrity_checker_thread = threading.Thread(target=self._integrity_checker_worker, daemon=True)
        self._integrity_checker_thread.start()
        self.logger.info("Integrity checker started")

    def _integrity_checker_worker(self):
        """Integrity checker worker thread"""
        while self._integrity_checker_running:
            try:
                # Check integrity of all backups
                self._check_all_backups_integrity()

                # Wait for next check
                interval_seconds = self.backup_config['backup_integrity_check_interval_hours'] * 3600
                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Integrity checker error: {e}")
                time.sleep(3600)  # Wait 1 hour after error

    def _should_create_backup(self) -> bool:
        """Check if backup should be created"""
        try:
            if not self.backup_history:
                return True

            latest_backup = max(self.backup_history, key=lambda x: x.created_at)
            time_since_last = datetime.now() - latest_backup.created_at
            interval_hours = self.backup_config['auto_backup_interval_hours']

            return time_since_last.total_seconds() >= interval_hours * 3600

        except Exception as e:
            self.logger.error(f"Failed to check backup condition: {e}")
            return False

    def _check_all_backups_integrity(self):
        """Check integrity of all backups"""
        try:
            corrupted_backups = []

            for backup_info in self.backup_history:
                if not self.verify_backup(backup_info.backup_id):
                    backup_info.status = BackupStatus.CORRUPTED
                    corrupted_backups.append(backup_info.backup_id)

            if corrupted_backups:
                self.logger.warning(f"Found corrupted backups: {corrupted_backups}")
                # Notification mechanism can be added here

            self._save_backup_history()

        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")

    def create_differential_backup(self, base_backup_id: Optional[str] = None,
                                 description: Optional[str] = None) -> Optional[BackupInfo]:
        """
        Create differential backup

        Args:
            base_backup_id: Base backup ID
            description: Backup description

        Returns:
            Backup information
        """
        try:
            if not self.backup_config['differential_backup_enabled']:
                self.logger.warning("Differential backup is disabled")
                return None

            # Find base backup
            base_backup = None
            if base_backup_id:
                base_backup = next((b for b in self.backup_history if b.backup_id == base_backup_id), None)
            else:
                # Use latest full backup as base
                full_backups = [b for b in self.backup_history if b.backup_type == 'full']
                if full_backups:
                    base_backup = max(full_backups, key=lambda x: x.created_at)

            if not base_backup:
                self.logger.warning("Base backup not found, creating full backup")
                return self.create_full_backup(description)

            backup_id = f"diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_filename = f"{backup_id}.tar"

            if self.backup_config['compress_backups']:
                backup_filename += ".gz"

            backup_path = self.backup_config['backup_directory'] / backup_filename

            self.logger.info(f"Starting differential backup creation: {backup_id}")

            # Create temporary directory
            temp_dir = self.backup_config['backup_directory'] / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)

            try:
                # Get all changes since base backup
                changed_files = self._get_changed_files_since(base_backup.created_at)

                if not changed_files:
                    self.logger.info("No file changes, skipping differential backup")
                    return None

                # Backup changed files
                changes_dir = temp_dir / 'changes'
                changes_dir.mkdir()

                for file_path in changed_files:
                    try:
                        rel_path = file_path.relative_to(Path(self.config.data_directory))
                        dest_path = changes_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to backup file {file_path}: {e}")

                # Create differential backup metadata
                metadata = {
                    'backup_type': 'differential',
                    'base_backup_id': base_backup.backup_id,
                    'description': description or f'Differential backup {datetime.now()}',
                    'changed_files_count': len(changed_files),
                    'base_backup_date': base_backup.created_at.isoformat(),
                    'created_by': 'hibro_backup_manager',
                    'version': '1.0'
                }

                metadata_file = temp_dir / 'backup_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Create compressed archive
                self._create_archive(temp_dir, backup_path)

                # Encrypt backup file
                if self.backup_config['encrypt_backups'] and self.encryption_manager:
                    encrypted_path = self._encrypt_backup(backup_path)
                    if encrypted_path:
                        backup_path.unlink()
                        backup_path = encrypted_path

                # Calculate checksum
                checksum = self._calculate_checksum(backup_path)

                # Create backup information
                backup_info = BackupInfo(
                    backup_id=backup_id,
                    backup_type='differential',
                    created_at=datetime.now(),
                    file_path=backup_path,
                    size_bytes=backup_path.stat().st_size,
                    compressed=self.backup_config['compress_backups'],
                    encrypted=self.backup_config['encrypt_backups'] and self.encryption_manager is not None,
                    checksum=checksum,
                    metadata=metadata,
                    status=BackupStatus.COMPLETED,
                    parent_backup_id=base_backup.backup_id
                )

                # Add to history
                self.backup_history.append(backup_info)
                self._save_backup_history()

                self.logger.info(f"Differential backup created successfully: {backup_path}")
                return backup_info

            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Failed to create differential backup: {e}")
            return None

    def create_parallel_backup(self, backup_type: str = "full", description: Optional[str] = None) -> Optional[BackupInfo]:
        """
        Create parallel backup (multi-threaded)

        Args:
            backup_type: Backup type
            description: Backup description

        Returns:
            Backup information
        """
        try:
            backup_id = f"parallel_{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_filename = f"{backup_id}.tar"

            if self.backup_config['compress_backups']:
                backup_filename += ".gz"

            backup_path = self.backup_config['backup_directory'] / backup_filename

            self.logger.info(f"Starting parallel backup creation: {backup_id}")

            # Create temporary directory
            temp_dir = self.backup_config['backup_directory'] / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)

            try:
                # Parallel backup different components
                with ThreadPoolExecutor(max_workers=self.backup_config['parallel_backup_threads']) as executor:
                    futures = []

                    # Submit backup tasks
                    futures.append(executor.submit(self._backup_database, temp_dir))
                    futures.append(executor.submit(self._backup_filesystem, temp_dir))
                    futures.append(executor.submit(self._backup_configuration, temp_dir))

                    # Wait for all tasks to complete
                    backup_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                backup_results.append(result)
                        except Exception as e:
                            self.logger.error(f"Parallel backup task failed: {e}")

                # Create backup metadata
                metadata = {
                    'backup_type': backup_type,
                    'description': description or f'Parallel {backup_type} backup {datetime.now()}',
                    'parallel_threads': self.backup_config['parallel_backup_threads'],
                    'backup_components': len(backup_results),
                    'created_by': 'hibro_backup_manager',
                    'version': '1.0'
                }

                metadata_file = temp_dir / 'backup_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Create compressed archive
                self._create_archive(temp_dir, backup_path)

                # Encrypt backup file
                if self.backup_config['encrypt_backups'] and self.encryption_manager:
                    encrypted_path = self._encrypt_backup(backup_path)
                    if encrypted_path:
                        backup_path.unlink()
                        backup_path = encrypted_path

                # Calculate checksum
                checksum = self._calculate_checksum(backup_path)

                # Create backup information
                backup_info = BackupInfo(
                    backup_id=backup_id,
                    backup_type=backup_type,
                    created_at=datetime.now(),
                    file_path=backup_path,
                    size_bytes=backup_path.stat().st_size,
                    compressed=self.backup_config['compress_backups'],
                    encrypted=self.backup_config['encrypt_backups'] and self.encryption_manager is not None,
                    checksum=checksum,
                    metadata=metadata,
                    status=BackupStatus.COMPLETED
                )

                # Add to history
                self.backup_history.append(backup_info)
                self._save_backup_history()

                # Clean up old backups
                self._cleanup_old_backups()

                self.logger.info(f"Parallel backup created successfully: {backup_path}")
                return backup_info

            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Failed to create parallel backup: {e}")
            return None

    def create_disaster_recovery_backup(self, target_location: Optional[Path] = None) -> Optional[BackupInfo]:
        """
        Create disaster recovery backup

        Args:
            target_location: Target location

        Returns:
            Backup information
        """
        try:
            if not self.backup_config['disaster_recovery_enabled']:
                self.logger.warning("Disaster recovery backup is disabled")
                return None

            backup_id = f"dr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Use external location or default disaster recovery directory
            if target_location:
                dr_backup_dir = target_location
            else:
                dr_backup_dir = self.backup_config['backup_directory'] / 'disaster_recovery'

            dr_backup_dir.mkdir(parents=True, exist_ok=True)

            backup_filename = f"{backup_id}.tar.gz.enc"
            backup_path = dr_backup_dir / backup_filename

            self.logger.info(f"Starting disaster recovery backup creation: {backup_id}")

            # Create full backup
            full_backup = self.create_full_backup(description=f"Disaster recovery backup {datetime.now()}")
            if not full_backup:
                raise Exception("Failed to create full backup")

            # Copy to disaster recovery location
            shutil.copy2(full_backup.file_path, backup_path)

            # Create disaster recovery metadata
            dr_metadata = {
                'disaster_recovery': True,
                'original_backup_id': full_backup.backup_id,
                'recovery_instructions': {
                    'step1': 'Decrypt backup file',
                    'step2': 'Extract backup file',
                    'step3': 'Restore database',
                    'step4': 'Restore filesystem',
                    'step5': 'Restore configuration files'
                },
                'system_info': {
                    'platform': 'win32',
                    'hibro_version': '2.0.0',
                    'backup_format_version': '1.0'
                }
            }

            dr_metadata_file = dr_backup_dir / f"{backup_id}_recovery_info.json"
            with open(dr_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(dr_metadata, f, indent=2, ensure_ascii=False)

            # Update backup information
            full_backup.metadata.update(dr_metadata)
            full_backup.tags.append('disaster_recovery')

            self.logger.info(f"Disaster recovery backup created successfully: {backup_path}")
            return full_backup

        except Exception as e:
            self.logger.error(f"Failed to create disaster recovery backup: {e}")
            return None

    def get_enhanced_backup_statistics(self) -> Dict[str, Any]:
        """Get enhanced backup statistics"""
        try:
            basic_stats = self.get_backup_statistics()

            # Add enterprise-level statistics
            differential_backups = len([b for b in self.backup_history if b.backup_type == 'differential'])
            corrupted_backups = len([b for b in self.backup_history if b.status == BackupStatus.CORRUPTED])

            # Calculate backup efficiency
            total_size = sum(b.size_bytes for b in self.backup_history)
            full_backup_size = sum(b.size_bytes for b in self.backup_history if b.backup_type == 'full')
            compression_ratio = 0
            if full_backup_size > 0:
                compression_ratio = (full_backup_size - total_size) / full_backup_size * 100

            # Restore and migration statistics
            restore_stats = {
                'total_restores': len(self.restore_manager.restore_history),
                'successful_restores': len([r for r in self.restore_manager.restore_history if r.status == 'completed']),
                'failed_restores': len([r for r in self.restore_manager.restore_history if r.status == 'failed'])
            }

            migration_stats = {
                'total_migrations': len(self.migration_manager.migration_history),
                'successful_migrations': len([m for m in self.migration_manager.migration_history if m.status == 'completed']),
                'registered_devices': len(self.migration_manager.sync_devices)
            }

            enhanced_stats = {
                **basic_stats,
                'differential_backups': differential_backups,
                'corrupted_backups': corrupted_backups,
                'compression_ratio_percent': compression_ratio,
                'backup_health_score': self._calculate_backup_health_score(),
                'restore_statistics': restore_stats,
                'migration_statistics': migration_stats,
                'automated_services': {
                    'backup_scheduler_running': self._scheduler_running,
                    'integrity_checker_running': self._integrity_checker_running,
                    'auto_backup_enabled': self.backup_config['auto_backup_enabled'],
                    'disaster_recovery_enabled': self.backup_config['disaster_recovery_enabled']
                }
            }

            return enhanced_stats

        except Exception as e:
            self.logger.error(f"Failed to get enhanced backup statistics: {e}")
            return self.get_backup_statistics()

    def _calculate_backup_health_score(self) -> float:
        """Calculate backup health score"""
        try:
            if not self.backup_history:
                return 0.0

            score = 100.0

            # Check backup frequency
            latest_backup = max(self.backup_history, key=lambda x: x.created_at)
            days_since_last = (datetime.now() - latest_backup.created_at).days
            if days_since_last > 7:
                score -= 20
            elif days_since_last > 3:
                score -= 10

            # Check backup integrity
            corrupted_count = len([b for b in self.backup_history if b.status == BackupStatus.CORRUPTED])
            if corrupted_count > 0:
                score -= corrupted_count * 15

            # Check backup diversity
            backup_types = set(b.backup_type for b in self.backup_history)
            if len(backup_types) < 2:
                score -= 10

            # Check encryption status
            encrypted_count = len([b for b in self.backup_history if b.encrypted])
            encryption_ratio = encrypted_count / len(self.backup_history)
            if encryption_ratio < 0.8:
                score -= 15

            return max(0.0, min(100.0, score))

        except Exception as e:
            self.logger.error(f"Failed to calculate backup health score: {e}")
            return 0.0

    def stop_automated_services(self):
        """Stop automated services"""
        try:
            self._scheduler_running = False
            self._integrity_checker_running = False

            if self._backup_scheduler_thread and self._backup_scheduler_thread.is_alive():
                self._backup_scheduler_thread.join(timeout=5)

            if self._integrity_checker_thread and self._integrity_checker_thread.is_alive():
                self._integrity_checker_thread.join(timeout=5)

            self.logger.info("Automated backup service stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop automated service: {e}")

    # Integration methods
    def restore_backup(self, backup_id: str, target_path: Optional[Path] = None,
                      restore_type: str = "full") -> Optional[RestoreInfo]:
        """Restore backup (integrated with RestoreManager)"""
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            self.logger.error(f"Backup does not exist: {backup_id}")
            return None

        if restore_type == "full":
            return self.restore_manager.restore_full_backup(backup_info, target_path)
        else:
            self.logger.error(f"Unsupported restore type: {restore_type}")
            return None

    def start_device_migration(self, source_device_id: str, target_device_id: str) -> Optional[MigrationInfo]:
        """Start device migration (integrated with MigrationManager)"""
        return self.migration_manager.start_cross_device_sync(source_device_id, target_device_id)

    def register_device(self, device_name: str, device_type: str, sync_path: Optional[Path] = None) -> str:
        """Register device (integrated with MigrationManager)"""
        return self.migration_manager.register_sync_device(device_name, device_type, sync_path)

    def _save_backup_history(self):
        """Save backup history"""
        try:
            history_file = self.backup_config['backup_directory'] / 'backup_history.json'
            history_data = []

            for backup_info in self.backup_history:
                history_data.append({
                    'backup_id': backup_info.backup_id,
                    'backup_type': backup_info.backup_type,
                    'created_at': backup_info.created_at.isoformat(),
                    'file_path': str(backup_info.file_path),
                    'size_bytes': backup_info.size_bytes,
                    'compressed': backup_info.compressed,
                    'encrypted': backup_info.encrypted,
                    'checksum': backup_info.checksum,
                    'metadata': backup_info.metadata
                })

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save backup history: {e}")

    def create_full_backup(self, description: Optional[str] = None) -> Optional[BackupInfo]:
        """
        Create full backup

        Args:
            description: Backup description

        Returns:
            Backup information
        """
        try:
            backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_filename = f"{backup_id}.tar"

            if self.backup_config['compress_backups']:
                backup_filename += ".gz"

            backup_path = self.backup_config['backup_directory'] / backup_filename

            self.logger.info(f"Starting full backup creation: {backup_id}")

            # Create temporary directory
            temp_dir = self.backup_config['backup_directory'] / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)

            try:
                # Backup database
                db_backup_path = self._backup_database(temp_dir)

                # Backup filesystem
                fs_backup_path = self._backup_filesystem(temp_dir)

                # Backup configuration files
                config_backup_path = self._backup_configuration(temp_dir)

                # Create backup metadata
                metadata = {
                    'backup_type': 'full',
                    'description': description or f'Full backup {datetime.now()}',
                    'database_backup': str(db_backup_path.name) if db_backup_path else None,
                    'filesystem_backup': str(fs_backup_path.name) if fs_backup_path else None,
                    'config_backup': str(config_backup_path.name) if config_backup_path else None,
                    'created_by': 'hibro_backup_manager',
                    'version': '1.0'
                }

                metadata_file = temp_dir / 'backup_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Create compressed archive
                self._create_archive(temp_dir, backup_path)

                # Encrypt backup file
                if self.backup_config['encrypt_backups'] and self.encryption_manager:
                    encrypted_path = self._encrypt_backup(backup_path)
                    if encrypted_path:
                        backup_path.unlink()  # Delete unencrypted version
                        backup_path = encrypted_path

                # Calculate checksum
                checksum = self._calculate_checksum(backup_path)

                # Create backup information
                backup_info = BackupInfo(
                    backup_id=backup_id,
                    backup_type='full',
                    created_at=datetime.now(),
                    file_path=backup_path,
                    size_bytes=backup_path.stat().st_size,
                    compressed=self.backup_config['compress_backups'],
                    encrypted=self.backup_config['encrypt_backups'] and self.encryption_manager is not None,
                    checksum=checksum,
                    metadata=metadata
                )

                # Add to history
                self.backup_history.append(backup_info)
                self._save_backup_history()

                # Clean up old backups
                self._cleanup_old_backups()

                self.logger.info(f"Full backup created successfully: {backup_path}")
                return backup_info

            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Failed to create full backup: {e}")
            return None

    def create_incremental_backup(self, base_backup_id: Optional[str] = None,
                                description: Optional[str] = None) -> Optional[BackupInfo]:
        """
        Create incremental backup

        Args:
            base_backup_id: Base backup ID
            description: Backup description

        Returns:
            Backup information
        """
        try:
            if not self.backup_config['incremental_backup_enabled']:
                self.logger.warning("Incremental backup is disabled")
                return None

            # Find base backup
            base_backup = None
            if base_backup_id:
                base_backup = next((b for b in self.backup_history if b.backup_id == base_backup_id), None)
            else:
                # Use latest full backup as base
                full_backups = [b for b in self.backup_history if b.backup_type == 'full']
                if full_backups:
                    base_backup = max(full_backups, key=lambda x: x.created_at)

            if not base_backup:
                self.logger.warning("Base backup not found, creating full backup")
                return self.create_full_backup(description)

            backup_id = f"inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_filename = f"{backup_id}.tar"

            if self.backup_config['compress_backups']:
                backup_filename += ".gz"

            backup_path = self.backup_config['backup_directory'] / backup_filename

            self.logger.info(f"Starting incremental backup creation: {backup_id}")

            # Create temporary directory
            temp_dir = self.backup_config['backup_directory'] / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)

            try:
                # Get changed files
                changed_files = self._get_changed_files_since(base_backup.created_at)

                if not changed_files:
                    self.logger.info("No file changes, skipping incremental backup")
                    return None

                # Backup changed files
                changes_dir = temp_dir / 'changes'
                changes_dir.mkdir()

                for file_path in changed_files:
                    try:
                        rel_path = file_path.relative_to(Path(self.config.data_directory))
                        dest_path = changes_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to backup file {file_path}: {e}")

                # Create incremental backup metadata
                metadata = {
                    'backup_type': 'incremental',
                    'base_backup_id': base_backup.backup_id,
                    'description': description or f'Incremental backup {datetime.now()}',
                    'changed_files_count': len(changed_files),
                    'base_backup_date': base_backup.created_at.isoformat(),
                    'created_by': 'hibro_backup_manager',
                    'version': '1.0'
                }

                metadata_file = temp_dir / 'backup_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Create compressed archive
                self._create_archive(temp_dir, backup_path)

                # Encrypt backup file
                if self.backup_config['encrypt_backups'] and self.encryption_manager:
                    encrypted_path = self._encrypt_backup(backup_path)
                    if encrypted_path:
                        backup_path.unlink()
                        backup_path = encrypted_path

                # Calculate checksum
                checksum = self._calculate_checksum(backup_path)

                # Create backup information
                backup_info = BackupInfo(
                    backup_id=backup_id,
                    backup_type='incremental',
                    created_at=datetime.now(),
                    file_path=backup_path,
                    size_bytes=backup_path.stat().st_size,
                    compressed=self.backup_config['compress_backups'],
                    encrypted=self.backup_config['encrypt_backups'] and self.encryption_manager is not None,
                    checksum=checksum,
                    metadata=metadata
                )

                # Add to history
                self.backup_history.append(backup_info)
                self._save_backup_history()

                self.logger.info(f"Incremental backup created successfully: {backup_path}")
                return backup_info

            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Failed to create incremental backup: {e}")
            return None

    def _backup_database(self, backup_dir: Path) -> Optional[Path]:
        """Backup database"""
        try:
            db_path = Path(self.config.data_directory) / 'hibro.db'
            if not db_path.exists():
                self.logger.warning("Database file does not exist")
                return None

            backup_db_path = backup_dir / 'database.db'

            # Use SQLite backup API
            with sqlite3.connect(db_path) as source_conn:
                with sqlite3.connect(backup_db_path) as backup_conn:
                    source_conn.backup(backup_conn)

            self.logger.info(f"Database backup completed: {backup_db_path}")
            return backup_db_path

        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return None

    def _backup_filesystem(self, backup_dir: Path) -> Optional[Path]:
        """Backup filesystem"""
        try:
            data_dir = Path(self.config.data_directory)
            fs_backup_dir = backup_dir / 'filesystem'
            fs_backup_dir.mkdir()

            # Backup important directories
            important_dirs = ['conversations', 'contexts', 'exports', 'logs']

            for dir_name in important_dirs:
                source_dir = data_dir / dir_name
                if source_dir.exists():
                    dest_dir = fs_backup_dir / dir_name
                    shutil.copytree(source_dir, dest_dir, ignore_dangling_symlinks=True)

            self.logger.info(f"Filesystem backup completed: {fs_backup_dir}")
            return fs_backup_dir

        except Exception as e:
            self.logger.error(f"Filesystem backup failed: {e}")
            return None

    def _backup_configuration(self, backup_dir: Path) -> Optional[Path]:
        """Backup configuration files"""
        try:
            config_backup_dir = backup_dir / 'config'
            config_backup_dir.mkdir()

            # Backup configuration files
            config_files = [
                'config.yaml',
                'automation_rules.yaml',
                'user_preferences.json'
            ]

            data_dir = Path(self.config.data_directory)
            for config_file in config_files:
                source_file = data_dir / config_file
                if source_file.exists():
                    dest_file = config_backup_dir / config_file
                    shutil.copy2(source_file, dest_file)

            self.logger.info(f"Configuration backup completed: {config_backup_dir}")
            return config_backup_dir

        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            return None

    def _create_archive(self, source_dir: Path, archive_path: Path):
        """Create compressed archive"""
        try:
            if self.backup_config['compress_backups']:
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(source_dir, arcname='.')
            else:
                with tarfile.open(archive_path, 'w') as tar:
                    tar.add(source_dir, arcname='.')

            self.logger.debug(f"Archive created successfully: {archive_path}")

        except Exception as e:
            self.logger.error(f"Failed to create archive: {e}")
            raise

    def _encrypt_backup(self, backup_path: Path) -> Optional[Path]:
        """Encrypt backup file"""
        try:
            if not self.encryption_manager or not self.encryption_manager.is_unlocked():
                self.logger.warning("Encryption manager not available, skipping encryption")
                return None

            encrypted_path = backup_path.with_suffix(backup_path.suffix + '.enc')
            return self.encryption_manager.encrypt_file(backup_path, encrypted_path, 'backup')

        except Exception as e:
            self.logger.error(f"Failed to encrypt backup file: {e}")
            return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            import hashlib

            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            return hash_md5.hexdigest()

        except Exception as e:
            self.logger.error(f"Failed to calculate checksum: {e}")
            return ""

    def _get_changed_files_since(self, since_time: datetime) -> List[Path]:
        """Get files changed after specified time"""
        changed_files = []

        try:
            data_dir = Path(self.config.data_directory)

            for file_path in data_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime > since_time:
                            changed_files.append(file_path)
                    except Exception:
                        continue

        except Exception as e:
            self.logger.error(f"Failed to get changed files: {e}")

        return changed_files

    def _cleanup_old_backups(self):
        """Clean up old backups"""
        try:
            # Sort by creation time
            sorted_backups = sorted(self.backup_history, key=lambda x: x.created_at, reverse=True)

            # Keep latest backups
            backups_to_keep = sorted_backups[:self.backup_config['max_backups']]
            backups_to_remove = sorted_backups[self.backup_config['max_backups']:]

            # Filter by retention period
            retention_cutoff = datetime.now() - timedelta(days=self.backup_config['backup_retention_days'])
            backups_to_remove.extend([
                b for b in backups_to_keep
                if b.created_at < retention_cutoff
            ])

            # Delete expired backups
            for backup_info in backups_to_remove:
                try:
                    if backup_info.file_path.exists():
                        backup_info.file_path.unlink()
                        self.logger.info(f"Deleted expired backup: {backup_info.backup_id}")

                    self.backup_history.remove(backup_info)

                except Exception as e:
                    self.logger.warning(f"Failed to delete backup {backup_info.backup_id}: {e}")

            if backups_to_remove:
                self._save_backup_history()

        except Exception as e:
            self.logger.error(f"Failed to clean up old backups: {e}")

    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity

        Args:
            backup_id: Backup ID

        Returns:
            Whether verification succeeded
        """
        try:
            backup_info = next((b for b in self.backup_history if b.backup_id == backup_id), None)
            if not backup_info:
                self.logger.error(f"Backup does not exist: {backup_id}")
                return False

            if not backup_info.file_path.exists():
                self.logger.error(f"Backup file does not exist: {backup_info.file_path}")
                return False

            # Verify checksum
            current_checksum = self._calculate_checksum(backup_info.file_path)
            if current_checksum != backup_info.checksum:
                self.logger.error(f"Backup checksum mismatch: {backup_id}")
                return False

            # Verify archive integrity
            if backup_info.compressed:
                try:
                    file_path = backup_info.file_path

                    # If encrypted file, decrypt first
                    if backup_info.encrypted and self.encryption_manager and self.encryption_manager.is_unlocked():
                        temp_file = file_path.with_suffix('.temp')
                        try:
                            decrypted_file = self.encryption_manager.decrypt_file(file_path, temp_file, 'backup')
                            if decrypted_file:
                                file_path = decrypted_file
                        except Exception as decrypt_error:
                            self.logger.error(f"Failed to decrypt backup file: {decrypt_error}")
                            return False

                    # Verify archive
                    with tarfile.open(file_path, 'r:gz' if backup_info.compressed else 'r') as tar:
                        tar.getnames()  # Try to read file list

                    # Clean up temporary file
                    if backup_info.encrypted and file_path != backup_info.file_path:
                        file_path.unlink()

                except Exception as e:
                    self.logger.error(f"Archive verification failed: {e}")
                    return False

            self.logger.info(f"Backup verification successful: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False

    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupInfo]:
        """
        List backups

        Args:
            backup_type: Backup type filter

        Returns:
            List of backup information
        """
        if backup_type:
            return [b for b in self.backup_history if b.backup_type == backup_type]
        return self.backup_history.copy()

    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """
        Get backup information

        Args:
            backup_id: Backup ID

        Returns:
            Backup information
        """
        return next((b for b in self.backup_history if b.backup_id == backup_id), None)

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete backup

        Args:
            backup_id: Backup ID

        Returns:
            Whether deletion was successful
        """
        try:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                self.logger.error(f"Backup does not exist: {backup_id}")
                return False

            # Delete backup file
            if backup_info.file_path.exists():
                backup_info.file_path.unlink()

            # Remove from history
            self.backup_history.remove(backup_info)
            self._save_backup_history()

            self.logger.info(f"Backup deleted: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete backup: {e}")
            return False

    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics"""
        try:
            total_backups = len(self.backup_history)
            full_backups = len([b for b in self.backup_history if b.backup_type == 'full'])
            incremental_backups = len([b for b in self.backup_history if b.backup_type == 'incremental'])

            total_size = sum(b.size_bytes for b in self.backup_history)
            latest_backup = max(self.backup_history, key=lambda x: x.created_at) if self.backup_history else None

            return {
                'total_backups': total_backups,
                'full_backups': full_backups,
                'incremental_backups': incremental_backups,
                'total_size_mb': total_size / (1024 * 1024),
                'latest_backup': {
                    'backup_id': latest_backup.backup_id,
                    'created_at': latest_backup.created_at.isoformat(),
                    'type': latest_backup.backup_type,
                    'size_mb': latest_backup.size_bytes / (1024 * 1024)
                } if latest_backup else None,
                'config': self.backup_config.copy()
            }

        except Exception as e:
            self.logger.error(f"Failed to get backup statistics: {e}")
            return {}

    def update_backup_config(self, **kwargs) -> bool:
        """
        Update backup configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.backup_config:
                    self.backup_config[key] = value
                    self.logger.info(f"Backup configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update backup configuration: {e}")
            return False


class RestoreManager:
    """Restore manager - Enterprise-level data restore functionality"""

    def __init__(self, config: Config, encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize restore manager

        Args:
            config: Configuration object
            encryption_manager: Encryption manager
        """
        self.config = config
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger('hibro.restore_manager')

        # Restore history
        self.restore_history: List[RestoreInfo] = []
        self._load_restore_history()

    def _load_restore_history(self):
        """Load restore history"""
        try:
            history_file = Path(self.config.data_directory) / 'backups' / 'restore_history.json'
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                for item in history_data:
                    restore_info = RestoreInfo(
                        restore_id=item['restore_id'],
                        backup_id=item['backup_id'],
                        restore_type=item['restore_type'],
                        target_path=Path(item['target_path']),
                        created_at=datetime.fromisoformat(item['created_at']),
                        completed_at=datetime.fromisoformat(item['completed_at']) if item.get('completed_at') else None,
                        status=item.get('status', 'pending'),
                        progress_percentage=item.get('progress_percentage', 0.0),
                        restored_files_count=item.get('restored_files_count', 0),
                        total_files_count=item.get('total_files_count', 0),
                        error_message=item.get('error_message'),
                        metadata=item.get('metadata', {})
                    )
                    self.restore_history.append(restore_info)

                self.logger.info(f"Loaded {len(self.restore_history)} restore history records")

        except Exception as e:
            self.logger.warning(f"Failed to load restore history: {e}")

    def _save_restore_history(self):
        """Save restore history"""
        try:
            history_file = Path(self.config.data_directory) / 'backups' / 'restore_history.json'
            history_data = []

            for restore_info in self.restore_history:
                history_data.append({
                    'restore_id': restore_info.restore_id,
                    'backup_id': restore_info.backup_id,
                    'restore_type': restore_info.restore_type,
                    'target_path': str(restore_info.target_path),
                    'created_at': restore_info.created_at.isoformat(),
                    'completed_at': restore_info.completed_at.isoformat() if restore_info.completed_at else None,
                    'status': restore_info.status,
                    'progress_percentage': restore_info.progress_percentage,
                    'restored_files_count': restore_info.restored_files_count,
                    'total_files_count': restore_info.total_files_count,
                    'error_message': restore_info.error_message,
                    'metadata': restore_info.metadata
                })

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save restore history: {e}")

    def restore_full_backup(self, backup_info: BackupInfo, target_path: Optional[Path] = None,
                          progress_callback: Optional[Callable[[float], None]] = None) -> Optional[RestoreInfo]:
        """
        Restore full backup

        Args:
            backup_info: Backup information
            target_path: Target path
            progress_callback: Progress callback function

        Returns:
            Restore information
        """
        try:
            restore_id = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            target_path = target_path or Path(self.config.data_directory) / 'restored'

            restore_info = RestoreInfo(
                restore_id=restore_id,
                backup_id=backup_info.backup_id,
                restore_type='full',
                target_path=target_path,
                created_at=datetime.now(),
                status='in_progress'
            )

            self.restore_history.append(restore_info)
            self.logger.info(f"Starting full backup restore: {backup_info.backup_id} -> {target_path}")

            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)

            # Decrypt backup file (if needed)
            backup_file = backup_info.file_path
            if backup_info.encrypted and self.encryption_manager and self.encryption_manager.is_unlocked():
                temp_file = backup_file.with_suffix('.temp')
                try:
                    decrypted_file = self.encryption_manager.decrypt_file(backup_file, temp_file, 'backup')
                    if decrypted_file:
                        backup_file = decrypted_file
                    else:
                        raise Exception("Failed to decrypt backup file")
                except Exception as decrypt_error:
                    self.logger.error(f"Failed to decrypt backup file: {decrypt_error}")
                    raise

            # Extract backup file
            with tarfile.open(backup_file, 'r:gz' if backup_info.compressed else 'r') as tar:
                members = tar.getmembers()
                restore_info.total_files_count = len(members)

                for i, member in enumerate(members):
                    try:
                        tar.extract(member, target_path)
                        restore_info.restored_files_count += 1
                        restore_info.progress_percentage = (i + 1) / len(members) * 100

                        if progress_callback:
                            progress_callback(restore_info.progress_percentage)

                    except Exception as e:
                        self.logger.warning(f"Failed to restore file {member.name}: {e}")

            # Clean up temporary file
            if backup_info.encrypted and backup_file != backup_info.file_path:
                backup_file.unlink()

            restore_info.status = 'completed'
            restore_info.completed_at = datetime.now()
            self._save_restore_history()

            self.logger.info(f"Full backup restore successful: {restore_id}")
            return restore_info

        except Exception as e:
            restore_info.status = 'failed'
            restore_info.error_message = str(e)
            self._save_restore_history()
            self.logger.error(f"Failed to restore full backup: {e}")
            return restore_info

    def restore_selective_files(self, backup_info: BackupInfo, file_patterns: List[str],
                              target_path: Optional[Path] = None) -> Optional[RestoreInfo]:
        """
        Selective file restore

        Args:
            backup_info: Backup information
            file_patterns: File pattern list
            target_path: Target path

        Returns:
            Restore information
        """
        try:
            restore_id = f"selective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            target_path = target_path or Path(self.config.data_directory) / 'restored'

            restore_info = RestoreInfo(
                restore_id=restore_id,
                backup_id=backup_info.backup_id,
                restore_type='selective',
                target_path=target_path,
                created_at=datetime.now(),
                status='in_progress',
                metadata={'file_patterns': file_patterns}
            )

            self.restore_history.append(restore_info)
            self.logger.info(f"Starting selective restore: {backup_info.backup_id}")

            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)

            # Decrypt backup file (if needed)
            backup_file = backup_info.file_path
            if backup_info.encrypted and self.encryption_manager and self.encryption_manager.is_unlocked():
                temp_file = backup_file.with_suffix('.temp')
                try:
                    decrypted_file = self.encryption_manager.decrypt_file(backup_file, temp_file, 'backup')
                    if decrypted_file:
                        backup_file = decrypted_file
                except Exception as decrypt_error:
                    self.logger.error(f"Failed to decrypt backup file: {decrypt_error}")
                    raise

            # Selective extraction
            import fnmatch
            with tarfile.open(backup_file, 'r:gz' if backup_info.compressed else 'r') as tar:
                all_members = tar.getmembers()
                selected_members = []

                for member in all_members:
                    for pattern in file_patterns:
                        if fnmatch.fnmatch(member.name, pattern):
                            selected_members.append(member)
                            break

                restore_info.total_files_count = len(selected_members)

                for i, member in enumerate(selected_members):
                    try:
                        tar.extract(member, target_path)
                        restore_info.restored_files_count += 1
                        restore_info.progress_percentage = (i + 1) / len(selected_members) * 100

                    except Exception as e:
                        self.logger.warning(f"Failed to restore file {member.name}: {e}")

            # Clean up temporary file
            if backup_info.encrypted and backup_file != backup_info.file_path:
                backup_file.unlink()

            restore_info.status = 'completed'
            restore_info.completed_at = datetime.now()
            self._save_restore_history()

            self.logger.info(f"Selective restore successful: {restore_id}")
            return restore_info

        except Exception as e:
            restore_info.status = 'failed'
            restore_info.error_message = str(e)
            self._save_restore_history()
            self.logger.error(f"Selective restore failed: {e}")
            return restore_info

    def get_restore_progress(self, restore_id: str) -> Optional[RestoreInfo]:
        """Get restore progress"""
        return next((r for r in self.restore_history if r.restore_id == restore_id), None)

    def list_restores(self) -> List[RestoreInfo]:
        """List all restore records"""
        return self.restore_history.copy()


class MigrationManager:
    """Migration manager - Enterprise-level data migration functionality"""

    def __init__(self, config: Config, encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize migration manager

        Args:
            config: Configuration object
            encryption_manager: Encryption manager
        """
        self.config = config
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger('hibro.migration_manager')

        # Migration history and device information
        self.migration_history: List[MigrationInfo] = []
        self.sync_devices: Dict[str, SyncDevice] = {}
        self._load_migration_data()

    def _load_migration_data(self):
        """Load migration data"""
        try:
            # Load migration history
            migration_file = Path(self.config.data_directory) / 'backups' / 'migration_history.json'
            if migration_file.exists():
                with open(migration_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                for item in history_data:
                    migration_info = MigrationInfo(
                        migration_id=item['migration_id'],
                        source_device=item['source_device'],
                        target_device=item['target_device'],
                        migration_type=item['migration_type'],
                        created_at=datetime.fromisoformat(item['created_at']),
                        completed_at=datetime.fromisoformat(item['completed_at']) if item.get('completed_at') else None,
                        status=item.get('status', 'pending'),
                        progress_percentage=item.get('progress_percentage', 0.0),
                        transferred_size_bytes=item.get('transferred_size_bytes', 0),
                        total_size_bytes=item.get('total_size_bytes', 0),
                        error_message=item.get('error_message'),
                        metadata=item.get('metadata', {})
                    )
                    self.migration_history.append(migration_info)

            # Load sync devices
            devices_file = Path(self.config.data_directory) / 'backups' / 'sync_devices.json'
            if devices_file.exists():
                with open(devices_file, 'r', encoding='utf-8') as f:
                    devices_data = json.load(f)

                for device_id, device_data in devices_data.items():
                    sync_device = SyncDevice(
                        device_id=device_id,
                        device_name=device_data['device_name'],
                        device_type=device_data['device_type'],
                        last_sync_at=datetime.fromisoformat(device_data['last_sync_at']) if device_data.get('last_sync_at') else None,
                        sync_enabled=device_data.get('sync_enabled', True),
                        sync_path=Path(device_data['sync_path']) if device_data.get('sync_path') else None,
                        connection_status=device_data.get('connection_status', 'offline'),
                        metadata=device_data.get('metadata', {})
                    )
                    self.sync_devices[device_id] = sync_device

            self.logger.info(f"Loaded {len(self.migration_history)} migration records and {len(self.sync_devices)} sync devices")

        except Exception as e:
            self.logger.warning(f"Failed to load migration data: {e}")

    def _save_migration_data(self):
        """Save migration data"""
        try:
            backup_dir = Path(self.config.data_directory) / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Save migration history
            migration_file = backup_dir / 'migration_history.json'
            history_data = []
            for migration_info in self.migration_history:
                history_data.append({
                    'migration_id': migration_info.migration_id,
                    'source_device': migration_info.source_device,
                    'target_device': migration_info.target_device,
                    'migration_type': migration_info.migration_type,
                    'created_at': migration_info.created_at.isoformat(),
                    'completed_at': migration_info.completed_at.isoformat() if migration_info.completed_at else None,
                    'status': migration_info.status,
                    'progress_percentage': migration_info.progress_percentage,
                    'transferred_size_bytes': migration_info.transferred_size_bytes,
                    'total_size_bytes': migration_info.total_size_bytes,
                    'error_message': migration_info.error_message,
                    'metadata': migration_info.metadata
                })

            with open(migration_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            # Save sync devices
            devices_file = backup_dir / 'sync_devices.json'
            devices_data = {}
            for device_id, sync_device in self.sync_devices.items():
                devices_data[device_id] = {
                    'device_name': sync_device.device_name,
                    'device_type': sync_device.device_type,
                    'last_sync_at': sync_device.last_sync_at.isoformat() if sync_device.last_sync_at else None,
                    'sync_enabled': sync_device.sync_enabled,
                    'sync_path': str(sync_device.sync_path) if sync_device.sync_path else None,
                    'connection_status': sync_device.connection_status,
                    'metadata': sync_device.metadata
                }

            with open(devices_file, 'w', encoding='utf-8') as f:
                json.dump(devices_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save migration data: {e}")

    def register_sync_device(self, device_name: str, device_type: str, sync_path: Optional[Path] = None) -> str:
        """
        Register sync device

        Args:
            device_name: Device name
            device_type: Device type
            sync_path: Sync path

        Returns:
            Device ID
        """
        try:
            device_id = f"device_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.sync_devices)}"

            sync_device = SyncDevice(
                device_id=device_id,
                device_name=device_name,
                device_type=device_type,
                sync_path=sync_path,
                connection_status='online'
            )

            self.sync_devices[device_id] = sync_device
            self._save_migration_data()

            self.logger.info(f"Sync device registered successfully: {device_name} ({device_id})")
            return device_id

        except Exception as e:
            self.logger.error(f"Failed to register sync device: {e}")
            return ""

    def start_cross_device_sync(self, source_device_id: str, target_device_id: str,
                              sync_type: str = "incremental") -> Optional[MigrationInfo]:
        """
        Start cross-device sync

        Args:
            source_device_id: Source device ID
            target_device_id: Target device ID
            sync_type: Sync type

        Returns:
            Migration information
        """
        try:
            migration_id = f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            source_device = self.sync_devices.get(source_device_id)
            target_device = self.sync_devices.get(target_device_id)

            if not source_device or not target_device:
                raise Exception("Source or target device does not exist")

            migration_info = MigrationInfo(
                migration_id=migration_id,
                source_device=source_device_id,
                target_device=target_device_id,
                migration_type=sync_type,
                created_at=datetime.now(),
                status='in_progress'
            )

            self.migration_history.append(migration_info)
            self.logger.info(f"Starting cross-device sync: {source_device_id} -> {target_device_id}")

            # Calculate data size to sync
            source_path = source_device.sync_path or Path(self.config.data_directory)
            total_size = sum(f.stat().st_size for f in source_path.rglob('*') if f.is_file())
            migration_info.total_size_bytes = total_size

            # Execute sync (simplified implementation, should use rsync or similar in production)
            if target_device.sync_path and target_device.sync_path.exists():
                self._sync_directories(source_path, target_device.sync_path, migration_info)

            migration_info.status = 'completed'
            migration_info.completed_at = datetime.now()

            # Update device sync time
            source_device.last_sync_at = datetime.now()
            target_device.last_sync_at = datetime.now()

            self._save_migration_data()

            self.logger.info(f"Cross-device sync completed: {migration_id}")
            return migration_info

        except Exception as e:
            migration_info.status = 'failed'
            migration_info.error_message = str(e)
            self._save_migration_data()
            self.logger.error(f"Cross-device sync failed: {e}")
            return migration_info

    def _sync_directories(self, source_dir: Path, target_dir: Path, migration_info: MigrationInfo):
        """Sync directories"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            for source_file in source_dir.rglob('*'):
                if source_file.is_file():
                    rel_path = source_file.relative_to(source_dir)
                    target_file = target_dir / rel_path

                    # Check if sync is needed
                    if not target_file.exists() or source_file.stat().st_mtime > target_file.stat().st_mtime:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_file)
                        migration_info.transferred_size_bytes += source_file.stat().st_size

                        # Update progress
                        if migration_info.total_size_bytes > 0:
                            migration_info.progress_percentage = (migration_info.transferred_size_bytes / migration_info.total_size_bytes) * 100

        except Exception as e:
            self.logger.error(f"Directory sync failed: {e}")
            raise

    def get_sync_devices(self) -> Dict[str, SyncDevice]:
        """Get sync device list"""
        return self.sync_devices.copy()

    def get_migration_progress(self, migration_id: str) -> Optional[MigrationInfo]:
        """Get migration progress"""
        return next((m for m in self.migration_history if m.migration_id == migration_id), None)

    def list_migrations(self) -> List[MigrationInfo]:
        """List all migration records"""
        return self.migration_history.copy()