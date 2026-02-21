#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore Manager
Provides functionality to restore data from backups
"""

import os
import shutil
import tarfile
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from ..utils.config import Config
from ..security.encryption import EncryptionManager
from .backup_manager import BackupManager, BackupInfo


@dataclass
class RestoreResult:
    """Restore result"""
    success: bool
    backup_id: str
    restored_items: List[str]
    errors: List[str]
    warnings: List[str]
    start_time: datetime
    end_time: datetime
    duration_seconds: float


class RestoreManager:
    """Restore manager"""

    def __init__(self, config: Config, backup_manager: BackupManager,
                encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize restore manager

        Args:
            config: Configuration object
            backup_manager: Backup manager
            encryption_manager: Encryption manager
        """
        self.config = config
        self.backup_manager = backup_manager
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger('hibro.restore_manager')

        # Restore configuration
        self.restore_config = {
            'verify_before_restore': True,
            'create_pre_restore_backup': True,
            'overwrite_existing': False,
            'restore_database': True,
            'restore_filesystem': True,
            'restore_configuration': True,
            'temp_directory': Path(config.data_directory) / 'temp_restore'
        }

        # Create temporary directory
        self.restore_config['temp_directory'].mkdir(parents=True, exist_ok=True)

    def restore_from_backup(self, backup_id: str,
                           restore_options: Optional[Dict[str, bool]] = None) -> RestoreResult:
        """
        Restore data from backup

        Args:
            backup_id: Backup ID
            restore_options: Restore options

        Returns:
            Restore result
        """
        start_time = datetime.now()
        restored_items = []
        errors = []
        warnings = []

        try:
            # Get backup info
            backup_info = self.backup_manager.get_backup_info(backup_id)
            if not backup_info:
                errors.append(f"Backup does not exist: {backup_id}")
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_items=restored_items,
                    errors=errors,
                    warnings=warnings,
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=0.0
                )

            self.logger.info(f"Starting restore from backup: {backup_id}")

            # Apply restore options
            options = self._get_restore_options(restore_options)

            # Verify backup
            if options.get('verify_before_restore', True):
                if not self.backup_manager.verify_backup(backup_id):
                    errors.append(f"Backup verification failed: {backup_id}")
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        restored_items=restored_items,
                        errors=errors,
                        warnings=warnings,
                        start_time=start_time,
                        end_time=datetime.now(),
                        duration_seconds=0.0
                    )

            # Create pre-restore backup
            if options.get('create_pre_restore_backup', True):
                pre_backup = self.backup_manager.create_full_backup("Auto backup before restore")
                if pre_backup:
                    warnings.append(f"Created pre-restore backup: {pre_backup.backup_id}")

            # Extract backup file
            temp_dir = self._extract_backup(backup_info)
            if not temp_dir:
                errors.append("Failed to extract backup file")
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_items=restored_items,
                    errors=errors,
                    warnings=warnings,
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=0.0
                )

            try:
                # Read backup metadata
                metadata = self._read_backup_metadata(temp_dir)
                if not metadata:
                    errors.append("Failed to read backup metadata")
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        restored_items=restored_items,
                        errors=errors,
                        warnings=warnings,
                        start_time=start_time,
                        end_time=datetime.now(),
                        duration_seconds=0.0
                    )

                # Restore database
                if options.get('restore_database', True):
                    if self._restore_database(temp_dir, metadata):
                        restored_items.append("Database")
                    else:
                        errors.append("Database restore failed")

                # Restore filesystem
                if options.get('restore_filesystem', True):
                    if self._restore_filesystem(temp_dir, metadata):
                        restored_items.append("Filesystem")
                    else:
                        warnings.append("Partial filesystem restore failed")

                # Restore configuration
                if options.get('restore_configuration', True):
                    if self._restore_configuration(temp_dir, metadata):
                        restored_items.append("Configuration")
                    else:
                        warnings.append("Configuration restore failed")

            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = len(restored_items) > 0 and len(errors) == 0

            self.logger.info(f"Restore completed: {backup_id}, success={success}")

            return RestoreResult(
                success=success,
                backup_id=backup_id,
                restored_items=restored_items,
                errors=errors,
                warnings=warnings,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )

        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            errors.append(str(e))

            return RestoreResult(
                success=False,
                backup_id=backup_id,
                restored_items=restored_items,
                errors=errors,
                warnings=warnings,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    def _get_restore_options(self, options: Optional[Dict[str, bool]]) -> Dict[str, bool]:
        """Get restore options"""
        default_options = {
            'verify_before_restore': self.restore_config['verify_before_restore'],
            'create_pre_restore_backup': self.restore_config['create_pre_restore_backup'],
            'overwrite_existing': self.restore_config['overwrite_existing'],
            'restore_database': self.restore_config['restore_database'],
            'restore_filesystem': self.restore_config['restore_filesystem'],
            'restore_configuration': self.restore_config['restore_configuration']
        }

        if options:
            default_options.update(options)

        return default_options

    def _extract_backup(self, backup_info: BackupInfo) -> Optional[Path]:
        """Extract backup file"""
        try:
            backup_path = backup_info.file_path

            # If encrypted file, decrypt first
            if backup_info.encrypted and self.encryption_manager:
                decrypted_path = self._decrypt_backup(backup_path)
                if decrypted_path:
                    backup_path = decrypted_path
                else:
                    return None

            # Create temporary directory
            temp_dir = self.restore_config['temp_directory'] / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract
            mode = 'r:gz' if backup_info.compressed else 'r'
            with tarfile.open(backup_path, mode) as tar:
                tar.extractall(temp_dir)

            self.logger.info(f"Backup extraction completed: {temp_dir}")
            return temp_dir

        except Exception as e:
            self.logger.error(f"Failed to extract backup: {e}")
            return None

    def _decrypt_backup(self, encrypted_path: Path) -> Optional[Path]:
        """Decrypt backup file"""
        try:
            if not self.encryption_manager or not self.encryption_manager.is_unlocked():
                self.logger.error("Encryption manager not unlocked, cannot decrypt backup")
                return None

            decrypted_path = self.restore_config['temp_directory'] / encrypted_path.name.replace('.enc', '')

            result = self.encryption_manager.decrypt_file(encrypted_path, decrypted_path, 'backup')
            if result:
                return result

            return None

        except Exception as e:
            self.logger.error(f"Failed to decrypt backup: {e}")
            return None

    def _read_backup_metadata(self, temp_dir: Path) -> Optional[Dict[str, Any]]:
        """Read backup metadata"""
        try:
            metadata_file = temp_dir / 'backup_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}

        except Exception as e:
            self.logger.error(f"Failed to read backup metadata: {e}")
            return None

    def _restore_database(self, temp_dir: Path, metadata: Dict[str, Any]) -> bool:
        """Restore database"""
        try:
            db_backup_name = metadata.get('database_backup')
            if not db_backup_name:
                self.logger.warning("Backup database info does not exist")
                return False

            db_backup_path = temp_dir / db_backup_name
            if not db_backup_path.exists():
                self.logger.error(f"Backup database file does not exist: {db_backup_path}")
                return False

            target_db_path = Path(self.config.data_directory) / 'hibro.db'

            # Backup current database
            if target_db_path.exists() and not self.restore_config['overwrite_existing']:
                backup_current = target_db_path.with_suffix('.db.bak')
                shutil.copy2(target_db_path, backup_current)
                self.logger.info(f"Current database backed up: {backup_current}")

            # Restore database
            shutil.copy2(db_backup_path, target_db_path)

            self.logger.info(f"DatabaseRestore completed: {target_db_path}")
            return True

        except Exception as e:
            self.logger.error(f"Database restore failed: {e}")
            return False

    def _restore_filesystem(self, temp_dir: Path, metadata: Dict[str, Any]) -> bool:
        """Restore filesystem"""
        try:
            fs_backup_name = metadata.get('filesystem_backup')
            if not fs_backup_name:
                self.logger.warning("Backup filesystem info does not exist")
                return False

            fs_backup_path = temp_dir / fs_backup_name
            if not fs_backup_path.exists():
                self.logger.error(f"Backup filesystem directory does not exist: {fs_backup_path}")
                return False

            data_dir = Path(self.config.data_directory)
            success = True

            # Restore each directory
            for item in fs_backup_path.iterdir():
                try:
                    target_path = data_dir / item.name

                    if target_path.exists():
                        if self.restore_config['overwrite_existing']:
                            if target_path.is_dir():
                                shutil.rmtree(target_path)
                            else:
                                target_path.unlink()
                        else:
                            self.logger.debug(f"Skip existing: {target_path}")
                            continue

                    if item.is_dir():
                        shutil.copytree(item, target_path)
                    else:
                        shutil.copy2(item, target_path)

                    self.logger.debug(f"Restored: {item.name}")

                except Exception as e:
                    self.logger.warning(f"Restore.*failed: {e}")
                    success = False

            self.logger.info(f"FilesystemRestore completed")
            return success

        except Exception as e:
            self.logger.error(f"FilesystemRestore failed: {e}")
            return False

    def _restore_configuration(self, temp_dir: Path, metadata: Dict[str, Any]) -> bool:
        """Restore configuration"""
        try:
            config_backup_name = metadata.get('config_backup')
            if not config_backup_name:
                self.logger.warning("Backup configuration info does not exist")
                return False

            config_backup_path = temp_dir / config_backup_name
            if not config_backup_path.exists():
                self.logger.error(f"Backup configuration directory does not exist: {config_backup_path}")
                return False

            data_dir = Path(self.config.data_directory)

            # Restore configuration files
            for config_file in config_backup_path.iterdir():
                try:
                    target_path = data_dir / config_file.name

                    if target_path.exists() and not self.restore_config['overwrite_existing']:
                        # Backup current configuration
                        backup_path = target_path.with_suffix(target_path.suffix + '.bak')
                        shutil.copy2(target_path, backup_path)

                    shutil.copy2(config_file, target_path)
                    self.logger.debug(f"Restore configuration: {config_file.name}")

                except Exception as e:
                    self.logger.warning(f"Failed to restore configuration {config_file.name}: {e}")

            self.logger.info("ConfigurationRestore completed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration restore failed: {e}")
            return False

    def restore_incremental_backup(self, backup_id: str) -> RestoreResult:
        """
        Restore incremental backup

        Args:
            backup_id: Incremental backup ID

        Returns:
            Restore result
        """
        try:
            backup_info = self.backup_manager.get_backup_info(backup_id)
            if not backup_info:
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_items=[],
                    errors=[f"Backup does not exist: {backup_id}"],
                    warnings=[],
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0
                )

            if backup_info.backup_type != 'incremental':
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_items=[],
                    errors=["Not an incremental backup"],
                    warnings=[],
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0
                )

            # Restore base backup first
            base_backup_id = backup_info.metadata.get('base_backup_id')
            if base_backup_id:
                base_result = self.restore_from_backup(base_backup_id)
                if not base_result.success:
                    return base_result

            # Then apply incremental backup
            return self._apply_incremental_backup(backup_info)

        except Exception as e:
            self.logger.error(f"Restore.*failed: {e}")
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                restored_items=[],
                errors=[str(e)],
                warnings=[],
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0
            )

    def _apply_incremental_backup(self, backup_info: BackupInfo) -> RestoreResult:
        """Apply incremental backup"""
        start_time = datetime.now()

        try:
            # Extract backup
            temp_dir = self._extract_backup(backup_info)
            if not temp_dir:
                return RestoreResult(
                    success=False,
                    backup_id=backup_info.backup_id,
                    restored_items=[],
                    errors=["Failed to extract incremental backup"],
                    warnings=[],
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=0.0
                )

            try:
                # Apply changes
                changes_dir = temp_dir / 'changes'
                data_dir = Path(self.config.data_directory)
                restored_items = []

                if changes_dir.exists():
                    for item in changes_dir.rglob('*'):
                        if item.is_file():
                            try:
                                rel_path = item.relative_to(changes_dir)
                                target_path = data_dir / rel_path
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, target_path)
                                restored_items.append(str(rel_path))
                            except Exception as e:
                                self.logger.warning(f"Failed to apply changes {item}: {e}")

                return RestoreResult(
                    success=True,
                    backup_id=backup_info.backup_id,
                    restored_items=restored_items,
                    errors=[],
                    warnings=[],
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )

            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Failed to apply incremental backup: {e}")
            return RestoreResult(
                success=False,
                backup_id=backup_info.backup_id,
                restored_items=[],
                errors=[str(e)],
                warnings=[],
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=0.0
            )

    def list_restorable_backups(self) -> List[Dict[str, Any]]:
        """List restorable backups"""
        try:
            backups = self.backup_manager.list_backups()
            restorable = []

            for backup in backups:
                # Check if backup file exists
                if backup.file_path.exists():
                    restorable.append({
                        'backup_id': backup.backup_id,
                        'type': backup.backup_type,
                        'created_at': backup.created_at.isoformat(),
                        'size_mb': backup.size_bytes / (1024 * 1024),
                        'encrypted': backup.encrypted,
                        'description': backup.metadata.get('description', '')
                    })

            return restorable

        except Exception as e:
            self.logger.error(f"Failed to list restorable backups: {e}")
            return []

    def update_restore_config(self, **kwargs) -> bool:
        """
        Update restore configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.restore_config:
                    self.restore_config[key] = value
                    self.logger.info(f"Restore configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update restore configuration: {e}")
            return False