#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database migration manager
Manages database schema version upgrades and rollbacks
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from ..database import DatabaseManager


class BaseMigration(ABC):
    """Migration base class"""

    VERSION: str = "0.0.0"
    DESCRIPTION: str = ""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(f'hibro.migration.{self.__class__.__name__}')

    @abstractmethod
    def up(self) -> bool:
        """Execute migration"""
        pass

    @abstractmethod
    def down(self) -> bool:
        """Rollback migration"""
        pass


class MigrationManager:
    """Database migration manager"""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize migration manager

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.db_path = str(db_manager.db_path)
        self.logger = logging.getLogger('hibro.migration_manager')

        # Registered migration list
        self.migrations: List[BaseMigration] = []

        # Initialize migration history table
        self._init_migration_table()

    def _init_migration_table(self):
        """Initialize migration history table"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS migration_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT UNIQUE NOT NULL,
                        description TEXT NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        rollback_sql TEXT
                    )
                """)
                conn.commit()

            self.logger.info("Migration history table initialization completed")

        except Exception as e:
            self.logger.error(f"Migration history table initialization failed: {e}")
            raise

    def register_migration(self, migration_class: type):
        """
        Register migration

        Args:
            migration_class: Migration class
        """
        migration = migration_class(self.db_path)
        self.migrations.append(migration)
        self.logger.info(f"Registered migration: {migration.VERSION} - {migration.DESCRIPTION}")

    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT version FROM migration_history ORDER BY applied_at")
                return [row['version'] for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get migration history: {e}")
            return []

    def get_pending_migrations(self) -> List[BaseMigration]:
        """Get list of pending migrations"""
        applied_versions = set(self.get_applied_migrations())
        pending = []

        for migration in sorted(self.migrations, key=lambda m: m.VERSION):
            if migration.VERSION not in applied_versions:
                pending.append(migration)

        return pending

    def migrate_up(self, target_version: Optional[str] = None) -> bool:
        """
        Execute migration upgrade

        Args:
            target_version: Target version, upgrade to latest if None

        Returns:
            Whether successful
        """
        pending_migrations = self.get_pending_migrations()

        if not pending_migrations:
            self.logger.info("No pending migrations to apply")
            return True

        # If target version specified, only execute up to that version
        if target_version:
            pending_migrations = [m for m in pending_migrations if m.VERSION <= target_version]

        success_count = 0
        for migration in pending_migrations:
            self.logger.info(f"Starting migration: {migration.VERSION} - {migration.DESCRIPTION}")

            try:
                # Execute migration
                if migration.up():
                    # Record migration history
                    self._record_migration(migration)
                    success_count += 1
                    self.logger.info(f"Migration applied successfully: {migration.VERSION}")
                else:
                    self.logger.error(f"Migration application failed: {migration.VERSION}")
                    break

            except Exception as e:
                self.logger.error(f"Migration application exception: {migration.VERSION} - {e}")
                break

        self.logger.info(f"Migration completed, successfully applied {success_count}/{len(pending_migrations)} migrations")
        return success_count == len(pending_migrations)

    def migrate_down(self, target_version: str) -> bool:
        """
        Execute migration rollback

        Args:
            target_version: Target version

        Returns:
            Whether successful
        """
        applied_versions = self.get_applied_migrations()

        # Find migrations to rollback
        migrations_to_rollback = []
        for version in reversed(applied_versions):
            if version > target_version:
                # Find corresponding migration object
                migration = next((m for m in self.migrations if m.VERSION == version), None)
                if migration:
                    migrations_to_rollback.append(migration)

        if not migrations_to_rollback:
            self.logger.info(f"No migrations to rollback to version {target_version}")
            return True

        success_count = 0
        for migration in migrations_to_rollback:
            self.logger.info(f"Starting rollback: {migration.VERSION} - {migration.DESCRIPTION}")

            try:
                # Execute rollback
                if migration.down():
                    # Remove from history
                    self._remove_migration_record(migration.VERSION)
                    success_count += 1
                    self.logger.info(f"Migration rollback successful: {migration.VERSION}")
                else:
                    self.logger.error(f"Migration rollback failed: {migration.VERSION}")
                    break

            except Exception as e:
                self.logger.error(f"Migration rollback exception: {migration.VERSION} - {e}")
                break

        self.logger.info(f"Rollback completed, successfully rolled back {success_count}/{len(migrations_to_rollback)} migrations")
        return success_count == len(migrations_to_rollback)

    def _record_migration(self, migration: BaseMigration):
        """Record migration history"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO migration_history (version, description, applied_at)
                    VALUES (?, ?, ?)
                """, (migration.VERSION, migration.DESCRIPTION, datetime.now()))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to record migration history: {e}")
            raise

    def _remove_migration_record(self, version: str):
        """Remove migration history record"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("DELETE FROM migration_history WHERE version = ?", (version,))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to remove migration history: {e}")
            raise

    def get_current_version(self) -> Optional[str]:
        """Get current database version"""
        applied_versions = self.get_applied_migrations()
        return applied_versions[-1] if applied_versions else None

    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status information"""
        applied_versions = self.get_applied_migrations()
        pending_migrations = self.get_pending_migrations()

        return {
            'current_version': self.get_current_version(),
            'applied_count': len(applied_versions),
            'pending_count': len(pending_migrations),
            'applied_versions': applied_versions,
            'pending_versions': [m.VERSION for m in pending_migrations],
            'total_migrations': len(self.migrations)
        }

    def validate_database_integrity(self) -> bool:
        """Validate database integrity"""
        try:
            with self.db_manager.get_connection() as conn:
                # Check foreign key constraints
                conn.execute("PRAGMA foreign_key_check")

                # Check database integrity
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result and result[0] == 'ok':
                    self.logger.info("Database integrity check passed")
                    return True
                else:
                    self.logger.error(f"Database integrity check failed: {result}")
                    return False

        except Exception as e:
            self.logger.error(f"Database integrity check exception: {e}")
            return False

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Backup database

        Args:
            backup_path: Backup file path, auto-generated if None

        Returns:
            Backup file path
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"

        try:
            # Use SQLite backup API
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)

            self.logger.info(f"Database backup completed: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise

    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup

        Args:
            backup_path: Backup file path

        Returns:
            Whether successful
        """
        try:
            if not Path(backup_path).exists():
                self.logger.error(f"Backup file does not exist: {backup_path}")
                return False

            # Backup current database
            current_backup = self.backup_database()

            try:
                # Restore database
                with sqlite3.connect(backup_path) as source:
                    with sqlite3.connect(self.db_path) as target:
                        source.backup(target)

                self.logger.info(f"Database restore completed: {backup_path}")
                return True

            except Exception as e:
                # Restore failed, rollback to current backup
                self.logger.error(f"Database restore failed, rolling back: {e}")
                with sqlite3.connect(current_backup) as source:
                    with sqlite3.connect(self.db_path) as target:
                        source.backup(target)
                return False

        except Exception as e:
            self.logger.error(f"Database restore exception: {e}")
            return False