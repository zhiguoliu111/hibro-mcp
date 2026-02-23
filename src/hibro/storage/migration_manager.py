#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration manager

Manages execution and version control of database migration scripts
"""

import sqlite3
import logging
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class MigrationManager:
    """Migration manager"""

    def __init__(self, db_manager):
        """
        Initialize migration manager

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.migration_manager')
        # Point to migrations subdirectory
        self.migrations_dir = Path(__file__).parent / 'migrations'

    def _ensure_migration_table(self, conn: sqlite3.Connection):
        """
        Ensure migration record table exists

        Args:
            conn: Database connection
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
        """)
        conn.commit()

    def _get_applied_migrations(self, conn: sqlite3.Connection) -> List[str]:
        """
        Get list of applied migration versions

        Args:
            conn: Database connection

        Returns:
            List of applied versions
        """
        cursor = conn.execute("""
            SELECT version FROM schema_migrations
            WHERE success = 1
            ORDER BY version
        """)
        return [row[0] for row in cursor.fetchall()]

    def _get_available_migrations(self) -> List[str]:
        """
        Get list of available migration scripts

        Returns:
            List of available versions
        """
        migrations = []
        for file in self.migrations_dir.glob('*.py'):
            # Skip __init__.py and other non-migration files
            if file.name.startswith('_'):
                continue

            # Only recognize migration scripts starting with digits (format: 001_description.py)
            if not file.stem[0].isdigit():
                continue

            # Extract version number
            version = file.stem.split('_')[0]
            migrations.append(version)

        return sorted(migrations)

    def _load_migration_module(self, version: str):
        """
        Load migration module

        Args:
            version: Migration version

        Returns:
            Migration module
        """
        # Find corresponding migration file
        for file in self.migrations_dir.glob(f'{version}_*.py'):
            module_name = f'hibro.storage.migrations.{file.stem}'
            return importlib.import_module(module_name)

        raise FileNotFoundError(f"Migration script not found: {version}")

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get migration status

        Returns:
            Migration status information
        """
        try:
            with self.db_manager.get_connection() as conn:
                self._ensure_migration_table(conn)

                applied = self._get_applied_migrations(conn)
                available = self._get_available_migrations()

                pending = [v for v in available if v not in applied]

                return {
                    'applied': applied,
                    'available': available,
                    'pending': pending,
                    'current_version': applied[-1] if applied else None
                }

        except Exception as e:
            self.logger.error(f"Failed to get migration status: {e}")
            return {
                'applied': [],
                'available': [],
                'pending': [],
                'current_version': None,
                'error': str(e)
            }

    def migrate(self, target_version: Optional[str] = None) -> bool:
        """
        Execute migration to specified version

        Args:
            target_version: Target version, None means migrate to latest

        Returns:
            Whether successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                self._ensure_migration_table(conn)

                applied = self._get_applied_migrations(conn)
                available = self._get_available_migrations()

                # Determine target version
                if target_version is None:
                    target_version = available[-1] if available else None

                if target_version is None:
                    self.logger.info("No available migration scripts")
                    return True

                # Find migrations to execute
                pending = [v for v in available if v not in applied and v <= target_version]

                if not pending:
                    self.logger.info(f"Database is already at version: {target_version}")
                    return True

                # Execute migrations
                for version in pending:
                    self.logger.info(f"Executing migration: {version}")

                    try:
                        # Load migration module
                        migration = self._load_migration_module(version)

                        # Execute upgrade
                        success = migration.upgrade(conn)

                        if success:
                            # Record successful migration
                            conn.execute("""
                                INSERT INTO schema_migrations (version, success)
                                VALUES (?, 1)
                            """, (version,))
                            conn.commit()

                            self.logger.info(f"Migration {version} executed successfully")
                        else:
                            self.logger.error(f"Migration {version} failed")
                            return False

                    except Exception as e:
                        self.logger.error(f"Migration {version} exception: {e}")
                        conn.rollback()
                        return False

                self.logger.info(f"All migrations completed, current version: {target_version}")
                return True

        except Exception as e:
            self.logger.error(f"Migration execution failed: {e}")
            return False

    def rollback(self, target_version: Optional[str] = None) -> bool:
        """
        Rollback to specified version

        Args:
            target_version: Target version, None means rollback one version

        Returns:
            Whether successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                self._ensure_migration_table(conn)

                applied = self._get_applied_migrations(conn)

                if not applied:
                    self.logger.info("No applied migrations to rollback")
                    return True

                # Determine target version
                if target_version is None:
                    # Rollback last version
                    versions_to_rollback = [applied[-1]]
                else:
                    # Rollback to specified version
                    versions_to_rollback = [v for v in reversed(applied) if v > target_version]

                # Execute rollback
                for version in versions_to_rollback:
                    self.logger.info(f"Rolling back migration: {version}")

                    try:
                        # Load migration module
                        migration = self._load_migration_module(version)

                        # Execute downgrade
                        success = migration.downgrade(conn)

                        if success:
                            # Delete migration record
                            conn.execute("""
                                DELETE FROM schema_migrations
                                WHERE version = ?
                            """, (version,))
                            conn.commit()

                            self.logger.info(f"Migration {version} rolled back successfully")
                        else:
                            self.logger.error(f"Migration {version} rollback failed")
                            return False

                    except Exception as e:
                        self.logger.error(f"Migration {version} rollback exception: {e}")
                        conn.rollback()
                        return False

                current_version = target_version if target_version else (applied[-2] if len(applied) > 1 else None)
                self.logger.info(f"Rollback completed, current version: {current_version}")
                return True

        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            return False
