#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration Manager
Provides data migration and version upgrade functionality
"""

import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from ..utils.config import Config
from ..storage.database import DatabaseManager
from ..storage.models import Memory


@dataclass
class MigrationInfo:
    """Migration information"""
    migration_id: str
    version_from: str
    version_to: str
    applied_at: datetime
    description: str
    success: bool


@dataclass
class MigrationResult:
    """Migration result"""
    success: bool
    version_from: str
    version_to: str
    applied_migrations: List[str]
    errors: List[str]
    warnings: List[str]
    backup_id: Optional[str]


class MigrationManager:
    """Migration manager"""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        """
        Initialize migration manager

        Args:
            config: Configuration object
            db_manager: Database manager
        """
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.migration_manager')

        # Migration configuration
        self.migration_config = {
            'auto_backup_before_migration': True,
            'validate_after_migration': True,
            'rollback_on_failure': True,
            'migrations_directory': Path(config.data_directory) / 'migrations'
        }

        # Current version
        self.current_version = "1.0.0"

        # Migration history
        self.migration_history: List[MigrationInfo] = []

        # Migration scripts
        self.migrations = self._initialize_migrations()

    def _initialize_migrations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize migration scripts"""
        return {
            '1.0.0_to_1.1.0': {
                'version_from': '1.0.0',
                'version_to': '1.1.0',
                'description': 'Add memory tags and relations feature',
                'up': self._migration_1_0_to_1_1_up,
                'down': self._migration_1_0_to_1_1_down
            },
            '1.1.0_to_1.2.0': {
                'version_from': '1.1.0',
                'version_to': '1.2.0',
                'description': 'Add memory version control and history',
                'up': self._migration_1_1_to_1_2_up,
                'down': self._migration_1_1_to_1_2_down
            },
            '1.2.0_to_1.3.0': {
                'version_from': '1.2.0',
                'version_to': '1.3.0',
                'description': 'Optimize indexes and add full-text search support',
                'up': self._migration_1_2_to_1_3_up,
                'down': self._migration_1_2_to_1_3_down
            }
        }

    def get_current_version(self) -> str:
        """Get current database version"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Check if version table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='schema_version'
                """)

                if cursor.fetchone():
                    cursor.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
                    result = cursor.fetchone()
                    if result:
                        return result[0]

                return "1.0.0"  # Default initial version

        except Exception as e:
            self.logger.error(f"Failed to get current version: {e}")
            return "1.0.0"

    def get_available_migrations(self) -> List[Dict[str, Any]]:
        """Get available migrations"""
        current_version = self.get_current_version()
        available = []

        for migration_id, migration_data in self.migrations.items():
            if migration_data['version_from'] == current_version:
                available.append({
                    'migration_id': migration_id,
                    'version_from': migration_data['version_from'],
                    'version_to': migration_data['version_to'],
                    'description': migration_data['description']
                })

        return available

    def migrate_to_version(self, target_version: str,
                          backup_manager: Optional[Any] = None) -> MigrationResult:
        """
        Migrate to specified version

        Args:
            target_version: Target version
            backup_manager: Backup manager

        Returns:
            Migration result
        """
        applied_migrations = []
        errors = []
        warnings = []
        backup_id = None

        try:
            current_version = self.get_current_version()
            self.logger.info(f"Starting migration: {current_version} -> {target_version}")

            if current_version == target_version:
                return MigrationResult(
                    success=True,
                    version_from=current_version,
                    version_to=target_version,
                    applied_migrations=[],
                    errors=[],
                    warnings=["Already at target version, no migration needed"],
                    backup_id=None
                )

            # Create pre-migration backup
            if self.migration_config['auto_backup_before_migration'] and backup_manager:
                backup = backup_manager.create_full_backup("Auto backup before migration")
                if backup:
                    backup_id = backup.backup_id
                    self.logger.info(f"Created pre-migration backup: {backup_id}")

            # Determine migration path
            migration_path = self._determine_migration_path(current_version, target_version)

            if not migration_path:
                errors.append(f"Cannot find migration path from {current_version} to {target_version}")
                return MigrationResult(
                    success=False,
                    version_from=current_version,
                    version_to=target_version,
                    applied_migrations=applied_migrations,
                    errors=errors,
                    warnings=warnings,
                    backup_id=backup_id
                )

            # Execute migration
            for migration_id in migration_path:
                try:
                    migration = self.migrations.get(migration_id)
                    if not migration:
                        errors.append(f"Migration script does not exist: {migration_id}")
                        break

                    # Execute migration
                    success = migration['up']()

                    if success:
                        # Record migration
                        self._record_migration(migration_id, migration)
                        applied_migrations.append(migration_id)
                        self.logger.info(f"Migration completed: {migration_id}")
                    else:
                        errors.append(f"Migration execution failed: {migration_id}")
                        if self.migration_config['rollback_on_failure']:
                            self._rollback_migrations(applied_migrations)
                        break

                except Exception as e:
                    errors.append(f"Migration exception {migration_id}: {e}")
                    if self.migration_config['rollback_on_failure']:
                        self._rollback_migrations(applied_migrations)
                    break

            # Validate migration result
            if not errors and self.migration_config['validate_after_migration']:
                validation_errors = self._validate_migration()
                if validation_errors:
                    warnings.extend(validation_errors)

            success = len(applied_migrations) > 0 and len(errors) == 0

            return MigrationResult(
                success=success,
                version_from=current_version,
                version_to=self.get_current_version(),
                applied_migrations=applied_migrations,
                errors=errors,
                warnings=warnings,
                backup_id=backup_id
            )

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            errors.append(str(e))

            return MigrationResult(
                success=False,
                version_from=self.get_current_version(),
                version_to=target_version,
                applied_migrations=applied_migrations,
                errors=errors,
                warnings=warnings,
                backup_id=backup_id
            )

    def _determine_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Determine migration path"""
        path = []
        current = from_version

        while current != to_version:
            found = False
            for migration_id, migration in self.migrations.items():
                if migration['version_from'] == current:
                    path.append(migration_id)
                    current = migration['version_to']
                    found = True
                    break

            if not found:
                return []  # Cannot find migration path

        return path

    def _record_migration(self, migration_id: str, migration: Dict[str, Any]):
        """Record migration"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Create version table (if not exists)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        migration_id TEXT NOT NULL,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Record migration
                cursor.execute("""
                    INSERT INTO schema_version (version, migration_id, description)
                    VALUES (?, ?, ?)
                """, (migration['version_to'], migration_id, migration['description']))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to record migration: {e}")

    def _rollback_migrations(self, applied_migrations: List[str]) -> bool:
        """Rollback migrations"""
        try:
            # Rollback in reverse order
            for migration_id in reversed(applied_migrations):
                migration = self.migrations.get(migration_id)
                if migration and 'down' in migration:
                    migration['down']()
                    self.logger.info(f"Rolled back migration: {migration_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to rollback migration: {e}")
            return False

    def _validate_migration(self) -> List[str]:
        """Validate migration result"""
        errors = []

        try:
            # Check database integrity
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                if result[0] != 'ok':
                    errors.append(f"Database integrity check failed: {result[0]}")

                # Check table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = ['memories', 'projects', 'preferences', 'schema_version']
                for table in required_tables:
                    if table not in tables:
                        errors.append(f"Missing required table: {table}")

        except Exception as e:
            errors.append(f"Validation exception: {e}")

        return errors

    # Migration script implementations

    def _migration_1_0_to_1_1_up(self) -> bool:
        """1.0 -> 1.1 upgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Add tags table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id TEXT NOT NULL,
                        tag TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    )
                """)

                # Add memory relations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_memory_id TEXT NOT NULL,
                        target_memory_id TEXT NOT NULL,
                        relation_type TEXT DEFAULT 'related',
                        strength REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                        FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    )
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_memory_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_memory_id)")

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Migration 1.0->1.1 failed: {e}")
            return False

    def _migration_1_0_to_1_1_down(self) -> bool:
        """1.1 -> 1.0 downgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DROP TABLE IF EXISTS memory_tags")
                cursor.execute("DROP TABLE IF EXISTS memory_relations")

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Rollback 1.1->1.0 failed: {e}")
            return False

    def _migration_1_1_to_1_2_up(self) -> bool:
        """1.1 -> 1.2 upgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Add memory versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id TEXT NOT NULL,
                        version INTEGER DEFAULT 1,
                        content TEXT NOT NULL,
                        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        changed_by TEXT,
                        change_description TEXT,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    )
                """)

                # Add version field to memories table
                try:
                    cursor.execute("ALTER TABLE memories ADD COLUMN current_version INTEGER DEFAULT 1")
                except sqlite3.OperationalError:
                    pass  # column may already exist

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_versions_memory ON memory_versions(memory_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_versions_version ON memory_versions(memory_id, version)")

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False

    def _migration_1_1_to_1_2_down(self) -> bool:
        """1.2 -> 1.1 downgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DROP TABLE IF EXISTS memory_versions")

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def _migration_1_2_to_1_3_up(self) -> bool:
        """1.2 -> 1.3 upgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Create full-text search virtual table
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id,
                        content,
                        category,
                        memory_type,
                        content='memories',
                        content_rowid='rowid'
                    )
                """)

                # Optimize existing indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type_importance ON memories(memory_type, importance)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")

                # Populate full-text search index
                cursor.execute("""
                    INSERT INTO memories_fts(rowid, id, content, category, memory_type)
                    SELECT rowid, id, content, category, memory_type FROM memories
                """)

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False

    def _migration_1_2_to_1_3_down(self) -> bool:
        """1.3 -> 1.2 downgrade script"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DROP TABLE IF EXISTS memories_fts")

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def export_data(self, export_path: Path, include_metadata: bool = True) -> bool:
        """
        Export data

        Args:
            export_path: Export path
            include_metadata: Whether to include metadata

        Returns:
            Whether export was successful
        """
        try:
            export_data = {
                'version': self.get_current_version(),
                'exported_at': datetime.now().isoformat(),
                'data': {}
            }

            with sqlite3.connect(self.db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Export all table data
                tables = ['memories', 'projects', 'preferences', 'memory_tags', 'memory_relations']

                for table in tables:
                    try:
                        cursor.execute(f"SELECT * FROM {table}")
                        rows = cursor.fetchall()
                        export_data['data'][table] = [dict(row) for row in rows]
                    except sqlite3.OperationalError:
                        pass  # table may not exist

            # Write export file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Data export completed: {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            return False

    def import_data(self, import_path: Path, merge_mode: bool = False) -> Tuple[bool, List[str]]:
        """
        Import data

        Args:
            import_path: Import path
            merge_mode: Whether merge mode (do not overwrite existing data)

        Returns:
            (Whether successful, error list)
        """
        errors = []

        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            source_version = import_data.get('version', '1.0.0')
            data = import_data.get('data', {})

            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Import each table data
                for table, rows in data.items():
                    if not rows:
                        continue

                    try:
                        # Get column names
                        columns = list(rows[0].keys())
                        placeholders = ', '.join(['?' for _ in columns])
                        columns_str = ', '.join(columns)

                        if merge_mode:
                            # Merge mode: use INSERT OR IGNORE
                            sql = f"INSERT OR IGNORE INTO {table} ({columns_str}) VALUES ({placeholders})"
                        else:
                            # Overwrite mode: clear then insert
                            cursor.execute(f"DELETE FROM {table}")
                            sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

                        for row in rows:
                            values = [row.get(col) for col in columns]
                            cursor.execute(sql, values)

                    except Exception as e:
                        errors.append(f"Import table {table} failed: {e}")

                conn.commit()

            self.logger.info(f"Data import completed: {import_path}")
            return len(errors) == 0, errors

        except Exception as e:
            self.logger.error(f"Data import failed: {e}")
            return False, [str(e)]

    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()

                # Get migration history
                cursor.execute("""
                    SELECT version, migration_id, description, applied_at
                    FROM schema_version ORDER BY applied_at DESC
                """)
                history = cursor.fetchall()

            return {
                'current_version': self.get_current_version(),
                'available_migrations': self.get_available_migrations(),
                'migration_history': [
                    {
                        'version': row[0],
                        'migration_id': row[1],
                        'description': row[2],
                        'applied_at': row[3]
                    }
                    for row in history
                ]
            }

        except Exception as e:
            self.logger.error(f"Failed to get migration status: {e}")
            return {
                'current_version': self.get_current_version(),
                'available_migrations': [],
                'migration_history': []
            }