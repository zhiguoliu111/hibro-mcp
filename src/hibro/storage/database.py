#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database management module
Implements SQLite database creation, connection and basic operations
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from contextlib import contextmanager

from ..utils.config import Config


class DatabaseManager:
    """Database manager"""

    def __init__(self, config: Config):
        """
        Initialize database manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.db_path = config.get_database_path()
        self.logger = logging.getLogger('hibro.database')

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Get database connection (context manager)

        Yields:
            sqlite3.Connection: Database connection object
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dictionary-style access
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_database(self):
        """Initialize database structure"""
        try:
            with self.get_connection() as conn:
                self._create_tables(conn)
                self._create_indexes(conn)
                conn.commit()

            self.logger.info(f"Database initialization completed: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def _create_tables(self, conn: sqlite3.Connection):
        """Create database table structure"""

        # Memory main table (with knowledge graph support)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'conversation',
                importance REAL NOT NULL DEFAULT 0.5,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,  -- JSON format metadata
                graph_node_type TEXT,  -- Knowledge graph: node type (file, class, function, etc.)
                graph_metadata TEXT,  -- Knowledge graph: node metadata (JSON)
                file_hash TEXT,  -- Knowledge graph: file hash for change detection
                CONSTRAINT chk_importance CHECK (importance >= 0.0 AND importance <= 1.0)
            )
        """)

        # Project association table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT UNIQUE NOT NULL,
                tech_stack TEXT,  -- JSON format tech stack list
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        # Preference settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key),
                CONSTRAINT chk_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
            )
        """)

        # Memory relation table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                related_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (related_id) REFERENCES memories(id) ON DELETE CASCADE,
                CONSTRAINT chk_strength CHECK (strength >= 0.0 AND strength <= 1.0),
                CONSTRAINT chk_different_memories CHECK (memory_id != related_id)
            )
        """)

        # Project memory association table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                relevance_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(project_id, memory_id),
                CONSTRAINT chk_relevance CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0)
            )
        """)

        # System configuration table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.logger.info("Database table structure creation completed")

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes"""

        # Memory table indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)")

        # Project table indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(is_active)")

        # Preference table indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category)")

        # Relation table indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_memory_id ON memory_relations(memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_related_id ON memory_relations(related_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_type ON memory_relations(relation_type)")

        # Project memory association indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project_memories_project_id ON project_memories(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project_memories_memory_id ON project_memories(memory_id)")

        self.logger.info("Database index creation completed")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        try:
            with self.get_connection() as conn:
                stats = {}

                # Get record count for each table
                tables = ['memories', 'projects', 'preferences', 'memory_relations', 'project_memories']
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]

                # Get database file size
                if self.db_path.exists():
                    stats['db_size_bytes'] = self.db_path.stat().st_size
                    stats['db_size_mb'] = stats['db_size_bytes'] / (1024 * 1024)
                else:
                    stats['db_size_bytes'] = 0
                    stats['db_size_mb'] = 0.0

                # Get recent activity time
                cursor = conn.execute("""
                    SELECT MAX(last_accessed) FROM memories
                    WHERE last_accessed IS NOT NULL
                """)
                result = cursor.fetchone()
                stats['last_activity'] = result[0] if result and result[0] else None

                return stats

        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {e}")
            return {}

    def backup_database(self, backup_path: Optional[Path] = None) -> Path:
        """
        Backup database

        Args:
            backup_path: Backup file path, auto-generated if None

        Returns:
            Backup file path
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.db_path.parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f'memories_backup_{timestamp}.db'

        try:
            with self.get_connection() as source_conn:
                with sqlite3.connect(str(backup_path)) as backup_conn:
                    source_conn.backup(backup_conn)

            self.logger.info(f"Database backup completed: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            raise

    def vacuum_database(self):
        """Optimize database (clean up fragments)"""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")

            self.logger.info("Database optimization completed")

        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            raise

    def check_database_integrity(self) -> bool:
        """
        Check database integrity

        Returns:
            Whether database is intact
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                is_ok = result and result[0] == 'ok'

                if is_ok:
                    self.logger.info("Database integrity check passed")
                else:
                    self.logger.warning(f"Database integrity check failed: {result}")

                return is_ok

        except Exception as e:
            self.logger.error(f"Database integrity check failed: {e}")
            return False