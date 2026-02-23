#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration 001: Add knowledge graph support

Adds table fields and relation tables required for knowledge graph
"""

import sqlite3
import logging
from typing import Optional

logger = logging.getLogger('hibro.migration.001')


def check_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """
    Check if a column exists in a table

    Args:
        conn: Database connection
        table: Table name
        column: Column name

    Returns:
        Whether exists
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def check_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """
    Check if a table exists

    Args:
        conn: Database connection
        table: Table name

    Returns:
        Whether exists
    """
    cursor = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table,))
    return cursor.fetchone() is not None


def check_index_exists(conn: sqlite3.Connection, index: str) -> bool:
    """
    Check if an index exists

    Args:
        conn: Database connection
        index: Index name

    Returns:
        Whether exists
    """
    cursor = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND name=?
    """, (index,))
    return cursor.fetchone() is not None


def upgrade(conn: sqlite3.Connection) -> bool:
    """
    Execute upgrade migration

    Args:
        conn: Database connection

    Returns:
        Whether successful
    """
    try:
        logger.info("Starting migration 001: Add knowledge graph support")

        # 1. Extend memories table - add knowledge graph related fields
        if not check_column_exists(conn, 'memories', 'graph_node_type'):
            logger.info("Adding memories.graph_node_type field")
            conn.execute("""
                ALTER TABLE memories
                ADD COLUMN graph_node_type TEXT
            """)

        if not check_column_exists(conn, 'memories', 'graph_metadata'):
            logger.info("Adding memories.graph_metadata field")
            conn.execute("""
                ALTER TABLE memories
                ADD COLUMN graph_metadata TEXT
            """)

        if not check_column_exists(conn, 'memories', 'file_hash'):
            logger.info("Adding memories.file_hash field")
            conn.execute("""
                ALTER TABLE memories
                ADD COLUMN file_hash TEXT
            """)

        # 2. Create knowledge graph relations table
        if not check_table_exists(conn, 'knowledge_relations'):
            logger.info("Creating knowledge_relations table")
            conn.execute("""
                CREATE TABLE knowledge_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_memory_id INTEGER NOT NULL,
                    target_memory_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 0.5,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    CONSTRAINT chk_weight CHECK (weight >= 0.0 AND weight <= 1.0),
                    CONSTRAINT chk_different_nodes CHECK (source_memory_id != target_memory_id)
                )
            """)

        # 3. Create indexes
        if not check_index_exists(conn, 'idx_graph_node_type'):
            logger.info("Creating idx_graph_node_type index")
            conn.execute("""
                CREATE INDEX idx_graph_node_type
                ON memories(graph_node_type)
            """)

        if not check_index_exists(conn, 'idx_file_hash'):
            logger.info("Creating idx_file_hash index")
            conn.execute("""
                CREATE INDEX idx_file_hash
                ON memories(file_hash)
            """)

        if not check_index_exists(conn, 'idx_project_category'):
            logger.info("Creating idx_project_category index")
            conn.execute("""
                CREATE INDEX idx_project_category
                ON memories(category)
            """)

        if not check_index_exists(conn, 'idx_relation_source'):
            logger.info("Creating idx_relation_source index")
            conn.execute("""
                CREATE INDEX idx_relation_source
                ON knowledge_relations(source_memory_id)
            """)

        if not check_index_exists(conn, 'idx_relation_target'):
            logger.info("Creating idx_relation_target index")
            conn.execute("""
                CREATE INDEX idx_relation_target
                ON knowledge_relations(target_memory_id)
            """)

        if not check_index_exists(conn, 'idx_relation_type'):
            logger.info("Creating idx_relation_type index")
            conn.execute("""
                CREATE INDEX idx_relation_type
                ON knowledge_relations(relation_type)
            """)

        conn.commit()

        logger.info("Migration 001 executed successfully")
        return True

    except Exception as e:
        logger.error(f"Migration 001 failed: {e}")
        conn.rollback()
        return False


def downgrade(conn: sqlite3.Connection) -> bool:
    """
    Execute downgrade migration (rollback)

    Note: SQLite does not support DROP COLUMN, so downgrade is limited

    Args:
        conn: Database connection

    Returns:
        Whether successful
    """
    try:
        logger.info("Starting migration 001 rollback")

        # Drop knowledge graph relations table
        if check_table_exists(conn, 'knowledge_relations'):
            logger.info("Dropping knowledge_relations table")
            conn.execute("DROP TABLE knowledge_relations")

        # Drop indexes
        indexes = [
            'idx_graph_node_type',
            'idx_file_hash',
            'idx_relation_source',
            'idx_relation_target',
            'idx_relation_type'
        ]

        for index in indexes:
            if check_index_exists(conn, index):
                logger.info(f"Dropping index {index}")
                conn.execute(f"DROP INDEX {index}")

        # Note: SQLite does not support DROP COLUMN
        # Full rollback would require rebuilding the entire memories table

        conn.commit()

        logger.info("Migration 001 rollback successful")
        return True

    except Exception as e:
        logger.error(f"Migration 001 rollback failed: {e}")
        conn.rollback()
        return False
