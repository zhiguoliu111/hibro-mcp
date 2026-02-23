#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test database migration functionality
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.utils.config import Config
from src.hibro.storage.database import DatabaseManager
from src.hibro.storage.migration_manager import MigrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_migration():
    """Test migration functionality"""

    logger.info("=" * 60)
    logger.info("Starting database migration test")
    logger.info("=" * 60)

    try:
        # 1. Initialize config and database manager
        logger.info("\n1. Initializing database manager")

        config = Config()
        db_manager = DatabaseManager(config)

        # Ensure database is initialized
        db_manager.initialize_database()
        logger.info("Database manager initialized successfully")

        # 2. Create migration manager
        logger.info("\n2. Creating migration manager")

        migration_manager = MigrationManager(db_manager)
        logger.info("Migration manager created successfully")

        # 3. Get migration status
        logger.info("\n3. Getting migration status")

        status = migration_manager.get_migration_status()
        logger.info(f"Current version: {status['current_version']}")
        logger.info(f"Applied migrations: {status['applied']}")
        logger.info(f"Available migrations: {status['available']}")
        logger.info(f"Pending migrations: {status['pending']}")

        # 4. Execute migration
        if status['pending']:
            logger.info("\n4. Executing pending migrations")

            success = migration_manager.migrate()

            if success:
                logger.info("Migration executed successfully")
            else:
                logger.error("Migration execution failed")
                return False
        else:
            logger.info("\n4. No pending migrations")

        # 5. Verify migration results
        logger.info("\n5. Verifying migration results")

        with db_manager.get_connection() as conn:
            # Check if memories table has new fields
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            required_columns = ['graph_node_type', 'graph_metadata', 'file_hash']
            for col in required_columns:
                if col in columns:
                    logger.info(f"Column {col} exists")
                else:
                    logger.error(f"Column {col} does not exist")
                    return False

            # Check if knowledge_relations table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relations'
            """)
            if cursor.fetchone():
                logger.info("knowledge_relations table exists")
            else:
                logger.error("knowledge_relations table does not exist")
                return False

            # Check if indexes exist
            required_indexes = [
                'idx_graph_node_type',
                'idx_file_hash',
                'idx_relation_source',
                'idx_relation_target',
                'idx_relation_type'
            ]

            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index'
            """)
            existing_indexes = {row[0] for row in cursor.fetchall()}

            for idx in required_indexes:
                if idx in existing_indexes:
                    logger.info(f"Index {idx} exists")
                else:
                    logger.error(f"Index {idx} does not exist")
                    return False

        # 6. Get final status
        logger.info("\n6. Getting final migration status")

        final_status = migration_manager.get_migration_status()
        logger.info(f"Current version: {final_status['current_version']}")
        logger.info(f"Applied migrations: {final_status['applied']}")

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_migration()
    sys.exit(0 if success else 1)
