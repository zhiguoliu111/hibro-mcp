#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test knowledge graph with real projects

Tests the core functionality using actual project code.
Can be used to verify knowledge graph works correctly.
"""

import sys
import logging
from pathlib import Path

# Add hibro to path
hibro_path = Path(__file__).parent.parent
sys.path.insert(0, str(hibro_path / "src"))

from hibro.utils.config import Config
from hibro.storage.database import DatabaseManager
from hibro.storage.migration_manager import MigrationManager
from hibro.knowledge.graph_storage import GraphStorage, GraphNode, GraphNodeType
from hibro.knowledge.code_query import KnowledgeGraphQuery
from hibro.knowledge.session_update import SessionUpdateManager
from hibro.parsers.code_analyzer import CodeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_with_project(project_path: str) -> bool:
    """Test knowledge graph with a project"""
    logger.info("=" * 60)
    logger.info(f"Testing knowledge graph with: {project_path}")
    logger.info("=" * 60)

    try:
        # Initialize
        config = Config()
        db_manager = DatabaseManager(config)
        db_manager.initialize_database()

        migration_manager = MigrationManager(db_manager)
        migration_manager.migrate()

        storage = GraphStorage(db_manager)
        query = KnowledgeGraphQuery(storage)
        manager = SessionUpdateManager(project_path=project_path, storage=storage)

        # 1. Check if update needed
        logger.info("\n1. Checking if knowledge graph needs update...")
        check = manager.check_needs_update()
        logger.info(f"   Needs update: {check.needs_update}")
        logger.info(f"   Reason: {check.reason}")

        # 2. Perform full scan
        logger.info("\n2. Performing full scan...")
        stats = manager.perform_full_scan()
        logger.info(f"   Files processed: {stats['files_processed']}")
        logger.info(f"   Classes added: {stats['classes_added']}")
        logger.info(f"   Functions added: {stats['functions_added']}")
        logger.info(f"   Errors: {stats['errors']}")

        # 3. Query quick context
        logger.info("\n3. Testing quick context query...")
        result = query.get_quick_project_context(project_path)
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Query time: {result.query_time_ms:.2f}ms")
        logger.info(f"   Token estimate: {result.token_estimate}")

        if result.success and result.data:
            logger.info(f"   File count: {result.data.file_count}")
            logger.info(f"   Class count: {result.data.class_count}")
            logger.info(f"   Function count: {result.data.function_count}")
            logger.info(f"   Key classes: {result.data.key_classes[:5]}")

        # 4. Search code
        if result.data and result.data.key_classes:
            search_term = result.data.key_classes[0] if result.data.key_classes else None
            if search_term:
                logger.info(f"\n4. Testing code search for: {search_term}...")
                result = query.search_code(project_path, search_term)
                logger.info(f"   Found: {len(result.data)} results")

        # 5. Cleanup
        logger.info("\n5. Cleaning up test data...")
        nodes = storage.search_nodes(project_path=project_path, limit=10000)
        for node in nodes:
            storage.delete_node(node.node_id)
        logger.info(f"   Deleted {len(nodes)} nodes")

        logger.info("\n✓ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests"""
    if len(sys.argv) < 2:
        print("Usage: python test_knowledge_graph.py <project_path>")
        print("Example: python test_knowledge_graph.py D:/projects/my-project")
        return 1

    project_path = sys.argv[1]

    if not Path(project_path).exists():
        print(f"Error: Project path does not exist: {project_path}")
        return 1

    success = test_with_project(project_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
