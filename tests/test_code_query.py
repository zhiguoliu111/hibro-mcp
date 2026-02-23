#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test knowledge graph code query
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
from src.hibro.knowledge.graph_storage import GraphStorage, GraphNode, GraphNodeType
from src.hibro.knowledge.code_query import (
    KnowledgeGraphQuery, QueryResult, ProjectContext
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_code_query():
    """Test knowledge graph code query"""

    logger.info("=" * 60)
    logger.info("Starting code query test")
    logger.info("=" * 60)

    try:
        # Initialize database and storage
        config = Config()
        db_manager = DatabaseManager(config)
        db_manager.initialize_database()

        migration_manager = MigrationManager(db_manager)
        migration_manager.migrate()

        storage = GraphStorage(db_manager)

        # Create query interface
        query = KnowledgeGraphQuery(storage)

        # Setup test data
        project_path = "/test/query_project"

        # Create test nodes
        logger.info("\n1. Creating test data")

        # File node
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name="app.py",
            file_path="src/app.py",
            importance=0.9,
            project_path=project_path
        )
        file_id = storage.create_node(file_node)

        # Class node
        class_node = GraphNode(
            node_type=GraphNodeType.CLASS,
            name="Application",
            file_path="src/app.py",
            line_number=10,
            importance=0.95,
            project_path=project_path,
            metadata={"methods": ["run", "stop"], "bases": ["BaseApp"]}
        )
        class_id = storage.create_node(class_node)

        # Function node
        func_node = GraphNode(
            node_type=GraphNodeType.FUNCTION,
            name="main",
            file_path="src/app.py",
            line_number=50,
            importance=0.8,
            project_path=project_path,
            metadata={"is_async": False, "return_type": "int"}
        )
        func_id = storage.create_node(func_node)

        # API endpoint node
        api_node = GraphNode(
            node_type=GraphNodeType.API_ENDPOINT,
            name="GET /api/status",
            file_path="src/app.py",
            line_number=100,
            importance=0.9,
            project_path=project_path,
            metadata={"method": "GET", "path": "/api/status"}
        )
        api_id = storage.create_node(api_node)

        logger.info("Test data created")

        # 2. Test get_quick_project_context
        logger.info("\n2. Testing get_quick_project_context")

        result = query.get_quick_project_context(project_path)

        assert result.success, f"Query should succeed, error: {result.error}"
        assert result.data is not None, "Should return data"
        assert result.query_time_ms < 1000, f"Query should be fast (<1s), took {result.query_time_ms}ms"
        logger.info(f"Query time: {result.query_time_ms:.2f}ms")
        logger.info(f"Token estimate: {result.token_estimate}")

        if isinstance(result.data, ProjectContext):
            logger.info(f"File count: {result.data.file_count}")
            logger.info(f"Class count: {result.data.class_count}")
            logger.info(f"Function count: {result.data.function_count}")
            assert result.data.class_count >= 1, "Should have at least 1 class"

        # 3. Test get_detailed_context
        logger.info("\n3. Testing get_detailed_context")

        result = query.get_detailed_context(project_path)

        assert result.success, f"Query should succeed, error: {result.error}"
        logger.info(f"Query time: {result.query_time_ms:.2f}ms")
        logger.info(f"Token estimate: {result.token_estimate}")

        if result.data:
            classes = result.data.get('classes', [])
            logger.info(f"Classes in summary: {len(classes)}")

        # 4. Test search_code
        logger.info("\n4. Testing search_code")

        result = query.search_code(project_path, "Application")

        assert result.success, f"Search should succeed, error: {result.error}"
        assert len(result.data) >= 1, "Should find at least 1 result"
        logger.info(f"Search results: {len(result.data)}")
        for item in result.data[:3]:
            logger.info(f"  - {item['name']} ({item['type']})")

        # 5. Test get_class_hierarchy
        logger.info("\n5. Testing get_class_hierarchy")

        result = query.get_class_hierarchy(project_path, "Application")

        assert result.success, f"Query should succeed, error: {result.error}"
        logger.info(f"Class: {result.data['class_name']}")
        logger.info(f"Parent: {result.data['parent']}")
        logger.info(f"Methods: {result.data['methods']}")

        # 6. Test get_file_structure
        logger.info("\n6. Testing get_file_structure")

        result = query.get_file_structure(project_path, "src/app.py")

        assert result.success, f"Query should succeed, error: {result.error}"
        logger.info(f"File: {result.data['file_path']}")
        logger.info(f"Classes: {len(result.data['classes'])}")
        logger.info(f"Functions: {len(result.data['functions'])}")
        logger.info(f"API endpoints: {len(result.data['api_endpoints'])}")

        # 7. Test get_api_endpoints
        logger.info("\n7. Testing get_api_endpoints")

        result = query.get_api_endpoints(project_path)

        assert result.success, f"Query should succeed, error: {result.error}"
        logger.info(f"API endpoints found: {len(result.data)}")
        for endpoint in result.data[:3]:
            logger.info(f"  - {endpoint['method']} {endpoint['path']}")

        # 8. Test get_progressive_context
        logger.info("\n8. Testing get_progressive_context")

        # Test with different token budgets
        for available in [50, 400, 1500, 5000]:
            result = query.get_progressive_context(project_path, 0, available)
            assert result.success, f"Progressive context should succeed with {available} tokens"
            logger.info(f"Budget {available}: tokens_used={result.token_estimate}")

        # 9. Test caching
        logger.info("\n9. Testing caching")

        # First query
        result1 = query.get_quick_project_context(project_path)

        # Second query (should hit cache)
        result2 = query.get_quick_project_context(project_path)

        logger.info(f"First query: {result1.query_time_ms:.2f}ms")
        logger.info(f"Second query (cached): {result2.query_time_ms:.2f}ms")

        # Clear cache
        query.clear_cache()
        logger.info("Cache cleared")

        # 10. Test QueryResult
        logger.info("\n10. Testing QueryResult")

        test_result = QueryResult(
            success=True,
            data={"test": "data"},
            query_time_ms=1.5,
            token_estimate=10
        )
        assert test_result.success, "Result should be successful"
        assert test_result.data["test"] == "data", "Data should match"
        logger.info("QueryResult works correctly")

        # Cleanup test data
        logger.info("\n11. Cleaning up test data")
        for node_id in [file_id, class_id, func_id, api_id]:
            storage.delete_node(node_id)
        logger.info("Test data cleaned up")

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

        return True

    except AssertionError as e:
        logger.error(f"\nAssertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_code_query()
    sys.exit(0 if success else 1)
