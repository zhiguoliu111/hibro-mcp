#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test summary generator
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
from src.hibro.knowledge.graph_storage import (
    GraphStorage, GraphNode, GraphRelation,
    GraphNodeType, RelationType
)
from src.hibro.knowledge.summary_generator import SummaryGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_summary_generator():
    """Test summary generator"""

    logger.info("=" * 60)
    logger.info("Starting summary generator test")
    logger.info("=" * 60)

    try:
        # 1. Initialize
        logger.info("\n1. Initializing database and storage layer")

        config = Config()
        db_manager = DatabaseManager(config)
        db_manager.initialize_database()

        # Ensure migration is executed
        migration_manager = MigrationManager(db_manager)
        migration_manager.migrate()

        storage = GraphStorage(db_manager)
        generator = SummaryGenerator(storage)
        logger.info("Storage layer and generator initialized successfully")

        # 2. Create test data
        logger.info("\n2. Creating test data")

        project_path = "/test/summary_project"

        # Create file nodes
        file_node1 = GraphNode(
            node_type=GraphNodeType.FILE,
            name="main.py",
            file_path="src/main.py",
            importance=0.9,
            project_path=project_path,
            metadata={"language": "python", "lines": 200}
        )
        file_id1 = storage.create_node(file_node1)
        logger.info(f"Created file node: {file_id1}")

        file_node2 = GraphNode(
            node_type=GraphNodeType.FILE,
            name="utils.py",
            file_path="src/utils.py",
            importance=0.7,
            project_path=project_path,
            metadata={"language": "python", "lines": 100}
        )
        file_id2 = storage.create_node(file_node2)
        logger.info(f"Created file node: {file_id2}")

        # Create class nodes
        class_node1 = GraphNode(
            node_type=GraphNodeType.CLASS,
            name="Application",
            file_path="src/main.py",
            line_number=10,
            importance=0.95,
            project_path=project_path,
            metadata={
                "methods": ["__init__", "run", "shutdown"],
                "inherits_from": [],
                "docstring": "Main application class for managing the system"
            }
        )
        class_id1 = storage.create_node(class_node1)
        logger.info(f"Created class node: {class_id1}")

        class_node2 = GraphNode(
            node_type=GraphNodeType.CLASS,
            name="Config",
            file_path="src/utils.py",
            line_number=5,
            importance=0.8,
            project_path=project_path,
            metadata={
                "methods": ["load", "save", "validate"],
                "inherits_from": ["BaseModel"],
                "docstring": "Configuration management class"
            }
        )
        class_id2 = storage.create_node(class_node2)
        logger.info(f"Created class node: {class_id2}")

        # Create function nodes
        func_node1 = GraphNode(
            node_type=GraphNodeType.FUNCTION,
            name="initialize_app",
            file_path="src/main.py",
            line_number=50,
            importance=0.85,
            project_path=project_path,
            metadata={
                "signature": "def initialize_app(config: Config) -> Application",
                "parameters": ["config"],
                "return_type": "Application"
            }
        )
        func_id1 = storage.create_node(func_node1)
        logger.info(f"Created function node: {func_id1}")

        func_node2 = GraphNode(
            node_type=GraphNodeType.FUNCTION,
            name="parse_args",
            file_path="src/utils.py",
            line_number=30,
            importance=0.7,
            project_path=project_path,
            metadata={
                "signature": "def parse_args(args: List[str]) -> Dict",
                "parameters": ["args"],
                "return_type": "Dict"
            }
        )
        func_id2 = storage.create_node(func_node2)
        logger.info(f"Created function node: {func_id2}")

        # Create API endpoint nodes
        api_node1 = GraphNode(
            node_type=GraphNodeType.API_ENDPOINT,
            name="/api/status",
            file_path="src/main.py",
            line_number=100,
            importance=0.9,
            project_path=project_path,
            metadata={
                "method": "GET",
                "handler_function": "get_status"
            }
        )
        api_id1 = storage.create_node(api_node1)
        logger.info(f"Created API endpoint node: {api_id1}")

        api_node2 = GraphNode(
            node_type=GraphNodeType.API_ENDPOINT,
            name="/api/config",
            file_path="src/main.py",
            line_number=120,
            importance=0.85,
            project_path=project_path,
            metadata={
                "method": "POST",
                "handler_function": "update_config"
            }
        )
        api_id2 = storage.create_node(api_node2)
        logger.info(f"Created API endpoint node: {api_id2}")

        # Create relations
        # File contains class
        storage.create_relation(GraphRelation(
            source_node_id=file_id1,
            target_node_id=class_id1,
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ))

        # File imports file
        storage.create_relation(GraphRelation(
            source_node_id=file_id1,
            target_node_id=file_id2,
            relation_type=RelationType.IMPORTS,
            weight=0.8
        ))

        # Class contains function
        storage.create_relation(GraphRelation(
            source_node_id=class_id1,
            target_node_id=func_id1,
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ))

        logger.info("Test data created successfully")

        # 3. Test lightweight summary
        logger.info("\n3. Testing lightweight summary generation")

        lightweight_summary = generator.generate_lightweight_summary(project_path)

        if lightweight_summary:
            logger.info("Lightweight summary generated successfully")
            logger.info(f"  - Project path: {lightweight_summary.get('project_path')}")
            logger.info(f"  - Summary type: {lightweight_summary.get('summary_type')}")

            stats = lightweight_summary.get("statistics", {})
            logger.info(f"  - Total files: {stats.get('total_files', 0)}")
            logger.info(f"  - Total classes: {stats.get('total_classes', 0)}")
            logger.info(f"  - Total functions: {stats.get('total_functions', 0)}")
            logger.info(f"  - Total API endpoints: {stats.get('total_api_endpoints', 0)}")

            core_modules = lightweight_summary.get("core_modules", [])
            logger.info(f"  - Core modules: {len(core_modules)}")

            key_classes = lightweight_summary.get("key_classes", [])
            logger.info(f"  - Key classes: {key_classes}")

            assert lightweight_summary["project_path"] == project_path
            assert lightweight_summary["summary_type"] == "lightweight"
            assert stats.get("total_files", 0) >= 2
            assert stats.get("total_classes", 0) >= 2
        else:
            logger.error("Lightweight summary generation failed")
            return False

        # 4. Test medium summary
        logger.info("\n4. Testing medium summary generation")

        medium_summary = generator.generate_medium_summary(project_path)

        if medium_summary:
            logger.info("Medium summary generated successfully")
            logger.info(f"  - Project path: {medium_summary.get('project_path')}")
            logger.info(f"  - Summary type: {medium_summary.get('summary_type')}")

            classes = medium_summary.get("classes", [])
            logger.info(f"  - Classes: {len(classes)}")
            if classes:
                first_class = classes[0]
                logger.info(f"    First class: {first_class.get('name')}")
                logger.info(f"    Methods: {first_class.get('methods', [])}")

            functions = medium_summary.get("functions", [])
            logger.info(f"  - Functions: {len(functions)}")
            if functions:
                first_func = functions[0]
                logger.info(f"    First function: {first_func.get('name')}")

            api_endpoints = medium_summary.get("api_endpoints", [])
            logger.info(f"  - API endpoints: {len(api_endpoints)}")

            assert medium_summary["project_path"] == project_path
            assert medium_summary["summary_type"] == "medium"
            assert len(classes) >= 2
            assert len(functions) >= 2
            assert len(api_endpoints) >= 2
        else:
            logger.error("Medium summary generation failed")
            return False

        # 5. Test node details
        logger.info("\n5. Testing node details retrieval")

        node_details = generator.get_node_details(class_id1)

        if node_details:
            logger.info("Node details retrieved successfully")
            logger.info(f"  - Node ID: {node_details.get('node_id')}")
            logger.info(f"  - Node type: {node_details.get('node_type')}")
            logger.info(f"  - Name: {node_details.get('name')}")
            logger.info(f"  - File path: {node_details.get('file_path')}")
            logger.info(f"  - Importance: {node_details.get('importance')}")

            relations = node_details.get("relations", {})
            logger.info(f"  - Outgoing relations: {len(relations.get('outgoing', []))}")
            logger.info(f"  - Incoming relations: {len(relations.get('incoming', []))}")

            assert node_details["node_id"] == class_id1
            assert node_details["node_type"] == "class"
            assert node_details["name"] == "Application"
        else:
            logger.error("Node details retrieval failed")
            return False

        # 6. Test text summary formatting
        logger.info("\n6. Testing text summary formatting")

        lightweight_text = generator.generate_text_summary(lightweight_summary)
        logger.info("Lightweight text summary:")
        logger.info("-" * 40)
        logger.info(lightweight_text[:500])
        logger.info("-" * 40)
        assert "Project:" in lightweight_text
        assert "Statistics:" in lightweight_text

        medium_text = generator.generate_text_summary(medium_summary)
        logger.info("Medium text summary:")
        logger.info("-" * 40)
        logger.info(medium_text[:800])
        logger.info("-" * 40)
        assert "Project:" in medium_text
        assert "Key Classes:" in medium_text

        # 7. Test empty project
        logger.info("\n7. Testing empty project handling")

        empty_summary = generator.generate_lightweight_summary("/nonexistent/project")
        # Should return empty dict or minimal summary
        logger.info(f"Empty project summary: {empty_summary}")

        # 8. Cleanup test data
        logger.info("\n8. Cleaning up test data")

        for node_id in [file_id1, file_id2, class_id1, class_id2, func_id1, func_id2, api_id1, api_id2]:
            storage.delete_node(node_id)
        logger.info("Test data cleaned up")

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
    success = test_summary_generator()
    sys.exit(0 if success else 1)
