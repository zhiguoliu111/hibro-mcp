#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test knowledge graph storage layer
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_graph_storage():
    """Test knowledge graph storage layer"""

    logger.info("=" * 60)
    logger.info("Starting graph storage layer test")
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
        logger.info("Storage layer initialized successfully")

        # 2. Test node creation
        logger.info("\n2. Testing node creation")

        # Create file node
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name="test_file.py",
            file_path="src/test/test_file.py",
            importance=0.8,
            file_hash=GraphStorage.calculate_file_hash("test content"),
            project_path="/test/project",
            metadata={"language": "python", "lines": 100}
        )
        file_node_id = storage.create_node(file_node)
        logger.info(f"File node created: ID={file_node_id}")

        # Create class node
        class_node = GraphNode(
            node_type=GraphNodeType.CLASS,
            name="TestClass",
            file_path="src/test/test_file.py",
            line_number=10,
            importance=0.9,
            project_path="/test/project",
            metadata={"methods": ["__init__", "test_method"], "inherits_from": []}
        )
        class_node_id = storage.create_node(class_node)
        logger.info(f"Class node created: ID={class_node_id}")

        # Create function node
        func_node = GraphNode(
            node_type=GraphNodeType.FUNCTION,
            name="test_method",
            file_path="src/test/test_file.py",
            line_number=20,
            importance=0.7,
            project_path="/test/project",
            metadata={"parameters": ["self", "arg1"], "return_type": "bool"}
        )
        func_node_id = storage.create_node(func_node)
        logger.info(f"Function node created: ID={func_node_id}")

        # 3. Test node retrieval
        logger.info("\n3. Testing node retrieval")

        retrieved_node = storage.get_node(file_node_id)
        if retrieved_node:
            logger.info(f"Node retrieved: {retrieved_node.name}")
            assert retrieved_node.node_type == GraphNodeType.FILE
            assert retrieved_node.name == "test_file.py"
        else:
            logger.error("Node retrieval failed")
            return False

        # 4. Test node search
        logger.info("\n4. Testing node search")

        # Search by type
        class_nodes = storage.search_nodes(node_type=GraphNodeType.CLASS, project_path="/test/project")
        logger.info(f"Found {len(class_nodes)} class nodes")
        assert len(class_nodes) >= 1

        # Search by file path
        file_nodes = storage.get_nodes_by_file("src/test/test_file.py", "/test/project")
        logger.info(f"Found {len(file_nodes)} nodes in file")
        assert len(file_nodes) >= 3

        # 5. Test node update
        logger.info("\n5. Testing node update")

        retrieved_node.importance = 0.95
        retrieved_node.metadata["updated"] = True
        success = storage.update_node(retrieved_node)
        if success:
            logger.info("Node updated successfully")

            # Verify update
            updated_node = storage.get_node(file_node_id)
            assert updated_node.importance == 0.95
            assert updated_node.metadata.get("updated") == True
        else:
            logger.error("Node update failed")
            return False

        # 6. Test relation creation
        logger.info("\n6. Testing relation creation")

        # Create contains relation (file contains class)
        contains_relation = GraphRelation(
            source_node_id=file_node_id,
            target_node_id=class_node_id,
            relation_type=RelationType.CONTAINS,
            weight=1.0,
            metadata={"description": "file contains class"}
        )
        contains_rel_id = storage.create_relation(contains_relation)
        logger.info(f"Contains relation created: ID={contains_rel_id}")

        # Create contains relation (class contains function)
        contains_relation2 = GraphRelation(
            source_node_id=class_node_id,
            target_node_id=func_node_id,
            relation_type=RelationType.CONTAINS,
            weight=1.0,
            metadata={"description": "class contains method"}
        )
        contains_rel_id2 = storage.create_relation(contains_relation2)
        logger.info(f"Contains relation created: ID={contains_rel_id2}")

        # 7. Test relation retrieval
        logger.info("\n7. Testing relation retrieval")

        # Get outgoing relations of file node
        outgoing_relations = storage.get_node_relations(file_node_id, direction='outgoing')
        logger.info(f"File node has {len(outgoing_relations)} outgoing relations")
        assert len(outgoing_relations) >= 1

        for relation, related_node in outgoing_relations:
            logger.info(f"  - {relation.relation_type.value} -> {related_node.name}")

        # Get all relations of class node
        all_relations = storage.get_node_relations(class_node_id, direction='both')
        logger.info(f"Class node has {len(all_relations)} relations")
        assert len(all_relations) >= 2

        # 8. Test relation deletion
        logger.info("\n8. Testing relation deletion")

        success = storage.delete_relation(contains_rel_id2)
        if success:
            logger.info("Relation deleted successfully")

            # Verify deletion
            deleted_relation = storage.get_relation(contains_rel_id2)
            assert deleted_relation is None
        else:
            logger.error("Relation deletion failed")
            return False

        # 9. Test node deletion
        logger.info("\n9. Testing node deletion")

        success = storage.delete_node(func_node_id)
        if success:
            logger.info("Node deleted successfully")

            # Verify deletion
            deleted_node = storage.get_node(func_node_id)
            assert deleted_node is None
        else:
            logger.error("Node deletion failed")
            return False

        # 10. Test file hash calculation
        logger.info("\n10. Testing file hash calculation")

        hash1 = GraphStorage.calculate_file_hash("test content")
        hash2 = GraphStorage.calculate_file_hash("test content")
        hash3 = GraphStorage.calculate_file_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        logger.info("File hash calculation correct")

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
    success = test_graph_storage()
    sys.exit(0 if success else 1)
