#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test session update manager
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.utils.config import Config
from src.hibro.storage.database import DatabaseManager
from src.hibro.storage.migration_manager import MigrationManager
from src.hibro.knowledge.graph_storage import GraphStorage
from src.hibro.knowledge.session_update import (
    SessionUpdateManager, SessionCheckResult, UpdateTask
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_session_update_manager():
    """Test session update manager"""

    logger.info("=" * 60)
    logger.info("Starting session update manager test")
    logger.info("=" * 60)

    try:
        # Initialize database and storage
        config = Config()
        db_manager = DatabaseManager(config)
        db_manager.initialize_database()

        migration_manager = MigrationManager(db_manager)
        migration_manager.migrate()

        storage = GraphStorage(db_manager)

        # 1. Test initialization
        logger.info("\n1. Testing initialization")

        manager = SessionUpdateManager(
            project_path=str(project_root),
            storage=storage
        )
        logger.info("SessionUpdateManager initialized successfully")

        # 2. Test check_needs_update
        logger.info("\n2. Testing check_needs_update")

        result = manager.check_needs_update()
        logger.info(f"Needs update: {result.needs_update}")
        logger.info(f"Reason: {result.reason}")
        logger.info(f"Files changed: {result.files_changed}")
        assert isinstance(result, SessionCheckResult), "Should return SessionCheckResult"

        # 3. Test metadata handling
        logger.info("\n3. Testing metadata handling")

        manager._metadata['test_key'] = 'test_value'
        manager._save_metadata()

        # Create new manager to test loading
        manager2 = SessionUpdateManager(
            project_path=str(project_root),
            storage=storage
        )
        assert manager2._metadata.get('test_key') == 'test_value', "Metadata should persist"
        logger.info("Metadata persistence works correctly")

        # 4. Test UpdateTask
        logger.info("\n4. Testing UpdateTask")

        task = UpdateTask(
            task_id="test_001",
            file_path="test.py",
            task_type="update",
            priority=1
        )
        assert task.status == 'pending', "New task should be pending"
        assert task.task_type == 'update', "Task type should be update"
        logger.info(f"UpdateTask created: {task.task_id}, type: {task.task_type}")

        # 5. Test queue functionality
        logger.info("\n5. Testing queue functionality")

        manager.queue_update("test_file.py", "update", priority=1)
        logger.info("Queued update task")

        # Check queue size (indirectly by checking it's not empty)
        assert not manager._update_queue.empty(), "Queue should have task"
        logger.info("Queue has pending task")

        # 6. Test file processing (with temp directory)
        logger.info("\n6. Testing file processing")

        temp_dir = tempfile.mkdtemp(prefix="hibro_session_test_")
        try:
            # Create a test Python file
            test_file = os.path.join(temp_dir, "test_module.py")
            with open(test_file, 'w') as f:
                f.write('''
class TestClass:
    """Test class."""
    def test_method(self):
        pass

def test_function():
    pass
''')

            # Create temp manager
            temp_storage = GraphStorage(db_manager)
            temp_manager = SessionUpdateManager(
                project_path=temp_dir,
                storage=temp_storage
            )

            # Initialize file detector with test file
            temp_manager.detector.scan_project_files()

            # Process the file
            stats = {'classes_added': 0, 'functions_added': 0, 'api_endpoints_added': 0}
            temp_manager._process_python_file("test_module.py", stats)

            logger.info(f"Processing stats: {stats}")
            assert stats['classes_added'] >= 1, "Should add at least 1 class"
            assert stats['functions_added'] >= 1, "Should add at least 1 function"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # 7. Test queue processor start/stop
        logger.info("\n7. Testing queue processor")

        manager.start_queue_processor()
        logger.info("Queue processor started")

        manager.stop_queue_processor()
        logger.info("Queue processor stopped")

        # 8. Test SessionCheckResult
        logger.info("\n8. Testing SessionCheckResult")

        check_result = SessionCheckResult(
            needs_update=True,
            last_scan_time=datetime.now(),
            files_changed=5,
            reason="Test check"
        )
        assert check_result.needs_update, "Should need update"
        assert check_result.files_changed == 5, "Files changed should be 5"
        logger.info("SessionCheckResult works correctly")

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
    success = test_session_update_manager()
    sys.exit(0 if success else 1)
