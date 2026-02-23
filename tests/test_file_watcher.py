#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file watcher
"""

import sys
import os
import time
import logging
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.knowledge.file_watcher import (
    FileWatcher, FileChangeEvent, WATCHDOG_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_file_watcher():
    """Test file watcher"""

    logger.info("=" * 60)
    logger.info("Starting file watcher test")
    logger.info("=" * 60)

    if not WATCHDOG_AVAILABLE:
        logger.warning("watchdog not available, skipping tests")
        logger.info("Install watchdog with: pip install watchdog")
        return True

    try:
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp(prefix="hibro_test_")
        logger.info(f"Created temp directory: {temp_dir}")

        # Track events
        events_received = []

        def on_change(event: FileChangeEvent):
            events_received.append(event)
            logger.info(f"Event received: {event.event_type} - {event.file_path}")

        # 1. Test watcher initialization
        logger.info("\n1. Testing watcher initialization")

        watcher = FileWatcher(temp_dir, on_change=on_change)
        assert not watcher.is_running(), "Watcher should not be running initially"
        logger.info("Watcher initialized successfully")

        # 2. Test starting watcher
        logger.info("\n2. Testing watcher start")

        success = watcher.start()
        assert success, "Watcher should start successfully"
        assert watcher.is_running(), "Watcher should be running"
        logger.info("Watcher started successfully")

        # Give watcher time to initialize
        time.sleep(0.5)

        # 3. Test file change detection
        logger.info("\n3. Testing file change detection")

        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, 'w') as f:
            f.write("# Test file\nprint('hello')\n")

        # Wait for event to be processed
        time.sleep(2)

        # On Windows, file creation often triggers 'modified' instead of 'created'
        change_events = [e for e in events_received if e.event_type in ('created', 'modified')]
        logger.info(f"Change events received: {len(change_events)}")
        assert len(change_events) > 0, "Should detect file changes"

        # 4. Test file modification detection
        logger.info("\n4. Testing file modification detection")

        events_received.clear()

        with open(test_file, 'a') as f:
            f.write("# Modified\n")

        time.sleep(2)

        modify_events = [e for e in events_received if e.event_type == 'modified']
        logger.info(f"Modification events received: {len(modify_events)}")
        assert len(modify_events) > 0, "Should detect file modification"

        # 5. Test exclusion of non-monitored files
        logger.info("\n5. Testing exclusion of non-monitored files")

        events_received.clear()

        txt_file = os.path.join(temp_dir, "test.txt")
        with open(txt_file, 'w') as f:
            f.write("This is a text file\n")

        time.sleep(1.5)

        txt_events = [e for e in events_received if 'test.txt' in e.file_path]
        logger.info(f"TXT file events: {len(txt_events)}")
        assert len(txt_events) == 0, "Should not monitor .txt files"

        # 6. Test file deletion detection
        logger.info("\n6. Testing file deletion detection")

        events_received.clear()

        os.remove(test_file)

        time.sleep(2)

        delete_events = [e for e in events_received if e.event_type == 'deleted']
        logger.info(f"Deletion events received: {len(delete_events)}")
        assert len(delete_events) > 0, "Should detect file deletion"

        # 7. Test watcher stop
        logger.info("\n7. Testing watcher stop")

        watcher.stop()
        assert not watcher.is_running(), "Watcher should not be running after stop"
        logger.info("Watcher stopped successfully")

        # 8. Test _should_monitor method
        logger.info("\n8. Testing _should_monitor method")

        watcher2 = FileWatcher(temp_dir)

        # Should monitor
        assert watcher2._should_monitor(os.path.join(temp_dir, "test.py")), "Should monitor .py files"
        assert watcher2._should_monitor(os.path.join(temp_dir, "test.js")), "Should monitor .js files"
        assert watcher2._should_monitor(os.path.join(temp_dir, "test.ts")), "Should monitor .ts files"

        # Should not monitor
        assert not watcher2._should_monitor(os.path.join(temp_dir, "test.txt")), "Should not monitor .txt files"
        assert not watcher2._should_monitor(os.path.join(temp_dir, "test.md")), "Should not monitor .md files"

        # Should exclude directories
        node_modules = os.path.join(temp_dir, "node_modules", "test.js")
        os.makedirs(os.path.dirname(node_modules), exist_ok=True)
        assert not watcher2._should_monitor(node_modules), "Should exclude node_modules"

        logger.info("_should_monitor tests passed")

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

        return True

    except AssertionError as e:
        logger.error(f"\nAssertion failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on failure
        if 'temp_dir' in dir():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on failure
        if 'temp_dir' in dir():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


if __name__ == '__main__':
    success = test_file_watcher()
    sys.exit(0 if success else 1)
