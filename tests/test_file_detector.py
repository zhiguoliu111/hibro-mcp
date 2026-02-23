#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file change detector
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.knowledge.file_detector import FileChangeDetector, FileChange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_file_detector():
    """Test file change detector"""

    logger.info("=" * 60)
    logger.info("Starting file change detector test")
    logger.info("=" * 60)

    try:
        # 1. Initialize detector
        logger.info("\n1. Initializing file change detector")

        project_path = str(project_root)
        detector = FileChangeDetector(project_path)
        logger.info(f"Detector initialized, project path: {project_path}")

        # 2. Test file scanning
        logger.info("\n2. Testing file scanning")

        files = detector.scan_project_files()
        logger.info(f"Scanned {len(files)} source code files")

        # Show first 5 files
        if files:
            logger.info("  First 5 files:")
            for file in files[:5]:
                logger.info(f"    - {file}")

        assert len(files) > 0, "Should scan at least some files"

        # 3. Test file info retrieval
        logger.info("\n3. Testing file info retrieval")

        if files:
            test_file = files[0]
            file_info = detector.get_file_info(test_file)

            if file_info:
                logger.info(f"File info retrieved: {test_file}")
                logger.info(f"  - Size: {file_info['size']} bytes")
                logger.info(f"  - Mtime: {file_info['mtime']}")
                logger.info(f"  - Hash: {file_info['hash'][:16]}...")
                assert file_info['hash'] is not None
            else:
                logger.error(f"File info retrieval failed: {test_file}")
                return False

        # 4. Test file state initialization
        logger.info("\n4. Testing file state initialization")

        # Initialize only first 10 files to speed up test
        test_files = files[:10]
        detector.initialize_file_states(test_files)

        cache_stats = detector.get_cache_stats()
        logger.info(f"File state initialized, cached {cache_stats['cached_files']} files")
        assert cache_stats['cached_files'] == len(test_files)

        # 5. Test change detection (no changes)
        logger.info("\n5. Testing change detection (no changes)")

        changes = detector.detect_changes(test_files)
        logger.info(f"Detection complete, found {len(changes)} changes")
        assert len(changes) == 0, "Should have no changes"

        # 6. Test mtime-based detection
        logger.info("\n6. Testing mtime-based detection")

        # Detect files modified in last 1 day
        since = datetime.now() - timedelta(days=1)
        recent_files = detector.detect_changes_by_mtime(since, test_files)
        logger.info(f"Files modified in last 1 day: {len(recent_files)}")

        # 7. Test file hash calculation
        logger.info("\n7. Testing file hash calculation")

        if test_files:
            test_file = test_files[0]
            hash1 = detector.calculate_file_hash(test_file)
            hash2 = detector.calculate_file_hash(test_file)

            assert hash1 == hash2, "Same file should have same hash"
            logger.info(f"File hash calculation consistent: {hash1[:16]}...")

        # 8. Test cache clearing
        logger.info("\n8. Testing cache clearing")

        detector.clear_cache()
        cache_stats = detector.get_cache_stats()
        logger.info(f"Cache cleared, current cache: {cache_stats['cached_files']} files")
        assert cache_stats['cached_files'] == 0

        # 9. Test non-existent file
        logger.info("\n9. Testing non-existent file")

        non_existent = "non_existent_file.py"
        mtime = detector.get_file_mtime(non_existent)
        file_hash = detector.calculate_file_hash(non_existent)
        file_info = detector.get_file_info(non_existent)

        assert mtime is None, "Non-existent file should return None"
        assert file_hash is None, "Non-existent file should return None"
        assert file_info is None, "Non-existent file should return None"
        logger.info("Non-existent file handled correctly")

        # 10. Test excluded directories
        logger.info("\n10. Testing excluded directories")

        all_files = detector.scan_project_files()
        excluded_found = False
        for file in all_files:
            # Check path parts (directory names), not the whole string
            path_parts = Path(file).parts
            for part in path_parts[:-1]:  # Exclude filename itself
                if part in FileChangeDetector.EXCLUDED_DIRS or part.startswith('.'):
                    excluded_found = True
                    logger.error(f"Found file in excluded directory: {file}")
                    break
            if excluded_found:
                break

        if not excluded_found:
            logger.info("Excluded directories filtered correctly")
        else:
            logger.error("Excluded directories filtering failed")
            return False

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
    success = test_file_detector()
    sys.exit(0 if success else 1)
