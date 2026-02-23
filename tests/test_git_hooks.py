#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Git hooks
"""

import sys
import os
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.knowledge.git_hooks import (
    GitHooksManager, get_branch_name, get_commit_hash
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_git_hooks():
    """Test Git hooks"""

    logger.info("=" * 60)
    logger.info("Starting Git hooks test")
    logger.info("=" * 60)

    try:
        # Use current project as test repo
        repo_path = str(project_root)

        # 1. Test GitHooksManager initialization
        logger.info("\n1. Testing GitHooksManager initialization")

        manager = GitHooksManager(repo_path)
        assert manager.is_git_repo(), "Project should be a Git repository"
        logger.info("GitHooksManager initialized successfully")

        # 2. Test getting installed hooks
        logger.info("\n2. Testing get_installed_hooks")

        installed = manager.get_installed_hooks()
        logger.info(f"Currently installed hooks: {installed}")

        # 3. Test get_branch_name
        logger.info("\n3. Testing get_branch_name")

        branch = get_branch_name(repo_path)
        logger.info(f"Current branch: {branch}")
        assert branch is not None, "Should get branch name"
        assert branch != "", "Branch name should not be empty"

        # 4. Test get_commit_hash
        logger.info("\n4. Testing get_commit_hash")

        commit = get_commit_hash(repo_path)
        logger.info(f"Current commit: {commit}")
        assert commit is not None, "Should get commit hash"
        assert len(commit) == 8, "Should return short hash (8 chars)"

        # 5. Test hooks directory existence
        logger.info("\n5. Testing hooks directory")

        hooks_dir = manager.hooks_dir
        logger.info(f"Hooks directory: {hooks_dir}")
        assert hooks_dir.exists(), "Hooks directory should exist"

        # 6. Test hook template
        logger.info("\n6. Testing hook template")

        assert '{hook_type}' in manager.HOOK_TEMPLATE, "Template should have hook_type placeholder"
        assert 'hibro' in manager.HOOK_TEMPLATE.lower(), "Template should include hibro reference"
        logger.info("Hook template validated")

        # 7. Test install/uninstall in temp repo
        logger.info("\n7. Testing install/uninstall in temp repo")

        # Create temp git repo
        temp_dir = tempfile.mkdtemp(prefix="hibro_git_test_")
        try:
            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=temp_dir, capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=temp_dir, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=temp_dir, capture_output=True)

            temp_manager = GitHooksManager(temp_dir)
            assert temp_manager.is_git_repo(), "Temp dir should be a Git repo"

            # Install hooks
            success, installed = temp_manager.install_hooks()
            assert success, "Hook installation should succeed"
            assert len(installed) == 2, f"Should install 2 hooks, got {len(installed)}"
            logger.info(f"Installed hooks: {installed}")

            # Check hooks exist and are executable
            for hook_name in installed:
                hook_path = temp_manager.hooks_dir / hook_name
                assert hook_path.exists(), f"Hook {hook_name} should exist"
                assert os.access(hook_path, os.X_OK), f"Hook {hook_name} should be executable"

            # Get installed hooks
            installed_list = temp_manager.get_installed_hooks()
            assert len(installed_list) == 2, "Should have 2 installed hooks"

            # Uninstall hooks
            success, removed = temp_manager.uninstall_hooks()
            assert success, "Hook uninstallation should succeed"
            assert len(removed) == 2, f"Should remove 2 hooks, got {len(removed)}"
            logger.info(f"Removed hooks: {removed}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
    success = test_git_hooks()
    sys.exit(0 if success else 1)
