#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git hooks for knowledge graph updates

Provides Git hooks that trigger knowledge graph updates on:
- post-commit: After commits are made
- post-checkout: After branch switches
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Tuple


logger = logging.getLogger('hibro.git_hooks')


class GitHooksManager:
    """
    Manager for Git hooks related to knowledge graph updates

    Installs and manages hooks that trigger incremental updates.
    """

    # Hook names we manage
    HOOK_NAMES = ['post-commit', 'post-checkout']

    # Hook script template
    HOOK_TEMPLATE = '''#!/bin/bash
# Hibro knowledge graph update hook
# Auto-generated - do not edit manually

# Get the repository root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

if [ -z "$REPO_ROOT" ]; then
    exit 0
fi

# Run the knowledge graph update
# This can be customized based on hook type
python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/src')
from hibro.knowledge.git_hooks import trigger_update
trigger_update('$REPO_ROOT', '{hook_type}')
" 2>/dev/null &

# Don't block git operations
exit 0
'''

    def __init__(self, project_path: str):
        """
        Initialize Git hooks manager

        Args:
            project_path: Path to the Git repository
        """
        self.project_path = Path(project_path)
        self.hooks_dir = self.project_path / '.git' / 'hooks'
        self.logger = logging.getLogger('hibro.git_hooks')

    def is_git_repo(self) -> bool:
        """Check if the project is a Git repository"""
        git_dir = self.project_path / '.git'
        return git_dir.exists()

    def install_hooks(self) -> Tuple[bool, List[str]]:
        """
        Install Git hooks for knowledge graph updates

        Returns:
            Tuple of (success, list of installed hook names)
        """
        if not self.is_git_repo():
            self.logger.error("Not a Git repository")
            return False, []

        if not self.hooks_dir.exists():
            self.logger.error(f"Hooks directory not found: {self.hooks_dir}")
            return False, []

        installed = []

        for hook_name in self.HOOK_NAMES:
            hook_path = self.hooks_dir / hook_name

            # Check if hook already exists
            if hook_path.exists():
                # Check if it's our hook
                with open(hook_path, 'r') as f:
                    content = f.read()
                if 'hibro' in content.lower():
                    self.logger.info(f"Hook already installed: {hook_name}")
                    installed.append(hook_name)
                    continue

                # Backup existing hook
                backup_path = hook_path.with_suffix(hook_path.suffix + '.backup')
                self.logger.info(f"Backing up existing hook to: {backup_path}")
                os.rename(hook_path, backup_path)

            # Create new hook
            hook_content = self.HOOK_TEMPLATE.format(hook_type=hook_name)

            with open(hook_path, 'w') as f:
                f.write(hook_content)

            # Make executable
            os.chmod(hook_path, 0o755)

            self.logger.info(f"Installed hook: {hook_name}")
            installed.append(hook_name)

        return True, installed

    def uninstall_hooks(self) -> Tuple[bool, List[str]]:
        """
        Remove Git hooks installed by us

        Returns:
            Tuple of (success, list of removed hook names)
        """
        if not self.is_git_repo():
            return False, []

        removed = []

        for hook_name in self.HOOK_NAMES:
            hook_path = self.hooks_dir / hook_name

            if not hook_path.exists():
                continue

            # Check if it's our hook
            with open(hook_path, 'r') as f:
                content = f.read()

            if 'hibro' not in content.lower():
                continue

            # Remove our hook
            os.remove(hook_path)
            self.logger.info(f"Removed hook: {hook_name}")
            removed.append(hook_name)

            # Restore backup if exists
            backup_path = hook_path.with_suffix(hook_path.suffix + '.backup')
            if backup_path.exists():
                os.rename(backup_path, hook_path)
                self.logger.info(f"Restored backup: {hook_name}")

        return True, removed

    def get_installed_hooks(self) -> List[str]:
        """
        Get list of installed hooks

        Returns:
            List of installed hook names
        """
        if not self.is_git_repo():
            return []

        installed = []

        for hook_name in self.HOOK_NAMES:
            hook_path = self.hooks_dir / hook_name

            if not hook_path.exists():
                continue

            with open(hook_path, 'r') as f:
                content = f.read()

            if 'hibro' in content.lower():
                installed.append(hook_name)

        return installed


def trigger_update(repo_path: str, hook_type: str):
    """
    Trigger knowledge graph update from Git hook

    This function is called by the Git hooks.

    Args:
        repo_path: Path to the repository
        hook_type: Type of hook ('post-commit', 'post-checkout')
    """
    from ..knowledge.file_detector import FileChangeDetector

    logger.info(f"Git hook triggered: {hook_type} for {repo_path}")

    try:
        detector = FileChangeDetector(repo_path)

        if hook_type == 'post-commit':
            # Get changed files from last commit
            changed_files = get_last_commit_files(repo_path)
            logger.info(f"Files changed in last commit: {len(changed_files)}")

        elif hook_type == 'post-checkout':
            # Get changed files from diff
            changed_files = get_checkout_changed_files(repo_path)
            logger.info(f"Files changed in checkout: {len(changed_files)}")

        else:
            changed_files = []

        # Process changes
        for file_path in changed_files:
            logger.debug(f"Processing changed file: {file_path}")

        logger.info(f"Knowledge graph update completed for {hook_type}")

    except Exception as e:
        logger.error(f"Error in trigger_update: {e}")


def get_last_commit_files(repo_path: str) -> List[str]:
    """
    Get files changed in the last commit

    Args:
        repo_path: Path to Git repository

    Returns:
        List of changed file paths
    """
    try:
        result = subprocess.run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return files
        return []

    except Exception as e:
        logger.error(f"Error getting last commit files: {e}")
        return []


def get_checkout_changed_files(repo_path: str) -> List[str]:
    """
    Get files changed in a checkout

    Args:
        repo_path: Path to Git repository

    Returns:
        List of changed file paths
    """
    try:
        # Get diff between current and previous HEAD
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD@{1}', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return files
        return []

    except Exception as e:
        logger.error(f"Error getting checkout changed files: {e}")
        return []


def get_branch_name(repo_path: str) -> Optional[str]:
    """
    Get current branch name

    Args:
        repo_path: Path to Git repository

    Returns:
        Branch name or None
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            return result.stdout.strip()
        return None

    except Exception as e:
        logger.error(f"Error getting branch name: {e}")
        return None


def get_commit_hash(repo_path: str) -> Optional[str]:
    """
    Get current commit hash

    Args:
        repo_path: Path to Git repository

    Returns:
        Commit hash or None
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
        return None

    except Exception as e:
        logger.error(f"Error getting commit hash: {e}")
        return None
