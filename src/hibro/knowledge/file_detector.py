#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File change detector

Detects project file changes to trigger incremental knowledge graph updates
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class FileChange:
    """
    File change information
    """
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted'
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_mtime: Optional[datetime] = None
    new_mtime: Optional[datetime] = None


class FileChangeDetector:
    """
    File change detector
    """

    # File extensions to monitor
    MONITORED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.go', '.rs', '.cpp', '.c', '.h',
        '.rb', '.php', '.swift', '.kt'
    }

    # Directories to exclude
    EXCLUDED_DIRS = {
        'node_modules', '__pycache__', '.git', '.venv', 'venv',
        'dist', 'build', '.next', '.nuxt', 'coverage',
        '.pytest_cache', 'migrations', 'logs', 'cache', 'tmp', 'temp'
    }

    def __init__(self, project_path: str):
        """
        Initialize file change detector

        Args:
            project_path: Project path
        """
        self.project_path = Path(project_path)
        self.logger = logging.getLogger('hibro.file_detector')

        # File state cache: {file_path: (mtime, hash)}
        self.file_states: Dict[str, Tuple[datetime, str]] = {}

    def scan_project_files(self) -> List[str]:
        """
        Scan all source code files in project

        Returns:
            List of file paths
        """
        files = []

        try:
            for root, dirs, filenames in os.walk(self.project_path):
                # Filter excluded directories (modify dirs list in-place)
                dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS and not d.startswith('.')]

                for filename in filenames:
                    # Check file extension
                    ext = Path(filename).suffix.lower()
                    if ext in self.MONITORED_EXTENSIONS:
                        file_path = os.path.join(root, filename)
                        # Convert to relative path
                        rel_path = os.path.relpath(file_path, self.project_path)

                        # Double check: ensure no excluded directories in path
                        path_parts = Path(rel_path).parts
                        if not any(part in self.EXCLUDED_DIRS or part.startswith('.') for part in path_parts[:-1]):
                            files.append(rel_path)

            self.logger.info(f"Scanned {len(files)} source code files")
            return files

        except Exception as e:
            self.logger.error(f"Scan project files failed: {e}")
            return []

    def get_file_mtime(self, file_path: str) -> Optional[datetime]:
        """
        Get file modification time

        Args:
            file_path: File path (relative)

        Returns:
            Modification time, None if not exists
        """
        try:
            full_path = self.project_path / file_path
            if full_path.exists():
                mtime = full_path.stat().st_mtime
                return datetime.fromtimestamp(mtime)
            return None
        except Exception as e:
            self.logger.warning(f"Get file mtime failed {file_path}: {e}")
            return None

    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate file content hash

        Args:
            file_path: File path (relative)

        Returns:
            SHA256 hash, None if not exists
        """
        try:
            full_path = self.project_path / file_path
            if not full_path.exists():
                return None

            with open(full_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()

        except Exception as e:
            self.logger.warning(f"Calculate file hash failed {file_path}: {e}")
            return None

    def initialize_file_states(self, files: Optional[List[str]] = None):
        """
        Initialize file state cache

        Args:
            files: File list, None means scan all
        """
        if files is None:
            files = self.scan_project_files()

        self.logger.info(f"Initializing state for {len(files)} files")

        for file_path in files:
            mtime = self.get_file_mtime(file_path)
            file_hash = self.calculate_file_hash(file_path)

            if mtime and file_hash:
                self.file_states[file_path] = (mtime, file_hash)

        self.logger.info(f"File state initialized, cached {len(self.file_states)} files")

    def detect_changes(self, files: Optional[List[str]] = None) -> List[FileChange]:
        """
        Detect file changes

        Args:
            files: Files to check, None means all

        Returns:
            List of file changes
        """
        if files is None:
            files = self.scan_project_files()

        changes = []
        current_files = set(files)
        cached_files = set(self.file_states.keys())

        # Detect added and modified files
        for file_path in current_files:
            current_mtime = self.get_file_mtime(file_path)
            current_hash = self.calculate_file_hash(file_path)

            if not current_mtime or not current_hash:
                continue

            if file_path not in self.file_states:
                # Added file
                changes.append(FileChange(
                    file_path=file_path,
                    change_type='added',
                    new_hash=current_hash,
                    new_mtime=current_mtime
                ))
                self.file_states[file_path] = (current_mtime, current_hash)

            else:
                # Check if modified
                old_mtime, old_hash = self.file_states[file_path]

                # Check mtime first, then hash if time changed
                if current_mtime > old_mtime:
                    if current_hash != old_hash:
                        # Content actually changed
                        changes.append(FileChange(
                            file_path=file_path,
                            change_type='modified',
                            old_hash=old_hash,
                            new_hash=current_hash,
                            old_mtime=old_mtime,
                            new_mtime=current_mtime
                        ))
                        self.file_states[file_path] = (current_mtime, current_hash)

        # Detect deleted files
        for file_path in cached_files - current_files:
            old_mtime, old_hash = self.file_states[file_path]
            changes.append(FileChange(
                file_path=file_path,
                change_type='deleted',
                old_hash=old_hash,
                old_mtime=old_mtime
            ))
            del self.file_states[file_path]

        if changes:
            self.logger.info(f"Detected {len(changes)} file changes")
            for change in changes:
                self.logger.debug(f"  {change.change_type}: {change.file_path}")
        else:
            self.logger.debug("No file changes detected")

        return changes

    def detect_changes_by_mtime(
        self,
        since: datetime,
        files: Optional[List[str]] = None
    ) -> List[str]:
        """
        Detect changed files by modification time

        Args:
            since: Start time
            files: Files to check, None means all

        Returns:
            List of changed file paths
        """
        if files is None:
            files = self.scan_project_files()

        changed_files = []

        for file_path in files:
            mtime = self.get_file_mtime(file_path)
            if mtime and mtime > since:
                changed_files.append(file_path)

        self.logger.info(f"Since {since}, {len(changed_files)} files modified")

        return changed_files

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get file information

        Args:
            file_path: File path (relative)

        Returns:
            File information dict
        """
        try:
            full_path = self.project_path / file_path
            if not full_path.exists():
                return None

            stat = full_path.stat()
            file_hash = self.calculate_file_hash(file_path)

            return {
                'file_path': file_path,
                'size': stat.st_size,
                'mtime': datetime.fromtimestamp(stat.st_mtime),
                'hash': file_hash,
                'extension': full_path.suffix.lower()
            }

        except Exception as e:
            self.logger.error(f"Get file info failed {file_path}: {e}")
            return None

    def clear_cache(self):
        """
        Clear file state cache
        """
        self.file_states.clear()
        self.logger.info("File state cache cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Statistics dict
        """
        return {
            'cached_files': len(self.file_states),
            'project_path': str(self.project_path)
        }
