#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session update logic for knowledge graph

Provides logic for checking and updating knowledge graph at session start:
- Detect if updates are needed
- Perform incremental updates
- Queue management for concurrent updates
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from .file_detector import FileChangeDetector, FileChange
from .graph_storage import GraphStorage, GraphNode, GraphNodeType


@dataclass
class UpdateTask:
    """Task for knowledge graph update"""
    task_id: str
    file_path: str
    task_type: str  # 'add', 'update', 'delete'
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = 'pending'  # 'pending', 'processing', 'completed', 'failed'
    error: Optional[str] = None


@dataclass
class SessionCheckResult:
    """Result of session update check"""
    needs_update: bool
    last_scan_time: Optional[datetime] = None
    files_changed: int = 0
    reason: str = ""
    changes: List[FileChange] = field(default_factory=list)


class SessionUpdateManager:
    """
    Manager for session-based knowledge graph updates

    Handles:
    - Checking if updates are needed at session start
    - Performing incremental updates efficiently
    - Managing update queue for concurrent operations
    """

    # Metadata file name
    METADATA_FILE = ".hibro_kg_metadata.json"

    def __init__(
        self,
        project_path: str,
        storage: GraphStorage,
        max_workers: int = 2
    ):
        """
        Initialize session update manager

        Args:
            project_path: Project root path
            storage: GraphStorage instance
            max_workers: Maximum concurrent update workers
        """
        self.project_path = Path(project_path)
        self.storage = storage
        self.detector = FileChangeDetector(project_path)

        self.logger = logging.getLogger('hibro.session_update')

        self._update_queue: Queue[UpdateTask] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._running = False

        # Metadata
        self._metadata_path = self.project_path / self.METADATA_FILE
        self._metadata: Dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load session metadata"""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
                self.logger.debug(f"Loaded metadata from {self._metadata_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
                self._metadata = {}

    def _save_metadata(self):
        """Save session metadata"""
        try:
            # Ensure parent directory exists
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved metadata to {self._metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def check_needs_update(self, force: bool = False) -> SessionCheckResult:
        """
        Check if knowledge graph needs update

        Args:
            force: Force check even if recently updated

        Returns:
            SessionCheckResult with update information
        """
        result = SessionCheckResult(needs_update=False)

        # Get last scan time from metadata
        last_scan_str = self._metadata.get('last_scan_time')
        if last_scan_str:
            try:
                result.last_scan_time = datetime.fromisoformat(last_scan_str)
            except:
                pass

        # Check if never scanned
        if not result.last_scan_time:
            result.needs_update = True
            result.reason = "Never scanned"
            return result

        # Check if scan is too old (older than 1 day)
        scan_age = datetime.now() - result.last_scan_time
        if scan_age > timedelta(days=1):
            result.needs_update = True
            result.reason = f"Last scan too old ({scan_age.days} days)"
            return result

        # Check for file changes
        if not force and scan_age < timedelta(minutes=5):
            # Skip check if recently updated (< 5 min)
            result.needs_update = False
            result.reason = "Recently updated"
            return result

        # Detect file changes
        changes = self.detector.detect_changes()
        result.changes = changes
        result.files_changed = len(changes)

        if changes:
            result.needs_update = True
            result.reason = f"{len(changes)} files changed"
        else:
            result.reason = "No changes detected"

        return result

    def perform_full_scan(self) -> Dict[str, int]:
        """
        Perform full project scan and build knowledge graph

        Returns:
            Statistics about scan results
        """
        from ..parsers.code_analyzer import CodeAnalyzer

        self.logger.info(f"Starting full scan of {self.project_path}")

        stats = {
            'files_processed': 0,
            'classes_added': 0,
            'functions_added': 0,
            'api_endpoints_added': 0,
            'errors': 0
        }

        try:
            # Initialize file states
            files = self.detector.scan_project_files()
            self.detector.initialize_file_states(files)

            # Use code analyzer to parse files
            analyzer = CodeAnalyzer()

            for file_path in files:
                try:
                    self._process_file(file_path, stats)
                    stats['files_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    stats['errors'] += 1

            # Update metadata
            self._metadata['last_scan_time'] = datetime.now().isoformat()
            self._metadata['last_scan_stats'] = stats
            self._metadata['file_count'] = len(files)
            self._save_metadata()

            self.logger.info(f"Full scan complete: {stats}")

        except Exception as e:
            self.logger.error(f"Full scan failed: {e}")
            stats['errors'] += 1

        return stats

    def _process_file(self, file_path: str, stats: Dict[str, int]):
        """
        Process a single file and update knowledge graph

        Args:
            file_path: File path relative to project
            stats: Statistics dictionary to update
        """
        from ..parsers.python_parser import PythonParser
        from ..parsers.js_parser import JSParser

        ext = Path(file_path).suffix.lower()

        if ext == '.py':
            self._process_python_file(file_path, stats)
        elif ext in {'.js', '.jsx', '.ts', '.tsx'}:
            self._process_js_file(file_path, stats)

    def _process_python_file(self, file_path: str, stats: Dict[str, int]):
        """Process Python file"""
        from ..parsers.python_parser import PythonParser

        parser = PythonParser()
        full_path = self.project_path / file_path
        parsed = parser.parse_file(str(full_path))

        if parsed.syntax_errors:
            return

        # Create file node
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name=Path(file_path).name,
            file_path=file_path,
            importance=0.5,
            project_path=str(self.project_path)
        )
        self.storage.create_node(file_node)

        # Create class nodes
        for cls in parsed.classes:
            class_node = GraphNode(
                node_type=GraphNodeType.CLASS,
                name=cls.name,
                file_path=file_path,
                line_number=cls.line_number,
                importance=0.8,
                project_path=str(self.project_path),
                metadata={
                    "methods": cls.methods,
                    "bases": cls.bases
                }
            )
            self.storage.create_node(class_node)
            stats['classes_added'] += 1

        # Create function nodes
        for func in parsed.functions:
            func_node = GraphNode(
                node_type=GraphNodeType.FUNCTION,
                name=func.name,
                file_path=file_path,
                line_number=func.line_number,
                importance=0.6,
                project_path=str(self.project_path),
                metadata={
                    "is_async": func.is_async,
                    "return_type": func.return_type
                }
            )
            self.storage.create_node(func_node)
            stats['functions_added'] += 1

    def _process_js_file(self, file_path: str, stats: Dict[str, int]):
        """Process JavaScript/TypeScript file"""
        from ..parsers.js_parser import JSParser

        parser = JSParser()
        full_path = self.project_path / file_path
        parsed = parser.parse_file(str(full_path))

        # Create file node
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name=Path(file_path).name,
            file_path=file_path,
            importance=0.5,
            project_path=str(self.project_path)
        )
        self.storage.create_node(file_node)

        # Create class nodes
        for cls in parsed.classes:
            class_node = GraphNode(
                node_type=GraphNodeType.CLASS,
                name=cls.name,
                file_path=file_path,
                line_number=cls.line_number,
                importance=0.8 if cls.is_component else 0.7,
                project_path=str(self.project_path),
                metadata={
                    "methods": cls.methods,
                    "extends": cls.extends,
                    "is_component": cls.is_component
                }
            )
            self.storage.create_node(class_node)
            stats['classes_added'] += 1

        # Create function nodes
        for func in parsed.functions:
            func_node = GraphNode(
                node_type=GraphNodeType.FUNCTION,
                name=func.name,
                file_path=file_path,
                line_number=func.line_number,
                importance=0.7 if func.is_component else 0.6,
                project_path=str(self.project_path),
                metadata={
                    "is_async": func.is_async,
                    "is_arrow": func.is_arrow
                }
            )
            self.storage.create_node(func_node)
            stats['functions_added'] += 1

        # Create API endpoint nodes
        for endpoint in parsed.api_endpoints:
            api_node = GraphNode(
                node_type=GraphNodeType.API_ENDPOINT,
                name=f"{endpoint.method} {endpoint.path}",
                file_path=file_path,
                line_number=endpoint.line_number,
                importance=0.8,
                project_path=str(self.project_path),
                metadata={
                    "method": endpoint.method,
                    "path": endpoint.path
                }
            )
            self.storage.create_node(api_node)
            stats['api_endpoints_added'] += 1

    def perform_incremental_update(self, changes: List[FileChange]) -> Dict[str, int]:
        """
        Perform incremental update based on file changes

        Args:
            changes: List of file changes

        Returns:
            Update statistics
        """
        stats = {
            'files_updated': 0,
            'files_deleted': 0,
            'nodes_removed': 0,
            'nodes_added': 0,
            'errors': 0
        }

        for change in changes:
            try:
                if change.change_type == 'deleted':
                    self._handle_file_deletion(change.file_path, stats)
                    stats['files_deleted'] += 1
                else:  # 'added' or 'modified'
                    self._handle_file_update(change.file_path, stats)
                    stats['files_updated'] += 1
            except Exception as e:
                self.logger.error(f"Error updating {change.file_path}: {e}")
                stats['errors'] += 1

        # Update metadata
        self._metadata['last_update_time'] = datetime.now().isoformat()
        self._save_metadata()

        return stats

    def _handle_file_deletion(self, file_path: str, stats: Dict[str, int]):
        """Handle deleted file"""
        nodes = self.storage.get_nodes_by_file(file_path, str(self.project_path))
        for node in nodes:
            self.storage.delete_node(node.node_id)
            stats['nodes_removed'] += 1

    def _handle_file_update(self, file_path: str, stats: Dict[str, int]):
        """Handle updated or new file"""
        # Remove old nodes
        old_nodes = self.storage.get_nodes_by_file(file_path, str(self.project_path))
        for node in old_nodes:
            self.storage.delete_node(node.node_id)
            stats['nodes_removed'] += 1

        # Process new content
        self._process_file(file_path, stats)
        stats['nodes_added'] = stats.get('classes_added', 0) + stats.get('functions_added', 0)

    def queue_update(self, file_path: str, task_type: str, priority: int = 0):
        """
        Queue a file update task

        Args:
            file_path: File to update
            task_type: Type of task ('add', 'update', 'delete')
            priority: Task priority (higher = more urgent)
        """
        task = UpdateTask(
            task_id=f"{datetime.now().timestamp()}_{file_path}",
            file_path=file_path,
            task_type=task_type,
            priority=priority
        )
        self._update_queue.put(task)
        self.logger.debug(f"Queued update task: {task_type} {file_path}")

    def start_queue_processor(self):
        """Start processing queued updates"""
        if self._running:
            return

        self._running = True
        self._executor.submit(self._process_queue)

    def stop_queue_processor(self):
        """Stop processing queued updates"""
        self._running = False
        self._executor.shutdown(wait=False)

    def _process_queue(self):
        """Process queued update tasks"""
        while self._running:
            try:
                task = self._update_queue.get(timeout=1.0)

                # Mark as processing
                task.status = 'processing'

                try:
                    if task.task_type == 'delete':
                        self._handle_file_deletion(task.file_path, {})
                    else:
                        self._handle_file_update(task.file_path, {})

                    task.status = 'completed'

                except Exception as e:
                    task.status = 'failed'
                    task.error = str(e)
                    self.logger.error(f"Task failed: {e}")

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
