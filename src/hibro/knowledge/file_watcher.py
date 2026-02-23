#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File watcher for knowledge graph updates

Monitors project files for changes and triggers incremental updates
to the knowledge graph using watchdog library.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Callable, Set, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


@dataclass
class FileChangeEvent:
    """File change event information"""
    file_path: str
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: datetime = field(default_factory=datetime.now)
    old_path: Optional[str] = None  # For moved events


class FileWatcher:
    """
    File watcher for detecting project file changes

    Uses watchdog for efficient file system monitoring.
    Falls back to polling if watchdog is not available.
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

    def __init__(
        self,
        project_path: str,
        on_change: Optional[Callable[[FileChangeEvent], None]] = None
    ):
        """
        Initialize file watcher

        Args:
            project_path: Root path of project to watch
            on_change: Callback function for file changes
        """
        self.project_path = Path(project_path)
        self.on_change = on_change
        self.logger = logging.getLogger('hibro.file_watcher')

        self._observer: Optional[Any] = None
        self._running = False
        self._debounce_seconds = 1.0
        self._pending_events: Dict[str, FileChangeEvent] = {}
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        if not WATCHDOG_AVAILABLE:
            self.logger.warning("watchdog not available, file watching disabled")

    def start(self) -> bool:
        """
        Start watching for file changes

        Returns:
            True if watching started successfully
        """
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Cannot start file watcher: watchdog not installed")
            return False

        if self._running:
            self.logger.warning("File watcher already running")
            return True

        try:
            self._observer = Observer()
            handler = _WatchdogEventHandler(self)

            # Watch the project directory recursively
            self._observer.schedule(
                handler,
                str(self.project_path),
                recursive=True
            )

            self._observer.start()
            self._running = True

            self.logger.info(f"File watcher started for: {self.project_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
            return False

    def stop(self):
        """Stop watching for file changes"""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None

        self._running = False
        self.logger.info("File watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running"""
        return self._running

    def _should_monitor(self, file_path: str) -> bool:
        """
        Check if a file should be monitored

        Args:
            file_path: File path to check

        Returns:
            True if file should be monitored
        """
        path = Path(file_path)

        # Check extension
        if path.suffix.lower() not in self.MONITORED_EXTENSIONS:
            return False

        # Check excluded directories
        try:
            rel_path = path.relative_to(self.project_path)
            for part in rel_path.parts[:-1]:  # Exclude filename
                if part in self.EXCLUDED_DIRS or part.startswith('.'):
                    return False
        except ValueError:
            return False

        return True

    def _handle_event(self, event: FileSystemEvent, event_type: str):
        """
        Handle a file system event

        Args:
            event: Watchdog event
            event_type: Type of change ('created', 'modified', 'deleted')
        """
        if event.is_directory:
            return

        file_path = event.src_path

        if not self._should_monitor(file_path):
            return

        # Create change event
        change_event = FileChangeEvent(
            file_path=file_path,
            event_type=event_type
        )

        # Debounce: wait before processing to avoid duplicate events
        with self._lock:
            self._pending_events[file_path] = change_event

            if self._debounce_timer:
                self._debounce_timer.cancel()

            self._debounce_timer = threading.Timer(
                self._debounce_seconds,
                self._process_pending_events
            )
            self._debounce_timer.start()

    def _handle_moved_event(self, event: FileSystemEvent):
        """Handle file moved event"""
        if event.is_directory:
            return

        old_path = event.src_path
        new_path = event.dest_path

        # Handle as delete + create
        if self._should_monitor(old_path):
            self._handle_event(event, 'deleted')

        if self._should_monitor(new_path):
            change_event = FileChangeEvent(
                file_path=new_path,
                event_type='moved',
                old_path=old_path
            )
            with self._lock:
                self._pending_events[new_path] = change_event

                if self._debounce_timer:
                    self._debounce_timer.cancel()

                self._debounce_timer = threading.Timer(
                    self._debounce_seconds,
                    self._process_pending_events
                )
                self._debounce_timer.start()

    def _process_pending_events(self):
        """Process pending debounced events"""
        with self._lock:
            events = list(self._pending_events.values())
            self._pending_events.clear()

        self.logger.debug(f"Processing {len(events)} file change events")

        for event in events:
            self.logger.info(f"File {event.event_type}: {event.file_path}")

            if self.on_change:
                try:
                    self.on_change(event)
                except Exception as e:
                    self.logger.error(f"Error in change callback: {e}")


class _WatchdogEventHandler(FileSystemEventHandler):
    """Internal event handler for watchdog"""

    def __init__(self, watcher: FileWatcher):
        self.watcher = watcher
        super().__init__()

    def on_created(self, event):
        self.watcher._handle_event(event, 'created')

    def on_modified(self, event):
        self.watcher._handle_event(event, 'modified')

    def on_deleted(self, event):
        self.watcher._handle_event(event, 'deleted')

    def on_moved(self, event):
        self.watcher._handle_moved_event(event)


class KnowledgeGraphUpdater:
    """
    Updates knowledge graph based on file changes

    Coordinates between file watcher, parsers, and graph storage.
    """

    def __init__(
        self,
        project_path: str,
        storage: Any,  # GraphStorage
        analyzer: Any   # CodeAnalyzer
    ):
        """
        Initialize knowledge graph updater

        Args:
            project_path: Project root path
            storage: GraphStorage instance
            analyzer: CodeAnalyzer instance
        """
        self.project_path = project_path
        self.storage = storage
        self.analyzer = analyzer
        self.logger = logging.getLogger('hibro.kg_updater')

        self._watcher = FileWatcher(project_path, self._on_file_change)

    def start(self) -> bool:
        """Start watching and updating"""
        return self._watcher.start()

    def stop(self):
        """Stop watching"""
        self._watcher.stop()

    def _on_file_change(self, event: FileChangeEvent):
        """
        Handle file change event

        Args:
            event: File change event
        """
        try:
            if event.event_type == 'deleted':
                self._handle_deletion(event.file_path)
            elif event.event_type in ('created', 'modified', 'moved'):
                self._handle_update(event.file_path)
        except Exception as e:
            self.logger.error(f"Error handling file change: {e}")

    def _handle_deletion(self, file_path: str):
        """
        Handle file deletion

        Args:
            file_path: Deleted file path
        """
        self.logger.info(f"Handling deletion: {file_path}")

        # Find and delete nodes associated with this file
        rel_path = os.path.relpath(file_path, self.project_path)
        nodes = self.storage.get_nodes_by_file(rel_path, self.project_path)

        for node in nodes:
            self.storage.delete_node(node.node_id)
            self.logger.debug(f"Deleted node: {node.name}")

        self.logger.info(f"Deleted {len(nodes)} nodes for deleted file")

    def _handle_update(self, file_path: str):
        """
        Handle file creation or modification

        Args:
            file_path: Updated file path
        """
        self.logger.info(f"Handling update: {file_path}")

        # First, remove old nodes for this file
        rel_path = os.path.relpath(file_path, self.project_path)
        old_nodes = self.storage.get_nodes_by_file(rel_path, self.project_path)

        for node in old_nodes:
            self.storage.delete_node(node.node_id)

        # Parse the file and create new nodes
        ext = Path(file_path).suffix.lower()

        if ext in {'.py'}:
            self._update_python_file(file_path)
        elif ext in {'.js', '.jsx', '.ts', '.tsx'}:
            self._update_js_file(file_path)

    def _update_python_file(self, file_path: str):
        """Update nodes from Python file"""
        from ..parsers.python_parser import PythonParser

        parser = PythonParser()
        parsed = parser.parse_file(file_path)

        if parsed.syntax_errors:
            self.logger.warning(f"Skipping file with syntax errors: {file_path}")
            return

        rel_path = os.path.relpath(file_path, self.project_path)

        # Create file node
        from ..knowledge.graph_storage import GraphNode, GraphNodeType
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name=Path(file_path).name,
            file_path=rel_path,
            importance=0.5,
            project_path=self.project_path
        )
        self.storage.create_node(file_node)

        # Create class nodes
        for cls in parsed.classes:
            class_node = GraphNode(
                node_type=GraphNodeType.CLASS,
                name=cls.name,
                file_path=rel_path,
                line_number=cls.line_number,
                importance=0.8,
                project_path=self.project_path,
                metadata={
                    "methods": cls.methods,
                    "bases": cls.bases,
                    "docstring": cls.docstring[:100] if cls.docstring else None
                }
            )
            self.storage.create_node(class_node)

        # Create function nodes
        for func in parsed.functions:
            func_node = GraphNode(
                node_type=GraphNodeType.FUNCTION,
                name=func.name,
                file_path=rel_path,
                line_number=func.line_number,
                importance=0.6,
                project_path=self.project_path,
                metadata={
                    "parameters": [
                        {"name": p["name"], "type": p.get("annotation")}
                        for p in func.parameters
                    ],
                    "return_type": func.return_type,
                    "is_async": func.is_async
                }
            )
            self.storage.create_node(func_node)

        self.logger.info(
            f"Updated Python file: {len(parsed.classes)} classes, "
            f"{len(parsed.functions)} functions"
        )

    def _update_js_file(self, file_path: str):
        """Update nodes from JavaScript/TypeScript file"""
        from ..parsers.js_parser import JSParser

        parser = JSParser()
        parsed = parser.parse_file(file_path)

        rel_path = os.path.relpath(file_path, self.project_path)

        from ..knowledge.graph_storage import GraphNode, GraphNodeType

        # Create file node
        file_node = GraphNode(
            node_type=GraphNodeType.FILE,
            name=Path(file_path).name,
            file_path=rel_path,
            importance=0.5,
            project_path=self.project_path
        )
        self.storage.create_node(file_node)

        # Create class nodes
        for cls in parsed.classes:
            class_node = GraphNode(
                node_type=GraphNodeType.CLASS,
                name=cls.name,
                file_path=rel_path,
                line_number=cls.line_number,
                importance=0.8 if cls.is_component else 0.7,
                project_path=self.project_path,
                metadata={
                    "methods": cls.methods,
                    "extends": cls.extends,
                    "is_component": cls.is_component
                }
            )
            self.storage.create_node(class_node)

        # Create function nodes
        for func in parsed.functions:
            func_node = GraphNode(
                node_type=GraphNodeType.FUNCTION,
                name=func.name,
                file_path=rel_path,
                line_number=func.line_number,
                importance=0.7 if func.is_component else 0.6,
                project_path=self.project_path,
                metadata={
                    "is_async": func.is_async,
                    "is_arrow": func.is_arrow,
                    "is_component": func.is_component
                }
            )
            self.storage.create_node(func_node)

        # Create API endpoint nodes
        for endpoint in parsed.api_endpoints:
            api_node = GraphNode(
                node_type=GraphNodeType.API_ENDPOINT,
                name=f"{endpoint.method} {endpoint.path}",
                file_path=rel_path,
                line_number=endpoint.line_number,
                importance=0.8,
                project_path=self.project_path,
                metadata={
                    "method": endpoint.method,
                    "path": endpoint.path,
                    "handler": endpoint.handler
                }
            )
            self.storage.create_node(api_node)

        self.logger.info(
            f"Updated JS/TS file: {len(parsed.classes)} classes, "
            f"{len(parsed.functions)} functions, {len(parsed.api_endpoints)} endpoints"
        )
