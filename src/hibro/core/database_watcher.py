#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Watcher
Monitor SQLite database file changes, notify MemoryEngine to refresh cache
Implement real-time synchronization between multiple IDE dialog windows
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


class DatabaseFileHandler(FileSystemEventHandler):
    """Database file change handler"""

    def __init__(self, db_path: str, callback: Callable, debounce_seconds: float = 1.0):
        """
        Initialize database file handler

        Args:
            db_path: Database file path
            callback: Change callback function
            debounce_seconds: Debounce time (seconds)
        """
        super().__init__()
        self.db_path = str(Path(db_path).absolute())
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.logger = logging.getLogger('hibro.database_watcher')

        # Debounce mechanism
        self._last_trigger_time = 0
        self._debounce_timer = None
        self._lock = threading.Lock()

    def on_modified(self, event):
        """File modification event handler"""
        if event.is_directory:
            return

        # Only monitor database file
        if str(Path(event.src_path).absolute()) == self.db_path:
            self._trigger_callback()

    def on_created(self, event):
        """File creation event handler"""
        if event.is_directory:
            return

        # Database was rebuilt
        if str(Path(event.src_path).absolute()) == self.db_path:
            self._trigger_callback()

    def _trigger_callback(self):
        """Trigger callback (with debounce)"""
        current_time = time.time()

        with self._lock:
            # If time since last trigger is too short, ignore
            if current_time - self._last_trigger_time < self.debounce_seconds:
                self.logger.debug("Database change event ignored by debounce")
                return

            self._last_trigger_time = current_time

        # Execute callback in new thread to avoid blocking file monitoring
        threading.Thread(
            target=self._execute_callback,
            daemon=True
        ).start()

    def _execute_callback(self):
        """Execute callback function"""
        try:
            self.logger.info("Database change detected, triggering cache refresh")
            self.callback()
        except Exception as e:
            self.logger.error(f"Execute database change callback failed: {e}")


class DatabaseWatcher:
    """Database Watcher"""

    def __init__(self, db_path: str, on_change_callback: Callable, debounce_seconds: float = 1.0):
        """
        Initialize database watcher

        Args:
            db_path: Database file path
            on_change_callback: Callback function when database changes
            debounce_seconds: Debounce time (seconds)
        """
        self.db_path = str(Path(db_path).absolute())
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds
        self.logger = logging.getLogger('hibro.database_watcher')

        # Listener components
        self.observer = None
        self.event_handler = None
        self.is_watching = False

        # Statistics
        self.change_count = 0
        self.last_change_time = None

    def start(self):
        """Start monitoring"""
        if self.is_watching:
            self.logger.warning("Database watcher is already running")
            return

        try:
            # Check if database file exists
            if not Path(self.db_path).exists():
                self.logger.warning(f"Database file does not exist: {self.db_path}")
                # Create parent directory
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Create event handler
            self.event_handler = DatabaseFileHandler(
                self.db_path,
                self._on_database_changed,
                self.debounce_seconds
            )

            # Create file listener
            self.observer = Observer()

            # Monitor database directory (because SQLite rebuilds files)
            watch_path = str(Path(self.db_path).parent)
            self.observer.schedule(
                self.event_handler,
                watch_path,
                recursive=False
            )

            # Start monitoring
            self.observer.start()
            self.is_watching = True

            self.logger.info(f"Database watcher started: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Start database watcher failed: {e}")
            raise

    def stop(self):
        """Stop monitoring"""
        if not self.is_watching:
            return

        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
                self.observer = None

            self.is_watching = False
            self.logger.info("Database watcher stopped")

        except Exception as e:
            self.logger.error(f"Stop database watcher failed: {e}")

    def _on_database_changed(self):
        """Database change handler"""
        self.change_count += 1
        self.last_change_time = time.time()

        self.logger.info(
            f"Database change detected (total: {self.change_count} times) - "
            f"triggering cache refresh"
        )

        # Execute callback
        if self.on_change_callback:
            try:
                self.on_change_callback()
            except Exception as e:
                self.logger.error(f"Execute database change callback failed: {e}")

    def get_stats(self) -> dict:
        """Get statistics"""
        return {
            'is_watching': self.is_watching,
            'db_path': self.db_path,
            'change_count': self.change_count,
            'last_change_time': self.last_change_time,
            'debounce_seconds': self.debounce_seconds
        }

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False
