#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup Scheduler Module
Schedules and executes periodic memory cleanup tasks
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable


class CleanupScheduler:
    """
    Memory cleanup scheduler

    Runs cleanup tasks at scheduled intervals (default: daily at 3 AM)
    """

    def __init__(self, cleaner: 'MemoryCleaner', config: dict):
        """
        Initialize cleanup scheduler

        Args:
            cleaner: MemoryCleaner instance for executing cleanup
            config: Configuration dict with keys:
                - cleanup_time_of_day: Time to run cleanup (HH:MM format)
                - cleanup_enabled: Whether scheduled cleanup is enabled
        """
        self.cleaner = cleaner
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Scheduler state
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_cleanup_time: Optional[datetime] = None
        self._cleanup_count = 0

        # Lock for thread safety
        self._lock = threading.Lock()

    def start(self):
        """Start the cleanup scheduler"""
        if self._running:
            self.logger.warning("Cleanup scheduler already running")
            return

        if not self.config.get('cleanup_enabled', True):
            self.logger.info("Scheduled cleanup is disabled")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="MemoryCleanupScheduler"
        )
        self._scheduler_thread.start()
        self.logger.info("Memory cleanup scheduler started")

    def stop(self):
        """Stop the cleanup scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.logger.info("Memory cleanup scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Calculate next cleanup time
                next_cleanup = self._get_next_cleanup_time()
                now = datetime.now()

                # Check if it's time to run cleanup
                if now >= next_cleanup:
                    self.logger.info("Starting scheduled memory cleanup")
                    self._execute_cleanup()
                    self._last_cleanup_time = now
                    self._cleanup_count += 1

                # Calculate sleep time (check every hour, or until next cleanup)
                sleep_seconds = min(3600, (next_cleanup - now).total_seconds())
                sleep_seconds = max(60, sleep_seconds)  # Minimum 1 minute

                # Sleep in short intervals to allow quick shutdown
                sleep_interval = 10  # Check every 10 seconds
                elapsed = 0
                while elapsed < sleep_seconds and self._running:
                    time.sleep(sleep_interval)
                    elapsed += sleep_interval

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes after error

    def _get_next_cleanup_time(self) -> datetime:
        """
        Calculate next cleanup time

        Returns:
            Datetime of next scheduled cleanup
        """
        cleanup_time = self.config.get('cleanup_time_of_day', '03:00')

        try:
            hour, minute = map(int, cleanup_time.split(':'))
        except (ValueError, AttributeError):
            hour, minute = 3, 0  # Default to 3:00 AM

        now = datetime.now()
        next_cleanup = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If cleanup time has passed today, schedule for tomorrow
        if next_cleanup <= now:
            next_cleanup += timedelta(days=1)

        return next_cleanup

    def _execute_cleanup(self):
        """Execute cleanup task"""
        try:
            with self._lock:
                result = self.cleaner.execute_cleanup(force=False)
            self.logger.info(
                f"Scheduled cleanup completed: deleted={result.get('deleted_count', 0)}, "
                f"duration={result.get('duration_ms', 0)}ms"
            )
        except Exception as e:
            self.logger.error(f"Cleanup execution failed: {e}")

    def trigger_immediate_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Trigger immediate cleanup (manual trigger)

        Args:
            force: Force cleanup even if recently run

        Returns:
            Cleanup result dict
        """
        with self._lock:
            result = self.cleaner.execute_cleanup(force=force)
            self._last_cleanup_time = datetime.now()
            self._cleanup_count += 1
        return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status

        Returns:
            Status information dict
        """
        return {
            "running": self._running,
            "cleanup_enabled": self.config.get('cleanup_enabled', True),
            "cleanup_time": self.config.get('cleanup_time_of_day', '03:00'),
            "last_cleanup_time": (
                self._last_cleanup_time.isoformat()
                if self._last_cleanup_time else None
            ),
            "next_cleanup_time": self._get_next_cleanup_time().isoformat(),
            "total_cleanups": self._cleanup_count
        }

    def update_config(self, config: dict):
        """
        Update scheduler configuration

        Args:
            config: New configuration dict
        """
        self.config.update(config)
        self.logger.info(f"Scheduler config updated: {config}")
