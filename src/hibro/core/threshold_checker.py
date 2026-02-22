#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold Checker Module
Checks memory usage thresholds before storage operations
"""

import logging
from typing import Tuple, Dict, Any

from ..utils.constants import (
    MEMORY_THRESHOLD_WARNING,
    MEMORY_THRESHOLD_CLEANUP,
    MEMORY_THRESHOLD_CRITICAL
)


class ThresholdChecker:
    """
    Memory threshold checker

    Checks memory usage before storage operations and triggers
    cleanup when thresholds are exceeded.

    Threshold levels:
    - Warning (70%): Log warning, allow storage
    - Cleanup (85%): Trigger async cleanup, allow storage
    - Critical (95%): Block storage until cleanup completes
    """

    def __init__(self, memory_repo, cleaner: 'MemoryCleaner', max_memories: int):
        """
        Initialize threshold checker

        Args:
            memory_repo: Memory repository for count queries
            cleaner: MemoryCleaner instance for triggering cleanup
            max_memories: Maximum number of memories allowed
        """
        self.memory_repo = memory_repo
        self.cleaner = cleaner
        self.max_memories = max_memories
        self.logger = logging.getLogger(__name__)

        # Track recent checks to avoid spam
        self._last_warning_time = None
        self._check_count = 0

    def check_before_store(self) -> Tuple[bool, str]:
        """
        Check if storage is allowed based on current usage

        Should be called before each memory storage operation.

        Returns:
            Tuple of (allow_store, message):
            - allow_store: Whether storage is allowed
            - message: Status message or warning (empty if normal)
        """
        self._check_count += 1

        try:
            current_count = self._get_memory_count()
            ratio = current_count / self.max_memories if self.max_memories > 0 else 0

            if ratio >= MEMORY_THRESHOLD_CRITICAL:
                # Critical: Force cleanup and block storage
                self.logger.warning(
                    f"Critical memory threshold reached: {ratio:.1%} "
                    f"({current_count}/{self.max_memories})"
                )

                # Force synchronous cleanup
                self.cleaner.execute_cleanup(force=True)

                # Re-check after cleanup
                new_count = self._get_memory_count()
                new_ratio = new_count / self.max_memories if self.max_memories > 0 else 0

                if new_ratio >= MEMORY_THRESHOLD_CRITICAL:
                    # Still critical after cleanup - block storage
                    return False, (
                        f"Memory capacity critical ({new_ratio:.1%}), "
                        "storage temporarily blocked. Please free up memory manually."
                    )

                return True, f"Memory cleaned, now at {new_ratio:.1%} capacity"

            elif ratio >= MEMORY_THRESHOLD_CLEANUP:
                # Cleanup threshold: Trigger async cleanup, allow storage
                self.logger.info(
                    f"Memory threshold reached: {ratio:.1%}, triggering cleanup"
                )
                self.cleaner.execute_cleanup_async()
                return True, f"Memory at {ratio:.1%}, cleanup triggered"

            elif ratio >= MEMORY_THRESHOLD_WARNING:
                # Warning threshold: Log warning, allow storage
                self.logger.info(f"Memory usage warning: {ratio:.1%}")
                return True, f"Warning: Memory at {ratio:.1%} capacity"

            # Normal operation
            return True, ""

        except Exception as e:
            self.logger.error(f"Threshold check failed: {e}")
            # Allow storage on error to avoid blocking operations
            return True, f"Warning: Threshold check failed ({e})"

    def _get_memory_count(self) -> int:
        """
        Get current memory count

        Returns:
            Number of memories in storage
        """
        try:
            if hasattr(self.memory_repo, 'get_memory_count'):
                return self.memory_repo.get_memory_count()
            elif hasattr(self.memory_repo, 'count'):
                return self.memory_repo.count()
            else:
                # Fallback: count from search
                memories = self.memory_repo.search_memories(query="", limit=1)
                # This is approximate, but better than nothing
                return len(memories) if memories else 0
        except Exception as e:
            self.logger.error(f"Failed to get memory count: {e}")
            return 0

    def get_usage_status(self) -> Dict[str, Any]:
        """
        Get current memory usage status

        Returns:
            Usage status dict with:
            - current_count: Current number of memories
            - max_memories: Maximum allowed memories
            - usage_ratio: Current usage ratio (0.0 - 1.0)
            - status: Current status (normal/warning/cleanup_needed/critical)
            - can_store: Whether storage is allowed
            - thresholds: Configured threshold values
        """
        current_count = self._get_memory_count()
        ratio = current_count / self.max_memories if self.max_memories > 0 else 0

        # Determine status
        status = "normal"
        can_store = True

        if ratio >= MEMORY_THRESHOLD_CRITICAL:
            status = "critical"
            can_store = False
        elif ratio >= MEMORY_THRESHOLD_CLEANUP:
            status = "cleanup_needed"
        elif ratio >= MEMORY_THRESHOLD_WARNING:
            status = "warning"

        return {
            "current_count": current_count,
            "max_memories": self.max_memories,
            "usage_ratio": round(ratio, 4),
            "usage_percent": f"{ratio:.1%}",
            "status": status,
            "can_store": can_store,
            "thresholds": {
                "warning": MEMORY_THRESHOLD_WARNING,
                "cleanup": MEMORY_THRESHOLD_CLEANUP,
                "critical": MEMORY_THRESHOLD_CRITICAL
            },
            "check_count": self._check_count
        }

    def update_max_memories(self, max_memories: int):
        """
        Update maximum memories limit

        Args:
            max_memories: New maximum memory count
        """
        if max_memories > 0:
            self.max_memories = max_memories
            self.logger.info(f"Max memories updated to: {max_memories}")
