#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Cleaner Module
Executes memory cleanup using triple eviction strategies:
1. LFU (Least Frequently Used) eviction
2. Time-based expiry
3. Importance-based eviction
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from threading import Thread

from ..utils.constants import (
    LFU_CLEANUP_BOTTOM_PERCENT,
    TIME_EXPIRY_DAYS_DEFAULT,
    TIME_EXPIRY_DAYS_LOW_IMPORTANCE,
    IMPORTANCE_THRESHOLD_CLEANUP,
    NEW_MEMORY_PROTECTION_DAYS
)


class MemoryCleaner:
    """
    Memory cleaner with triple eviction strategy

    Strategies:
    1. LFU eviction: Remove bottom N% of LFU-ranked memories
    2. Time expiry: Remove memories not accessed for extended period
    3. Importance eviction: Remove low-importance memories

    Protected memories (never cleaned unless forced):
    - Active tasks
    - Project initialization records
    - User preferences
    - Recently created memories (within protection period)
    """

    # Categories that should never be cleaned (unless forced)
    PROTECTED_CATEGORIES = {"active_task", "project_init", "preference"}

    def __init__(self, memory_engine, lfu_calculator):
        """
        Initialize memory cleaner

        Args:
            memory_engine: MemoryEngine instance
            lfu_calculator: LFUCalculator instance for scoring
        """
        self.memory_engine = memory_engine
        self.memory_repo = memory_engine.memory_repo
        self.lfu_calculator = lfu_calculator
        self.logger = logging.getLogger(__name__)

        # Statistics from last cleanup
        self._last_cleanup_stats: Optional[Dict] = None

        # Lock for thread safety
        self._cleanup_lock = None  # Will be set externally if needed

    def execute_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute memory cleanup (synchronous)

        Args:
            force: Force cleanup, ignoring protection rules

        Returns:
            {
                "success": bool,
                "deleted_count": int,
                "strategies": {
                    "lfu": int,
                    "time_expiry": int,
                    "importance": int
                },
                "duration_ms": int,
                "error": str (if failed)
            }
        """
        start_time = datetime.now()
        result = {
            "success": True,
            "deleted_count": 0,
            "strategies": {
                "lfu": 0,
                "time_expiry": 0,
                "importance": 0
            },
            "duration_ms": 0,
            "forced": force
        }

        self.logger.info(f"Starting memory cleanup (force={force})")

        try:
            # Strategy 1: LFU eviction - clean least frequently used
            lfu_deleted = self._cleanup_by_lfu(force)
            result["strategies"]["lfu"] = lfu_deleted
            result["deleted_count"] += lfu_deleted
            self.logger.info(f"LFU eviction: {lfu_deleted} memories deleted")

            # Strategy 2: Time expiry - clean old unused memories
            time_deleted = self._cleanup_by_time_expiry(force)
            result["strategies"]["time_expiry"] = time_deleted
            result["deleted_count"] += time_deleted
            self.logger.info(f"Time expiry: {time_deleted} memories deleted")

            # Strategy 3: Importance eviction - clean low-value memories
            importance_deleted = self._cleanup_by_importance(force)
            result["strategies"]["importance"] = importance_deleted
            result["deleted_count"] += importance_deleted
            self.logger.info(f"Importance eviction: {importance_deleted} memories deleted")

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.logger.error(f"Cleanup failed: {e}")

        # Calculate duration
        result["duration_ms"] = int((datetime.now() - start_time).total_seconds() * 1000)

        # Store stats
        self._last_cleanup_stats = result

        self.logger.info(
            f"Cleanup completed: total_deleted={result['deleted_count']}, "
            f"lfu={result['strategies']['lfu']}, "
            f"time={result['strategies']['time_expiry']}, "
            f"importance={result['strategies']['importance']}, "
            f"duration={result['duration_ms']}ms"
        )

        return result

    def execute_cleanup_async(self):
        """
        Execute cleanup asynchronously (non-blocking)

        Creates a daemon thread to run cleanup in background.
        """
        thread = Thread(
            target=self.execute_cleanup,
            kwargs={"force": False},
            daemon=True,
            name="AsyncMemoryCleanup"
        )
        thread.start()
        self.logger.info("Async cleanup started")

    def _cleanup_by_lfu(self, force: bool) -> int:
        """
        Strategy 1: LFU eviction

        Removes memories with lowest LFU scores (bottom N%).

        Args:
            force: Ignore protection rules

        Returns:
            Number of memories deleted
        """
        try:
            all_memories = self._get_all_memories()
            if not all_memories:
                return 0

            # Calculate LFU score for each memory
            now = datetime.now()
            lfu_scores = []

            for memory in all_memories:
                try:
                    score = self.lfu_calculator.lfu_decay(memory, now)
                    lfu_scores.append((memory, score))
                except Exception as e:
                    self.logger.warning(f"LFU calculation failed for memory: {e}")
                    # Default to low score
                    lfu_scores.append((memory, 0.0))

            # Sort by score (ascending - lowest first)
            lfu_scores.sort(key=lambda x: x[1])

            # Calculate number to clean (bottom N%)
            cleanup_count = int(len(all_memories) * LFU_CLEANUP_BOTTOM_PERCENT)
            if cleanup_count < 1:
                cleanup_count = 1  # At least clean 1 if triggered

            deleted = 0
            for memory, score in lfu_scores:
                if deleted >= cleanup_count:
                    break

                # Check protection (unless forced)
                if not force and self._is_protected(memory):
                    continue

                # Delete memory
                if self._delete_memory(memory):
                    deleted += 1

            return deleted

        except Exception as e:
            self.logger.error(f"LFU cleanup failed: {e}")
            return 0

    def _cleanup_by_time_expiry(self, force: bool) -> int:
        """
        Strategy 2: Time-based expiry

        Removes memories not accessed for extended period.
        Retention period varies by importance:
        - High importance (>=0.3): TIME_EXPIRY_DAYS_DEFAULT (365 days)
        - Low importance (<0.3): TIME_EXPIRY_DAYS_LOW_IMPORTANCE (90 days)

        Args:
            force: Ignore protection rules

        Returns:
            Number of memories deleted
        """
        deleted = 0
        now = datetime.now()

        try:
            all_memories = self._get_all_memories()

            for memory in all_memories:
                if not memory.last_accessed:
                    continue

                # Determine retention period based on importance
                if memory.importance < 0.3:
                    expiry_days = TIME_EXPIRY_DAYS_LOW_IMPORTANCE
                else:
                    expiry_days = TIME_EXPIRY_DAYS_DEFAULT

                expiry_date = now - timedelta(days=expiry_days)

                # Check if memory has expired
                if memory.last_accessed < expiry_date:
                    # Check protection
                    if not force and self._is_protected(memory):
                        continue

                    if self._delete_memory(memory):
                        deleted += 1

            return deleted

        except Exception as e:
            self.logger.error(f"Time expiry cleanup failed: {e}")
            return 0

    def _cleanup_by_importance(self, force: bool) -> int:
        """
        Strategy 3: Importance-based eviction

        Removes memories with importance below threshold.
        Gives new memories a grace period to prove value.

        Args:
            force: Ignore protection rules

        Returns:
            Number of memories deleted
        """
        deleted = 0
        now = datetime.now()

        try:
            # Search for low importance memories
            low_importance_memories = self._search_memories_by_importance(
                max_importance=IMPORTANCE_THRESHOLD_CLEANUP,
                limit=1000
            )

            for memory in low_importance_memories:
                # Check if memory is too new (grace period)
                if memory.created_at:
                    age_days = (now - memory.created_at).days
                    if age_days < NEW_MEMORY_PROTECTION_DAYS:
                        continue  # Skip new memories

                # Check protection
                if not force and self._is_protected(memory):
                    continue

                if self._delete_memory(memory):
                    deleted += 1

            return deleted

        except Exception as e:
            self.logger.error(f"Importance cleanup failed: {e}")
            return 0

    def _get_all_memories(self) -> List:
        """Get all memories from repository"""
        try:
            if hasattr(self.memory_repo, 'get_all_memories'):
                return self.memory_repo.get_all_memories()
            elif hasattr(self.memory_repo, 'search_memories'):
                # Fallback: use search with high limit
                return self.memory_repo.search_memories(query="", limit=100000)
            else:
                return []
        except Exception as e:
            self.logger.error(f"Failed to get all memories: {e}")
            return []

    def _search_memories_by_importance(
        self,
        max_importance: float,
        limit: int = 1000
    ) -> List:
        """Search memories by importance range"""
        try:
            if hasattr(self.memory_repo, 'search_memories'):
                # Use min_importance=0 and filter results
                memories = self.memory_repo.search_memories(
                    query="",
                    min_importance=0.0,
                    limit=limit * 2  # Get more to filter
                )
                # Filter by max_importance
                return [m for m in memories if m.importance <= max_importance][:limit]
            return []
        except Exception as e:
            self.logger.error(f"Failed to search by importance: {e}")
            return []

    def _is_protected(self, memory) -> bool:
        """
        Check if memory is protected from cleanup

        Protected memories:
        - Active tasks
        - Project initialization records
        - User preferences
        - High importance memories (>=0.7)

        Args:
            memory: Memory object to check

        Returns:
            True if memory should be protected
        """
        # Check category
        category = getattr(memory, 'category', None)
        if category and category in self.PROTECTED_CATEGORIES:
            return True

        # Check memory_type for backward compatibility
        memory_type = getattr(memory, 'memory_type', None)
        if memory_type == 'active_task':
            return True
        if memory_type == 'preference':
            return True

        # Protect high importance memories
        if memory.importance >= 0.7:
            return True

        return False

    def _delete_memory(self, memory) -> bool:
        """
        Delete a memory from repository

        Args:
            memory: Memory object to delete

        Returns:
            True if deletion succeeded
        """
        try:
            memory_id = memory.id
            if hasattr(self.memory_repo, 'delete_memory'):
                return self.memory_repo.delete_memory(memory_id)
            else:
                self.logger.warning("No delete_memory method available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory.id}: {e}")
            return False

    def get_cleanup_stats(self) -> Optional[Dict]:
        """
        Get statistics from last cleanup

        Returns:
            Last cleanup stats dict or None
        """
        return self._last_cleanup_stats

    def get_cleanup_preview(self) -> Dict[str, Any]:
        """
        Preview what would be cleaned without actually cleaning

        Returns:
            Preview dict with counts per strategy
        """
        try:
            all_memories = self._get_all_memories()
            now = datetime.now()

            # LFU preview
            lfu_scores = []
            for memory in all_memories:
                try:
                    score = self.lfu_calculator.lfu_decay(memory, now)
                    lfu_scores.append((memory, score))
                except Exception:
                    lfu_scores.append((memory, 0.0))
            lfu_scores.sort(key=lambda x: x[1])
            lfu_count = int(len(all_memories) * LFU_CLEANUP_BOTTOM_PERCENT)

            # Time expiry preview
            time_count = 0
            for memory in all_memories:
                if not memory.last_accessed:
                    continue
                expiry_days = (
                    TIME_EXPIRY_DAYS_LOW_IMPORTANCE
                    if memory.importance < 0.3
                    else TIME_EXPIRY_DAYS_DEFAULT
                )
                expiry_date = now - timedelta(days=expiry_days)
                if memory.last_accessed < expiry_date and not self._is_protected(memory):
                    time_count += 1

            # Importance preview
            importance_count = 0
            for memory in all_memories:
                if memory.importance <= IMPORTANCE_THRESHOLD_CLEANUP:
                    if memory.created_at:
                        age_days = (now - memory.created_at).days
                        if age_days < NEW_MEMORY_PROTECTION_DAYS:
                            continue
                    if not self._is_protected(memory):
                        importance_count += 1

            return {
                "total_memories": len(all_memories),
                "lfu_would_delete": lfu_count,
                "time_expiry_would_delete": time_count,
                "importance_would_delete": importance_count,
                "total_would_delete": lfu_count + time_count + importance_count
            }

        except Exception as e:
            self.logger.error(f"Failed to generate cleanup preview: {e}")
            return {"error": str(e)}
