#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFU (Least Frequently Used) Algorithm Implementation
Implements time-decay based LFU cache algorithm to avoid cache pollution problem
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

from ..storage.models import Memory
from ..utils.constants import (
    LFU_DECAY_TIME, LFU_COUNTER_INIT, LFU_PROB_BASE, LFU_DECAY_RATE,
    GLOBAL_HOT_TOP_N, PROJECT_HOT_TOP_N, COLD_THRESHOLD
)


class LFUCalculator:
    """LFU algorithm calculator"""

    def __init__(self):
        """Initialize LFU calculator"""
        self.logger = logging.getLogger('hibro.lfu_calculator')

    def lfu_decay(self, memory: Memory, current_time: Optional[datetime] = None) -> float:
        """
        LFU counter decay

        Decay LFU counter based on time interval to avoid short-term hotspots occupying cache long-term

        Args:
            memory: Memory object
            current_time: Current time, uses system time if None

        Returns:
            Decayed LFU counter value

        Formula: counter = counter - num_periods * decay_rate
        """
        if current_time is None:
            current_time = datetime.now()

        # Get current LFU counter value
        if memory.metadata and 'lfu_counter' in memory.metadata:
            lfu_counter = float(memory.metadata['lfu_counter'])
        else:
            # If no LFU counter, use access count as initial value
            lfu_counter = float(memory.access_count) if memory.access_count > 0 else LFU_COUNTER_INIT

        # Calculate time interval (minutes)
        if memory.last_accessed:
            elapsed_time = current_time - memory.last_accessed
            elapsed_minutes = elapsed_time.total_seconds() / 60
        else:
            # If no access time, assume it was long ago
            elapsed_minutes = 60 * 24  # 1 day

        # Calculate decay periods
        num_periods = elapsed_minutes / LFU_DECAY_TIME

        # Apply decay
        if num_periods > 0:
            decay_amount = num_periods * LFU_DECAY_RATE
            lfu_counter = max(0, lfu_counter - decay_amount)

        return lfu_counter

    def lfu_increment(self, memory: Memory) -> float:
        """
        Increment LFU counter on access (probabilistic growth)

        Use logarithmic growth to avoid counter explosion: counter++ if random() < 1/(counter+1)

        Args:
            memory: Memory object

        Returns:
            Updated LFU counter value
        """
        # Get current LFU counter value
        if memory.metadata and 'lfu_counter' in memory.metadata:
            lfu_counter = float(memory.metadata['lfu_counter'])
        else:
            lfu_counter = LFU_COUNTER_INIT

        # Probabilistic growth: use logarithmic growth to avoid counter explosion
        probability = LFU_PROB_BASE / (lfu_counter + 1)
        if random.random() < probability:
            lfu_counter += 1

        # Update memory object metadata
        if memory.metadata is None:
            memory.metadata = {}
        memory.metadata['lfu_counter'] = lfu_counter

        # Update access information
        memory.access_count += 1
        memory.last_accessed = datetime.now()

        self.logger.debug(f"LFU counter updated: ID={memory.id}, counter={lfu_counter}, probability={probability:.3f}")

        return lfu_counter

    def get_hot_memories(self, memories: List[Memory], current_time: Optional[datetime] = None,
                        top_n: int = GLOBAL_HOT_TOP_N) -> List[Memory]:
        """
        Sort by LFU counter (decayed) to get hot data

        Args:
            memories: Memory list
            current_time: Current time, uses system time if None
            top_n: Number of hot data to return

        Returns:
            Memory list sorted by heat (top_n items)
        """
        if not memories:
            return []

        if current_time is None:
            current_time = datetime.now()

        # Calculate decayed LFU counter for each memory
        memories_with_lfu = []
        for memory in memories:
            try:
                lfu_score = self.lfu_decay(memory, current_time)
                memories_with_lfu.append((memory, lfu_score))
            except Exception as e:
                self.logger.error(f"Failed to calculate LFU score: memory_id={memory.id}, error={e}")
                # Use default score
                memories_with_lfu.append((memory, 0.0))

        # Sort by LFU score descending
        sorted_memories = sorted(memories_with_lfu, key=lambda x: x[1], reverse=True)

        # Update memory object metadata lfu_counter
        hot_memories = []
        for i, (memory, lfu_score) in enumerate(sorted_memories[:top_n]):
            if memory.metadata is None:
                memory.metadata = {}
            memory.metadata['lfu_counter'] = lfu_score
            hot_memories.append(memory)

        self.logger.info(f"Hot data calculation complete: total={len(memories)}, hot_data={len(hot_memories)}, threshold={top_n}")

        return hot_memories

    def is_cold_memory(self, memory: Memory, current_time: Optional[datetime] = None) -> bool:
        """
        Determine if memory is cold data

        Args:
            memory: Memory object
            current_time: Current time

        Returns:
            Whether memory is cold data
        """
        lfu_score = self.lfu_decay(memory, current_time)
        return lfu_score < COLD_THRESHOLD

    def get_memory_heat_score(self, memory: Memory, current_time: Optional[datetime] = None) -> float:
        """
        Get memory heat score

        Args:
            memory: Memory object
            current_time: Current time

        Returns:
            Heat score (0.0 - infinity)
        """
        return self.lfu_decay(memory, current_time)

    def update_memory_access(self, memory: Memory) -> Memory:
        """
        Update memory access information (including LFU counter)

        Args:
            memory: Memory object

        Returns:
            Updated memory object
        """
        # Increment LFU counter
        self.lfu_increment(memory)

        return memory

    def batch_calculate_lfu_scores(self, memories: List[Memory],
                                  current_time: Optional[datetime] = None) -> Dict[int, float]:
        """
        Batch calculate LFU scores for memories

        Args:
            memories: Memory list
            current_time: Current time

        Returns:
            Mapping of memory ID to LFU score
        """
        if current_time is None:
            current_time = datetime.now()

        scores = {}
        for memory in memories:
            try:
                scores[memory.id] = self.lfu_decay(memory, current_time)
            except Exception as e:
                self.logger.error(f"Batch calculate LFU score failed: memory_id={memory.id}, error={e}")
                scores[memory.id] = 0.0

        return scores

    def get_statistics(self, memories: List[Memory], current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get LFU algorithm statistics

        Args:
            memories: Memory list
            current_time: Current time

        Returns:
            Statistics dictionary
        """
        if not memories:
            return {
                'total_memories': 0,
                'hot_memories': 0,
                'cold_memories': 0,
                'avg_lfu_score': 0.0,
                'max_lfu_score': 0.0,
                'min_lfu_score': 0.0
            }

        scores = self.batch_calculate_lfu_scores(memories, current_time)
        score_values = list(scores.values())

        hot_count = sum(1 for score in score_values if score >= COLD_THRESHOLD)
        cold_count = len(score_values) - hot_count

        return {
            'total_memories': len(memories),
            'hot_memories': hot_count,
            'cold_memories': cold_count,
            'avg_lfu_score': sum(score_values) / len(score_values),
            'max_lfu_score': max(score_values),
            'min_lfu_score': min(score_values),
            'cold_threshold': COLD_THRESHOLD
        }


# Global LFU calculator instance
lfu_calculator = LFUCalculator()


def lfu_decay(memory: Memory, current_time: Optional[datetime] = None) -> float:
    """
    LFU counter decay (convenience function)

    Args:
        memory: Memory object
        current_time: Current time

    Returns:
        Decayed LFU counter value
    """
    return lfu_calculator.lfu_decay(memory, current_time)


def lfu_increment(memory: Memory) -> float:
    """
    Increment LFU counter on access (convenience function)

    Args:
        memory: Memory object

    Returns:
        Updated LFU counter value
    """
    return lfu_calculator.lfu_increment(memory)


def get_hot_memories(memories: List[Memory], current_time: Optional[datetime] = None,
                    top_n: int = GLOBAL_HOT_TOP_N) -> List[Memory]:
    """
    Get hot data (convenience function)

    Args:
        memories: Memory list
        current_time: Current time
        top_n: Number of hot data to return

    Returns:
        Memory list sorted by heat
    """
    return lfu_calculator.get_hot_memories(memories, current_time, top_n)