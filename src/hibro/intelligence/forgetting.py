#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Forgetting Mechanism Module
Implements human-like memory forgetting algorithm based on time decay, importance, and usage frequency
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..storage.models import Memory
from ..utils.helpers import calculate_time_decay


@dataclass
class ForgettingRule:
    """Forgetting rule configuration"""
    memory_type: str
    base_decay_rate: float
    min_importance_threshold: float
    access_frequency_weight: float
    time_weight: float
    importance_weight: float


class ForgettingManager:
    """Intelligent forgetting manager"""

    def __init__(self, config):
        """
        Initialize forgetting manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.forgetting_manager')

        # Forgetting rules for different memory types
        self.forgetting_rules = {
            'preference': ForgettingRule(
                memory_type='preference',
                base_decay_rate=0.02,  # Preference memories decay slowly
                min_importance_threshold=0.3,
                access_frequency_weight=0.4,
                time_weight=0.2,
                importance_weight=0.4
            ),
            'decision': ForgettingRule(
                memory_type='decision',
                base_decay_rate=0.05,  # Decision memories decay moderately
                min_importance_threshold=0.4,
                access_frequency_weight=0.3,
                time_weight=0.3,
                importance_weight=0.4
            ),
            'project': ForgettingRule(
                memory_type='project',
                base_decay_rate=0.08,  # Project memories decay faster
                min_importance_threshold=0.3,
                access_frequency_weight=0.4,
                time_weight=0.4,
                importance_weight=0.2
            ),
            'conversation': ForgettingRule(
                memory_type='conversation',
                base_decay_rate=0.15,  # General conversations decay quickly
                min_importance_threshold=0.2,
                access_frequency_weight=0.5,
                time_weight=0.4,
                importance_weight=0.1
            ),
            'important': ForgettingRule(
                memory_type='important',
                base_decay_rate=0.01,  # Important memories barely decay
                min_importance_threshold=0.5,
                access_frequency_weight=0.2,
                time_weight=0.1,
                importance_weight=0.7
            ),
            'learning': ForgettingRule(
                memory_type='learning',
                base_decay_rate=0.06,  # Learning memories decay moderately
                min_importance_threshold=0.3,
                access_frequency_weight=0.4,
                time_weight=0.3,
                importance_weight=0.3
            )
        }

        # Global forgetting parameters
        self.global_decay_rate = config.forgetting.time_decay_rate
        self.min_global_importance = config.forgetting.min_importance
        self.cleanup_interval_days = config.forgetting.cleanup_interval_days

    def calculate_current_importance(self, memory: Memory) -> float:
        """
        Calculate current importance of memory (considering time decay)

        Args:
            memory: Memory object

        Returns:
            Current importance score
        """
        if not memory.last_accessed:
            return memory.importance

        # Get forgetting rule for memory type
        rule = self.forgetting_rules.get(memory.memory_type, self.forgetting_rules['conversation'])

        # Calculate time factor
        days_since_access = (datetime.now() - memory.last_accessed).days
        time_factor = self._calculate_time_factor(days_since_access, rule.base_decay_rate)

        # Calculate access frequency factor
        frequency_factor = self._calculate_frequency_factor(memory.access_count, days_since_access)

        # Calculate importance factor
        importance_factor = self._calculate_importance_factor(memory.importance)

        # Calculate current importance
        current_importance = (
            time_factor * rule.time_weight +
            frequency_factor * rule.access_frequency_weight +
            importance_factor * rule.importance_weight
        ) * memory.importance

        return max(current_importance, 0.0)

    def should_forget(self, memory: Memory) -> Tuple[bool, str]:
        """
        Determine if a memory should be forgotten

        Args:
            memory: Memory object

        Returns:
            (should_forget, forget_reason)
        """
        # Get current importance
        current_importance = self.calculate_current_importance(memory)

        # Get forgetting rule for memory type
        rule = self.forgetting_rules.get(memory.memory_type, self.forgetting_rules['conversation'])

        # Check importance threshold
        if current_importance < rule.min_importance_threshold:
            return True, f"Importance too low ({current_importance:.3f} < {rule.min_importance_threshold})"

        # Check global minimum importance
        if current_importance < self.min_global_importance:
            return True, f"Below global minimum importance ({current_importance:.3f} < {self.min_global_importance})"

        # Check long-term no access
        if memory.last_accessed:
            days_since_access = (datetime.now() - memory.last_accessed).days
            max_days_without_access = self._get_max_days_without_access(memory.memory_type)

            if days_since_access > max_days_without_access and current_importance < 0.5:
                return True, f"Long time no access ({days_since_access} days > {max_days_without_access} days)"

        return False, ""

    def get_forgetting_candidates(self, memories: List[Memory],
                                 target_count: Optional[int] = None) -> List[Tuple[Memory, str, float]]:
        """
        Get list of forgetting candidate memories

        Args:
            memories: List of memories
            target_count: Target number to forget, if None return all candidates

        Returns:
            List of (memory, forget_reason, current_importance)
        """
        candidates = []

        for memory in memories:
            should_forget, reason = self.should_forget(memory)
            if should_forget:
                current_importance = self.calculate_current_importance(memory)
                candidates.append((memory, reason, current_importance))

        # Sort by current importance (low importance first)
        candidates.sort(key=lambda x: x[2])

        if target_count is not None:
            candidates = candidates[:target_count]

        self.logger.info(f"Found {len(candidates)} forgetting candidate memories")
        return candidates

    def update_memory_importance(self, memory: Memory) -> Memory:
        """
        Update memory importance score

        Args:
            memory: Memory object

        Returns:
            Updated memory object
        """
        # Calculate new importance
        new_importance = self.calculate_current_importance(memory)

        # Update memory object
        memory.importance = new_importance

        return memory

    def reinforce_memory(self, memory: Memory, reinforcement_factor: float = 0.1) -> Memory:
        """
        Reinforce memory (increase importance)

        Args:
            memory: Memory object
            reinforcement_factor: Reinforcement factor

        Returns:
            Reinforced memory object
        """
        # Increase importance but not exceed 1.0
        memory.importance = min(memory.importance + reinforcement_factor, 1.0)

        # Update access information
        memory.update_access()

        self.logger.debug(f"Memory reinforced: ID={memory.id}, new importance={memory.importance:.3f}")
        return memory

    def decay_memory(self, memory: Memory, decay_factor: float = 0.1) -> Memory:
        """
        Decay memory (decrease importance)

        Args:
            memory: Memory object
            decay_factor: Decay factor

        Returns:
            Decayed memory object
        """
        # Decrease importance but not below 0.0
        memory.importance = max(memory.importance - decay_factor, 0.0)

        self.logger.debug(f"Memory decayed: ID={memory.id}, new importance={memory.importance:.3f}")
        return memory

    def get_forgetting_statistics(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Get forgetting statistics

        Args:
            memories: List of memories

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_memories': len(memories),
            'forgetting_candidates': 0,
            'by_type': {},
            'importance_distribution': {
                'high': 0,    # >= 0.7
                'medium': 0,  # 0.3 - 0.7
                'low': 0      # < 0.3
            },
            'age_distribution': {
                'recent': 0,    # < 7 days
                'medium': 0,    # 7-30 days
                'old': 0        # > 30 days
            }
        }

        for memory in memories:
            # Count forgetting candidates
            should_forget, _ = self.should_forget(memory)
            if should_forget:
                stats['forgetting_candidates'] += 1

            # Statistics by type
            memory_type = memory.memory_type
            if memory_type not in stats['by_type']:
                stats['by_type'][memory_type] = {
                    'count': 0,
                    'avg_importance': 0.0,
                    'forgetting_candidates': 0
                }

            stats['by_type'][memory_type]['count'] += 1
            stats['by_type'][memory_type]['avg_importance'] += memory.importance

            if should_forget:
                stats['by_type'][memory_type]['forgetting_candidates'] += 1

            # Importance distribution
            current_importance = self.calculate_current_importance(memory)
            if current_importance >= 0.7:
                stats['importance_distribution']['high'] += 1
            elif current_importance >= 0.3:
                stats['importance_distribution']['medium'] += 1
            else:
                stats['importance_distribution']['low'] += 1

            # Age distribution
            if memory.created_at:
                days_old = (datetime.now() - memory.created_at).days
                if days_old < 7:
                    stats['age_distribution']['recent'] += 1
                elif days_old < 30:
                    stats['age_distribution']['medium'] += 1
                else:
                    stats['age_distribution']['old'] += 1

        # Calculate average importance
        for type_stats in stats['by_type'].values():
            if type_stats['count'] > 0:
                type_stats['avg_importance'] /= type_stats['count']

        return stats

    def _calculate_time_factor(self, days_since_access: int, decay_rate: float) -> float:
        """
        Calculate time decay factor

        Args:
            days_since_access: Days since last access
            decay_rate: Decay rate

        Returns:
            Time factor (0.0-1.0)
        """
        return math.exp(-decay_rate * days_since_access)

    def _calculate_frequency_factor(self, access_count: int, days_since_creation: int) -> float:
        """
        Calculate access frequency factor

        Args:
            access_count: Access count
            days_since_creation: Days since creation

        Returns:
            Frequency factor (0.0-1.0)
        """
        if days_since_creation <= 0:
            return 1.0

        # Calculate average access frequency (accesses/day)
        avg_frequency = access_count / max(days_since_creation, 1)

        # Use logarithmic function to smooth frequency impact
        frequency_factor = math.log(1 + avg_frequency * 10) / math.log(11)

        return min(frequency_factor, 1.0)

    def _calculate_importance_factor(self, importance: float) -> float:
        """
        Calculate importance factor

        Args:
            importance: Original importance

        Returns:
            Importance factor (0.0-1.0)
        """
        # Use square root function to make high importance memories harder to forget
        return math.sqrt(importance)

    def _get_max_days_without_access(self, memory_type: str) -> int:
        """
        Get maximum days without access for memory type

        Args:
            memory_type: Memory type

        Returns:
            Maximum days without access
        """
        max_days_map = {
            'preference': 365,    # Preference memories can go long without access
            'decision': 180,      # Decision memories: half a year
            'project': 90,        # Project memories: 3 months
            'important': 730,     # Important memories: 2 years
            'learning': 120,      # Learning memories: 4 months
            'conversation': 30    # General conversations: 1 month
        }

        return max_days_map.get(memory_type, 60)  # Default: 2 months

    def create_forgetting_report(self, memories: List[Memory]) -> str:
        """
        Create forgetting report

        Args:
            memories: List of memories

        Returns:
            Forgetting report text
        """
        stats = self.get_forgetting_statistics(memories)
        candidates = self.get_forgetting_candidates(memories)

        report = []
        report.append("# Intelligent Forgetting System Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Overall Statistics")
        report.append(f"- Total memories: {stats['total_memories']}")
        report.append(f"- Forgetting candidates: {stats['forgetting_candidates']}")
        report.append(f"- Forgetting rate: {stats['forgetting_candidates']/max(stats['total_memories'], 1)*100:.1f}%")
        report.append("")

        report.append("## Importance Distribution")
        imp_dist = stats['importance_distribution']
        report.append(f"- High importance (≥0.7): {imp_dist['high']}")
        report.append(f"- Medium importance (0.3-0.7): {imp_dist['medium']}")
        report.append(f"- Low importance (<0.3): {imp_dist['low']}")
        report.append("")

        report.append("## Statistics by Type")
        for memory_type, type_stats in stats['by_type'].items():
            report.append(f"### {memory_type}")
            report.append(f"- Count: {type_stats['count']}")
            report.append(f"- Average importance: {type_stats['avg_importance']:.3f}")
            report.append(f"- Forgetting candidates: {type_stats['forgetting_candidates']}")
            report.append("")

        if candidates:
            report.append("## Forgetting Candidate Details")
            for i, (memory, reason, current_importance) in enumerate(candidates[:10], 1):
                report.append(f"{i}. ID: {memory.id}")
                report.append(f"   Type: {memory.memory_type}")
                report.append(f"   Current importance: {current_importance:.3f}")
                report.append(f"   Forget reason: {reason}")
                report.append(f"   Content: {memory.content[:50]}...")
                report.append("")

        return "\n".join(report)