#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Importance Scoring System
Dynamically calculates and adjusts memory importance scores based on multiple factors
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..storage.models import Memory


class ImportanceFactorType(Enum):
    """Importance factor types"""
    USER_EXPLICIT = "user_explicit"      # User explicit marking
    CONTENT_ANALYSIS = "content_analysis" # Content analysis
    ACCESS_PATTERN = "access_pattern"     # Access pattern
    TEMPORAL = "temporal"                 # Temporal factor
    CONTEXTUAL = "contextual"            # Contextual relevance
    REINFORCEMENT = "reinforcement"       # Reinforcement learning


@dataclass
class ImportanceFactor:
    """Importance factor"""
    factor_type: ImportanceFactorType
    weight: float
    score: float
    confidence: float
    description: str


class ImportanceScorer:
    """Memory importance scorer"""

    def __init__(self, config):
        """
        Initialize importance scorer

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.importance_scorer')

        # Different memory type base weight configuration
        self.type_weights = {
            'preference': {
                'user_explicit': 0.4,
                'content_analysis': 0.3,
                'access_pattern': 0.2,
                'temporal': 0.05,
                'contextual': 0.05
            },
            'decision': {
                'user_explicit': 0.3,
                'content_analysis': 0.4,
                'access_pattern': 0.15,
                'temporal': 0.1,
                'contextual': 0.05
            },
            'project': {
                'user_explicit': 0.2,
                'content_analysis': 0.3,
                'access_pattern': 0.3,
                'temporal': 0.1,
                'contextual': 0.1
            },
            'important': {
                'user_explicit': 0.5,
                'content_analysis': 0.3,
                'access_pattern': 0.1,
                'temporal': 0.05,
                'contextual': 0.05
            },
            'conversation': {
                'user_explicit': 0.2,
                'content_analysis': 0.2,
                'access_pattern': 0.4,
                'temporal': 0.15,
                'contextual': 0.05
            },
            'learning': {
                'user_explicit': 0.25,
                'content_analysis': 0.35,
                'access_pattern': 0.25,
                'temporal': 0.1,
                'contextual': 0.05
            }
        }

        # Content analysis keyword weights
        self.content_keywords = {
            # High importance keywords
            'high': {
                'keywords': ['important', 'key', 'core', 'must', 'focus', 'remember', 'special', 'absolutely'],
                'weight': 1.0
            },
            # Medium importance keywords
            'medium': {
                'keywords': ['note', 'consider', 'suggest', 'recommend', 'better', 'should', 'need'],
                'weight': 0.6
            },
            # Technical keywords
            'technical': {
                'keywords': ['architecture', 'design', 'implementation', 'optimization', 'performance', 'security', 'testing', 'deployment'],
                'weight': 0.7
            },
            # Decision keywords
            'decision': {
                'keywords': ['choose', 'decide', 'adopt', 'use', 'solution', 'strategy', 'method'],
                'weight': 0.8
            }
        }

    def calculate_importance(self, memory: Memory, context: Optional[Dict[str, Any]] = None) -> Tuple[float, List[ImportanceFactor]]:
        """
        Calculate memory importance score

        Args:
            memory: Memory object
            context: Context information

        Returns:
            (Importance score, Factor list)
        """
        factors = []

        # 1. User explicit marking factor
        user_factor = self._calculate_user_explicit_factor(memory)
        factors.append(user_factor)

        # 2. Content analysis factor
        content_factor = self._calculate_content_analysis_factor(memory)
        factors.append(content_factor)

        # 3. Access pattern factor
        access_factor = self._calculate_access_pattern_factor(memory)
        factors.append(access_factor)

        # 4. Temporal factor
        temporal_factor = self._calculate_temporal_factor(memory)
        factors.append(temporal_factor)

        # 5. Contextual relevance factor
        contextual_factor = self._calculate_contextual_factor(memory, context)
        factors.append(contextual_factor)

        # Calculate weighted total score
        total_score = self._calculate_weighted_score(memory.memory_type, factors)

        # Ensure score is in reasonable range
        final_score = max(0.0, min(1.0, total_score))

        self.logger.debug(f"Memory {memory.id} importance score: {final_score:.3f}")
        return final_score, factors

    def _calculate_user_explicit_factor(self, memory: Memory) -> ImportanceFactor:
        """Calculate user explicit marking factor"""
        # Check if user explicitly marked importance
        base_importance = memory.importance

        # Check user marking in metadata
        user_marked = False
        if memory.metadata and 'user_marked_important' in memory.metadata:
            user_marked = memory.metadata['user_marked_important']

        if user_marked:
            score = 1.0
            confidence = 1.0
            description = "User explicitly marked as important"
        else:
            # Based on initial importance score
            score = base_importance
            confidence = 0.7
            description = f"Based on initial importance score ({base_importance:.2f})"

        return ImportanceFactor(
            factor_type=ImportanceFactorType.USER_EXPLICIT,
            weight=0.0,  # Weight will be set in subsequent calculation
            score=score,
            confidence=confidence,
            description=description
        )

    def _calculate_content_analysis_factor(self, memory: Memory) -> ImportanceFactor:
        """Calculate content analysis factor"""
        content = memory.content.lower()
        score = 0.0
        matched_keywords = []

        # Analyze keywords
        for category, config in self.content_keywords.items():
            for keyword in config['keywords']:
                if keyword in content:
                    score += config['weight']
                    matched_keywords.append(f"{keyword}({category})")

        # Analyze content length (moderate length usually more important)
        content_length = len(memory.content)
        if 50 <= content_length <= 200:
            score += 0.2
        elif content_length > 500:
            score -= 0.1

        # Analyze punctuation (exclamation mark indicates emphasis)
        if '!' in memory.content:
            score += 0.1

        # Normalize score
        score = min(score / 2.0, 1.0)  # Divide by 2 for normalization

        confidence = 0.8 if matched_keywords else 0.5
        description = f"Keyword matches: {', '.join(matched_keywords[:3])}" if matched_keywords else "No special keywords"

        return ImportanceFactor(
            factor_type=ImportanceFactorType.CONTENT_ANALYSIS,
            weight=0.0,
            score=score,
            confidence=confidence,
            description=description
        )

    def _calculate_access_pattern_factor(self, memory: Memory) -> ImportanceFactor:
        """Calculate access pattern factor"""
        access_count = memory.access_count
        days_since_creation = 1

        if memory.created_at:
            days_since_creation = max(1, (datetime.now() - memory.created_at).days)

        # Calculate access frequency
        access_frequency = access_count / days_since_creation

        # Calculate access pattern score
        if access_frequency >= 1.0:  # Accessed at least once per day
            score = 1.0
        elif access_frequency >= 0.5:  # Accessed once every two days
            score = 0.8
        elif access_frequency >= 0.1:  # Accessed once every ten days
            score = 0.6
        elif access_count >= 3:  # Accessed at least 3 times
            score = 0.4
        elif access_count >= 1:  # Accessed at least once
            score = 0.2
        else:
            score = 0.1

        confidence = 0.9 if access_count > 0 else 0.3
        description = f"Accessed {access_count} times, frequency {access_frequency:.2f}/day"

        return ImportanceFactor(
            factor_type=ImportanceFactorType.ACCESS_PATTERN,
            weight=0.0,
            score=score,
            confidence=confidence,
            description=description
        )

    def _calculate_temporal_factor(self, memory: Memory) -> ImportanceFactor:
        """Calculate temporal factor"""
        if not memory.created_at:
            return ImportanceFactor(
                factor_type=ImportanceFactorType.TEMPORAL,
                weight=0.0,
                score=0.5,
                confidence=0.3,
                description="No creation time information"
            )

        days_since_creation = (datetime.now() - memory.created_at).days
        days_since_access = 0

        if memory.last_accessed:
            days_since_access = (datetime.now() - memory.last_accessed).days

        # Freshness score (newer is more important, but not linear)
        freshness_score = math.exp(-0.01 * days_since_creation)

        # Recent access score
        recency_score = math.exp(-0.05 * days_since_access) if days_since_access > 0 else 1.0

        # Combined temporal score
        score = (freshness_score + recency_score) / 2

        confidence = 0.8
        description = f"Created {days_since_creation} days ago, last accessed {days_since_access} days ago"

        return ImportanceFactor(
            factor_type=ImportanceFactorType.TEMPORAL,
            weight=0.0,
            score=score,
            confidence=confidence,
            description=description
        )

    def _calculate_contextual_factor(self, memory: Memory, context: Optional[Dict[str, Any]]) -> ImportanceFactor:
        """Calculate contextual relevance factor"""
        if not context:
            return ImportanceFactor(
                factor_type=ImportanceFactorType.CONTEXTUAL,
                weight=0.0,
                score=0.5,
                confidence=0.3,
                description="No context information"
            )

        score = 0.5  # Default score
        relevance_factors = []

        # Check project relevance
        if 'current_project' in context and memory.metadata:
            memory_project = memory.metadata.get('project_path')
            current_project = context['current_project']

            if memory_project and current_project:
                if memory_project == current_project:
                    score += 0.3
                    relevance_factors.append("Same project")
                elif memory_project in current_project or current_project in memory_project:
                    score += 0.1
                    relevance_factors.append("Related project")

        # Check category relevance
        if 'current_category' in context and memory.category:
            if memory.category == context['current_category']:
                score += 0.2
                relevance_factors.append("Same category")

        # Check tech stack relevance
        if 'tech_stack' in context:
            tech_stack = context['tech_stack']
            content_lower = memory.content.lower()

            for tech in tech_stack:
                if tech.lower() in content_lower:
                    score += 0.1
                    relevance_factors.append(f"Tech stack match({tech})")
                    break

        score = min(score, 1.0)
        confidence = 0.7 if relevance_factors else 0.4
        description = f"Relevance: {', '.join(relevance_factors)}" if relevance_factors else "No obvious relevance"

        return ImportanceFactor(
            factor_type=ImportanceFactorType.CONTEXTUAL,
            weight=0.0,
            score=score,
            confidence=confidence,
            description=description
        )

    def _calculate_weighted_score(self, memory_type: str, factors: List[ImportanceFactor]) -> float:
        """Calculate weighted total score"""
        weights = self.type_weights.get(memory_type, self.type_weights['conversation'])

        total_score = 0.0
        total_weight = 0.0

        for factor in factors:
            # Get weight based on factor type
            factor_key = factor.factor_type.value
            weight = weights.get(factor_key, 0.1)

            # Adjust weight considering confidence
            adjusted_weight = weight * factor.confidence

            # Update factor weight (for debugging)
            factor.weight = adjusted_weight

            total_score += factor.score * adjusted_weight
            total_weight += adjusted_weight

        # Normalize
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5  # Default score

    def update_importance_by_feedback(self, memory: Memory, feedback_type: str,
                                    feedback_strength: float = 0.1) -> float:
        """
        Update importance based on user feedback

        Args:
            memory: Memory object
            feedback_type: Feedback type ('positive', 'negative', 'neutral')
            feedback_strength: Feedback strength

        Returns:
            Updated importance score
        """
        current_importance = memory.importance

        if feedback_type == 'positive':
            # Positive feedback increases importance
            new_importance = min(1.0, current_importance + feedback_strength)
        elif feedback_type == 'negative':
            # Negative feedback decreases importance
            new_importance = max(0.0, current_importance - feedback_strength)
        else:
            # Neutral feedback no change
            new_importance = current_importance

        # Record feedback history
        if not memory.metadata:
            memory.metadata = {}

        if 'feedback_history' not in memory.metadata:
            memory.metadata['feedback_history'] = []

        memory.metadata['feedback_history'].append({
            'type': feedback_type,
            'strength': feedback_strength,
            'timestamp': datetime.now().isoformat(),
            'old_importance': current_importance,
            'new_importance': new_importance
        })

        memory.importance = new_importance

        self.logger.info(f"Memory {memory.id} importance updated: {current_importance:.3f} -> {new_importance:.3f} ({feedback_type})")
        return new_importance

    def batch_recalculate_importance(self, memories: List[Memory],
                                   context: Optional[Dict[str, Any]] = None) -> List[Tuple[Memory, float, List[ImportanceFactor]]]:
        """
        Batch recalculate memory importance

        Args:
            memories: List of memories
            context: Context information

        Returns:
            List of (memory, new importance score, factor list)
        """
        results = []

        for memory in memories:
            new_importance, factors = self.calculate_importance(memory, context)
            memory.importance = new_importance
            results.append((memory, new_importance, factors))

        self.logger.info(f"Batch recalculated importance for {len(memories)} memories")
        return results

    def get_importance_distribution(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Get importance distribution statistics

        Args:
            memories: List of memories

        Returns:
            Distribution statistics
        """
        if not memories:
            return {}

        importances = [memory.importance for memory in memories]

        return {
            'count': len(importances),
            'mean': sum(importances) / len(importances),
            'min': min(importances),
            'max': max(importances),
            'distribution': {
                'very_high': len([i for i in importances if i >= 0.9]),
                'high': len([i for i in importances if 0.7 <= i < 0.9]),
                'medium': len([i for i in importances if 0.4 <= i < 0.7]),
                'low': len([i for i in importances if 0.2 <= i < 0.4]),
                'very_low': len([i for i in importances if i < 0.2])
            }
        }

    def create_importance_report(self, memory: Memory, factors: List[ImportanceFactor]) -> str:
        """
        Create importance scoring report

        Args:
            memory: Memory object
            factors: List of importance factors

        Returns:
            Report text
        """
        report = []
        report.append(f"# Memory Importance Scoring Report")
        report.append(f"Memory ID: {memory.id}")
        report.append(f"Memory Type: {memory.memory_type}")
        report.append(f"Final Importance: {memory.importance:.3f}")
        report.append("")

        report.append("## Scoring Factor Details")
        for factor in factors:
            report.append(f"### {factor.factor_type.value}")
            report.append(f"- Score: {factor.score:.3f}")
            report.append(f"- Weight: {factor.weight:.3f}")
            report.append(f"- Confidence: {factor.confidence:.3f}")
            report.append(f"- Description: {factor.description}")
            report.append("")

        report.append("## Memory Content")
        report.append(f"```")
        report.append(memory.content[:200] + "..." if len(memory.content) > 200 else memory.content)
        report.append(f"```")

        return "\n".join(report)