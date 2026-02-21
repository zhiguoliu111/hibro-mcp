#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Behavior Analyzer
Tracks user query patterns, preference changes, and attention weights to provide data foundation for adaptive learning
"""

import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import re

from ..storage.models import Memory, MemoryRepository
from .bilingual_patterns import BILINGUAL_STOP_WORDS


class ActionType(Enum):
    """User action types"""
    QUERY = "query"           # Query memory
    STORE = "store"           # Store memory
    RECALL = "recall"         # Recall memory
    FEEDBACK = "feedback"     # User feedback
    CLICK = "click"           # Click recommendation
    IGNORE = "ignore"         # Ignore recommendation


class FeedbackType(Enum):
    """User feedback types"""
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"
    PARTIALLY_USEFUL = "partially_useful"
    VERY_USEFUL = "very_useful"


@dataclass
class UserBehavior:
    """User behavior data"""
    session_id: str
    action_type: ActionType
    target_memory_id: Optional[int] = None
    query_text: Optional[str] = None
    response_relevance: float = 0.0
    user_feedback: Optional[FeedbackType] = None
    interaction_duration: int = 0  # seconds
    context_data: Optional[Dict[str, Any]] = None
    project_path: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QueryPattern:
    """Query pattern"""
    pattern_type: str  # 'keyword', 'semantic', 'temporal'
    pattern_value: str
    frequency: int
    last_used: datetime
    success_rate: float  # Query success rate
    avg_relevance: float  # Average relevance


@dataclass
class PreferenceDrift:
    """Preference drift"""
    topic: str
    old_weight: float
    new_weight: float
    drift_rate: float  # Change rate
    confidence: float
    evidence_count: int
    detected_at: datetime


@dataclass
class AttentionWeight:
    """Attention weight"""
    topic: str
    weight: float
    decay_rate: float
    access_count: int
    last_access_time: Optional[datetime]
    boost_factor: float
    category: Optional[str]
    project_path: Optional[str]


class BehaviorAnalyzer:
    """User behavior analyzer"""

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize behavior analyzer

        Args:
            memory_repo: Memory repository
        """
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.behavior_analyzer')

        # Query pattern cache
        self._query_patterns_cache = {}
        self._cache_ttl = timedelta(hours=1)
        self._last_cache_update = None

    def track_user_behavior(self, behavior: UserBehavior) -> bool:
        """
        Track user behavior

        Args:
            behavior: User behavior data

        Returns:
            Whether successfully recorded
        """
        try:
            # Build context data
            context_data = behavior.context_data or {}
            context_json = json.dumps(context_data, ensure_ascii=False)

            # Insert behavior record
            query = """
                INSERT INTO user_behaviors (
                    session_id, action_type, target_memory_id, query_text,
                    response_relevance, user_feedback, interaction_duration,
                    context_data, project_path, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                behavior.session_id,
                behavior.action_type.value,
                behavior.target_memory_id,
                behavior.query_text,
                behavior.response_relevance,
                behavior.user_feedback.value if behavior.user_feedback else None,
                behavior.interaction_duration,
                context_json,
                behavior.project_path,
                behavior.timestamp
            )

            self.memory_repo.execute_query(query, params)
            self.logger.debug(f"Recorded user behavior: {behavior.action_type.value}")

            # Update attention weights
            if behavior.query_text:
                self._update_attention_weights_from_query(behavior.query_text, behavior.project_path)

            return True

        except Exception as e:
            self.logger.error(f"Failed to track user behavior: {e}")
            return False

    def analyze_query_patterns(self, session_id: Optional[str] = None,
                             days_back: int = 30) -> List[QueryPattern]:
        """
        Analyze query patterns

        Args:
            session_id: Session ID, analyze all sessions if None
            days_back: Number of days back to analyze

        Returns:
            List of query patterns
        """
        try:
            # Check cache
            cache_key = f"{session_id}_{days_back}"
            if self._is_cache_valid() and cache_key in self._query_patterns_cache:
                return self._query_patterns_cache[cache_key]

            # Query user behavior data
            query = """
                SELECT query_text, response_relevance, timestamp
                FROM user_behaviors
                WHERE action_type = 'query'
                AND query_text IS NOT NULL
                AND timestamp >= datetime('now', '-{} days')
            """.format(days_back)

            if session_id:
                query += " AND session_id = ?"
                params = (session_id,)
            else:
                params = ()

            query += " ORDER BY timestamp DESC"

            behaviors = self.memory_repo.execute_query(query, params)

            # Analyze patterns
            patterns = self._extract_query_patterns(behaviors)

            # Cache results
            self._query_patterns_cache[cache_key] = patterns
            self._last_cache_update = datetime.now()

            return patterns

        except Exception as e:
            self.logger.error(f"Failed to analyze query patterns: {e}")
            return []

    def detect_preference_drift(self, topic: str, project_path: Optional[str] = None,
                              window_days: int = 7) -> Optional[PreferenceDrift]:
        """
        Detect preference drift

        Args:
            topic: Topic
            project_path: Project path
            window_days: Time window (days)

        Returns:
            Preference drift information
        """
        try:
            # Get historical attention weights
            current_weight = self._get_current_attention_weight(topic, project_path)
            historical_weights = self._get_historical_attention_weights(
                topic, project_path, window_days
            )

            if len(historical_weights) < 2:
                return None

            # Calculate change trend
            old_avg = sum(w[1] for w in historical_weights[:len(historical_weights)//2]) / (len(historical_weights)//2)
            new_avg = sum(w[1] for w in historical_weights[len(historical_weights)//2:]) / (len(historical_weights) - len(historical_weights)//2)

            if abs(new_avg - old_avg) < 0.1:  # Change threshold
                return None

            # Calculate drift rate
            time_span = (historical_weights[-1][0] - historical_weights[0][0]).days
            drift_rate = (new_avg - old_avg) / max(time_span, 1)

            # Calculate confidence
            confidence = min(len(historical_weights) / 10.0, 1.0)

            return PreferenceDrift(
                topic=topic,
                old_weight=old_avg,
                new_weight=new_avg,
                drift_rate=drift_rate,
                confidence=confidence,
                evidence_count=len(historical_weights),
                detected_at=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Failed to detect preference drift: {e}")
            return None

    def calculate_attention_weights(self, topics: List[str],
                                  project_path: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate attention weights

        Args:
            topics: List of topics
            project_path: Project path

        Returns:
            Dictionary of topic weights
        """
        try:
            weights = {}

            for topic in topics:
                # Get base weight
                base_weight = self._get_current_attention_weight(topic, project_path)

                # Get access frequency
                access_count = self._get_topic_access_count(topic, project_path)

                # Get last access time
                last_access = self._get_last_access_time(topic, project_path)

                # Calculate time decay
                time_decay = self._calculate_time_decay(last_access)

                # Calculate final weight
                final_weight = base_weight * (1 + access_count * 0.1) * time_decay

                weights[topic] = min(final_weight, 2.0)  # Cap maximum weight

            return weights

        except Exception as e:
            self.logger.error(f"Failed to calculate attention weights: {e}")
            return {topic: 1.0 for topic in topics}

    def get_behavior_statistics(self, session_id: Optional[str] = None,
                              days_back: int = 30) -> Dict[str, Any]:
        """
        Get behavior statistics

        Args:
            session_id: Session ID
            days_back: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        try:
            base_query = """
                FROM user_behaviors
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days_back)

            if session_id:
                base_query += " AND session_id = ?"
                params = (session_id,)
            else:
                params = ()

            # Total behavior count
            total_query = "SELECT COUNT(*) " + base_query
            total_behaviors = self.memory_repo.execute_query(total_query, params)[0][0]

            # Statistics by type
            type_query = "SELECT action_type, COUNT(*) " + base_query + " GROUP BY action_type"
            type_stats = dict(self.memory_repo.execute_query(type_query, params))

            # Average relevance
            relevance_query = "SELECT AVG(response_relevance) " + base_query + " AND response_relevance > 0"
            avg_relevance = self.memory_repo.execute_query(relevance_query, params)[0][0] or 0.0

            # Feedback statistics
            feedback_query = "SELECT user_feedback, COUNT(*) " + base_query + " AND user_feedback IS NOT NULL GROUP BY user_feedback"
            feedback_stats = dict(self.memory_repo.execute_query(feedback_query, params))

            return {
                'total_behaviors': total_behaviors,
                'action_type_distribution': type_stats,
                'average_relevance': round(avg_relevance, 3),
                'feedback_distribution': feedback_stats,
                'analysis_period_days': days_back
            }

        except Exception as e:
            self.logger.error(f"Failed to get behavior statistics: {e}")
            return {}

    def _extract_query_patterns(self, behaviors: List[Tuple]) -> List[QueryPattern]:
        """Extract query patterns from behavior data"""
        patterns = []

        # Keyword patterns
        keyword_counter = Counter()
        keyword_relevance = defaultdict(list)

        for query_text, relevance, timestamp in behaviors:
            if not query_text:
                continue

            # Extract keywords
            keywords = self._extract_keywords(query_text)
            for keyword in keywords:
                keyword_counter[keyword] += 1
                keyword_relevance[keyword].append(relevance or 0.0)

        # Generate keyword patterns
        for keyword, frequency in keyword_counter.most_common(20):
            if frequency >= 2:  # Must appear at least twice
                relevances = keyword_relevance[keyword]
                avg_relevance = sum(relevances) / len(relevances)
                success_rate = sum(1 for r in relevances if r > 0.5) / len(relevances)

                patterns.append(QueryPattern(
                    pattern_type='keyword',
                    pattern_value=keyword,
                    frequency=frequency,
                    last_used=datetime.now(),  # Simplified
                    success_rate=success_rate,
                    avg_relevance=avg_relevance
                ))

        return patterns

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract query keywords"""
        # Simple keyword extraction
        # Remove punctuation, convert to lowercase, split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stop words
        # Chinese stop words: common particles and pronouns
        stop_words = BILINGUAL_STOP_WORDS  # Bilingual stop words (Chinese and English NLP)
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]

        return keywords

    def _update_attention_weights_from_query(self, query_text: str, project_path: Optional[str]):
        """Update attention weights from query"""
        try:
            keywords = self._extract_keywords(query_text)

            for keyword in keywords:
                # Get current weight
                current_weight = self._get_current_attention_weight(keyword, project_path)

                # Increase weight
                new_weight = min(current_weight + 0.1, 2.0)

                # Update database
                self._update_attention_weight(keyword, new_weight, project_path)

        except Exception as e:
            self.logger.error(f"Failed to update attention weights: {e}")

    def _get_current_attention_weight(self, topic: str, project_path: Optional[str]) -> float:
        """Get current attention weight"""
        try:
            query = """
                SELECT weight FROM attention_weights
                WHERE topic = ? AND (project_path = ? OR project_path IS NULL)
                ORDER BY project_path DESC LIMIT 1
            """

            result = self.memory_repo.execute_query(query, (topic, project_path))
            return result[0][0] if result else 1.0

        except Exception:
            return 1.0

    def _update_attention_weight(self, topic: str, weight: float, project_path: Optional[str]):
        """Update attention weight"""
        try:
            # Try to update existing record
            update_query = """
                UPDATE attention_weights
                SET weight = ?, access_count = access_count + 1, last_access_time = CURRENT_TIMESTAMP
                WHERE topic = ? AND (project_path = ? OR (project_path IS NULL AND ? IS NULL))
            """

            rows_affected = self.memory_repo.execute_query(update_query, (weight, topic, project_path, project_path))

            # If no existing record, insert new record
            if not rows_affected:
                insert_query = """
                    INSERT INTO attention_weights (topic, weight, project_path, access_count, last_access_time)
                    VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                """
                self.memory_repo.execute_query(insert_query, (topic, weight, project_path))

        except Exception as e:
            self.logger.error(f"Failed to update attention weight: {e}")

    def _get_historical_attention_weights(self, topic: str, project_path: Optional[str],
                                        days_back: int) -> List[Tuple[datetime, float]]:
        """Get historical attention weights"""
        # Simplified implementation, returns empty list
        return []

    def _get_topic_access_count(self, topic: str, project_path: Optional[str]) -> int:
        """Get topic access count"""
        try:
            query = """
                SELECT access_count FROM attention_weights
                WHERE topic = ? AND (project_path = ? OR project_path IS NULL)
                ORDER BY project_path DESC LIMIT 1
            """

            result = self.memory_repo.execute_query(query, (topic, project_path))
            return result[0][0] if result else 0

        except Exception:
            return 0

    def _get_last_access_time(self, topic: str, project_path: Optional[str]) -> Optional[datetime]:
        """Get last access time"""
        try:
            query = """
                SELECT last_access_time FROM attention_weights
                WHERE topic = ? AND (project_path = ? OR project_path IS NULL)
                ORDER BY project_path DESC LIMIT 1
            """

            result = self.memory_repo.execute_query(query, (topic, project_path))
            if result and result[0][0]:
                return datetime.fromisoformat(result[0][0])
            return None

        except Exception:
            return None

    def _calculate_time_decay(self, last_access: Optional[datetime]) -> float:
        """Calculate time decay factor"""
        if not last_access:
            return 0.5  # Default weight for never accessed

        days_since = (datetime.now() - last_access).days

        # Exponential decay with 7-day half-life
        decay_factor = 0.5 ** (days_since / 7.0)
        return max(decay_factor, 0.1)  # Minimum weight 0.1

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid"""
        if not self._last_cache_update:
            return False

        return datetime.now() - self._last_cache_update < self._cache_ttl