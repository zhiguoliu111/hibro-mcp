#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Prompt System
Provides intelligent prompts and suggestions based on context and historical memories
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.memory_engine import MemoryEngine
from ..intelligence import SimilarityCalculator, SemanticSearchEngine, ForgettingManager
from ..utils.config import Config


@dataclass
class Suggestion:
    """Intelligent Suggestion"""
    type: str  # 'memory', 'pattern', 'reminder', 'optimization'
    title: str
    content: str
    confidence: float
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class IntelligentPromptSystem:
    """Intelligent Prompt System"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize intelligent prompt system

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.intelligent_prompt')

        # Initialize components
        self.similarity_calc = SimilarityCalculator()
        self.search_engine = SemanticSearchEngine(self.similarity_calc)
        self.forgetting_manager = ForgettingManager(config)

        # Prompt rule configuration
        self.prompt_rules = {
            'similar_context': {
                'enabled': True,
                'threshold': 0.6,
                'max_suggestions': 3,
                'description': 'Memory reminders based on similar context'
            },
            'forgotten_memories': {
                'enabled': True,
                'days_threshold': 7,
                'importance_threshold': 0.6,
                'description': 'Reminders for important but forgotten memories'
            },
            'pattern_recognition': {
                'enabled': True,
                'min_occurrences': 3,
                'description': 'Repeated pattern recognition and suggestions'
            },
            'project_context': {
                'enabled': True,
                'auto_load': True,
                'description': 'Automatic loading of project-related memories'
            },
            'learning_reinforcement': {
                'enabled': True,
                'review_interval_days': 3,
                'description': 'Learning content review reminders'
            }
        }

    def get_contextual_suggestions(self, current_context: str,
                                 context_info: Optional[Dict[str, Any]] = None) -> List[Suggestion]:
        """
        Get intelligent suggestions based on current context

        Args:
            current_context: Current context content
            context_info: Context information (project path, file type, etc.)

        Returns:
            List of suggestions
        """
        suggestions = []

        try:
            # 1. Similar context memory reminders
            if self.prompt_rules['similar_context']['enabled']:
                similar_suggestions = self._get_similar_context_suggestions(current_context, context_info)
                suggestions.extend(similar_suggestions)

            # 2. Forgotten important memories
            if self.prompt_rules['forgotten_memories']['enabled']:
                forgotten_suggestions = self._get_forgotten_memory_suggestions(context_info)
                suggestions.extend(forgotten_suggestions)

            # 3. Pattern recognition suggestions
            if self.prompt_rules['pattern_recognition']['enabled']:
                pattern_suggestions = self._get_pattern_suggestions(current_context, context_info)
                suggestions.extend(pattern_suggestions)

            # 4. Project context suggestions
            if self.prompt_rules['project_context']['enabled'] and context_info:
                project_suggestions = self._get_project_context_suggestions(context_info)
                suggestions.extend(project_suggestions)

            # 5. Learning reinforcement suggestions
            if self.prompt_rules['learning_reinforcement']['enabled']:
                learning_suggestions = self._get_learning_reinforcement_suggestions()
                suggestions.extend(learning_suggestions)

            # Sort by priority and confidence
            suggestions.sort(key=lambda x: (x.priority, x.confidence), reverse=True)

            # Limit number of suggestions
            max_suggestions = 10
            suggestions = suggestions[:max_suggestions]

            self.logger.info(f"Generated {len(suggestions)} intelligent suggestions")
            return suggestions

        except Exception as e:
            self.logger.error(f"Failed to generate intelligent suggestions: {e}")
            return []

    def _get_similar_context_suggestions(self, current_context: str,
                                       context_info: Optional[Dict[str, Any]]) -> List[Suggestion]:
        """Get similar context suggestions"""
        suggestions = []

        try:
            # Search for similar historical memories
            memories = self.memory_engine.recall_memories(current_context, limit=50)

            if memories:
                memory_dicts = [memory.to_dict() for memory in memories]
                similar_results = self.search_engine.search_memories(
                    current_context, memory_dicts,
                    top_k=self.prompt_rules['similar_context']['max_suggestions'],
                    min_similarity=self.prompt_rules['similar_context']['threshold']
                )

                for memory_dict, similarity in similar_results:
                    suggestion = Suggestion(
                        type='memory',
                        title=f"Similar scenario memory (similarity: {similarity:.2f})",
                        content=f"Previously in similar situation: {memory_dict['content'][:100]}...",
                        confidence=similarity,
                        priority=8,
                        metadata={
                            'memory_id': memory_dict['id'],
                            'memory_type': memory_dict['memory_type'],
                            'similarity': similarity
                        }
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get similar context suggestions: {e}")

        return suggestions

    def _get_forgotten_memory_suggestions(self, context_info: Optional[Dict[str, Any]]) -> List[Suggestion]:
        """Get forgotten important memory suggestions"""
        suggestions = []

        try:
            # Get all memories
            memories = self.memory_engine.memory_repo.search_memories(limit=1000)

            # Find important but long-unaccessed memories
            cutoff_date = datetime.now() - timedelta(days=self.prompt_rules['forgotten_memories']['days_threshold'])

            for memory in memories:
                if (memory.importance >= self.prompt_rules['forgotten_memories']['importance_threshold'] and
                    memory.last_accessed and memory.last_accessed < cutoff_date):

                    days_forgotten = (datetime.now() - memory.last_accessed).days

                    suggestion = Suggestion(
                        type='reminder',
                        title=f"Important memory reminder ({days_forgotten} days unaccessed)",
                        content=f"You have an important memory that may need review: {memory.content[:80]}...",
                        confidence=memory.importance,
                        priority=7,
                        metadata={
                            'memory_id': memory.id,
                            'days_forgotten': days_forgotten,
                            'importance': memory.importance
                        }
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get forgotten memory suggestions: {e}")

        return suggestions

    def _get_pattern_suggestions(self, current_context: str,
                               context_info: Optional[Dict[str, Any]]) -> List[Suggestion]:
        """Get pattern recognition suggestions"""
        suggestions = []

        try:
            # Analyze repeated patterns
            memories = self.memory_engine.memory_repo.search_memories(limit=500)

            # Simple keyword frequency analysis
            keyword_counts = {}
            for memory in memories:
                words = memory.content.lower().split()
                for word in words:
                    if len(word) > 3:  # Ignore short words
                        keyword_counts[word] = keyword_counts.get(word, 0) + 1

            # Find high-frequency keywords
            frequent_keywords = [
                word for word, count in keyword_counts.items()
                if count >= self.prompt_rules['pattern_recognition']['min_occurrences']
            ]

            # Check if current context contains these patterns
            current_words = current_context.lower().split()
            matching_patterns = [word for word in frequent_keywords if word in current_words]

            for pattern in matching_patterns[:3]:  # Limit quantity
                count = keyword_counts[pattern]
                suggestion = Suggestion(
                    type='pattern',
                    title=f"Repeated pattern recognition: '{pattern}'",
                    content=f"You often mention '{pattern}' ({count} times), this may be an important topic.",
                    confidence=min(count / 10.0, 1.0),  # Normalize confidence
                    priority=5,
                    metadata={
                        'pattern': pattern,
                        'frequency': count
                    }
                )
                suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get pattern suggestions: {e}")

        return suggestions

    def _get_project_context_suggestions(self, context_info: Dict[str, Any]) -> List[Suggestion]:
        """Get project context suggestions"""
        suggestions = []

        try:
            project_path = context_info.get('project_path')
            if not project_path:
                return suggestions

            # Get project-related memories
            project_memories = self.memory_engine.get_project_memories(project_path, limit=20)

            if project_memories:
                # Sort by importance
                project_memories.sort(key=lambda x: x.importance, reverse=True)

                # Recommend most important project memories
                for memory in project_memories[:3]:
                    suggestion = Suggestion(
                        type='memory',
                        title=f"Project-related memory (importance: {memory.importance:.2f})",
                        content=f"Project memory: {memory.content[:80]}...",
                        confidence=memory.importance,
                        priority=6,
                        metadata={
                            'memory_id': memory.id,
                            'project_path': project_path,
                            'memory_type': memory.memory_type
                        }
                    )
                    suggestions.append(suggestion)

            # Check if project snapshot needs to be created
            if len(project_memories) > 10:
                suggestion = Suggestion(
                    type='optimization',
                    title="Project snapshot suggestion",
                    content=f"Project has {len(project_memories)} memories, suggest creating context snapshot to optimize performance.",
                    confidence=0.8,
                    priority=4,
                    metadata={
                        'project_path': project_path,
                        'memory_count': len(project_memories)
                    }
                )
                suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get project context suggestions: {e}")

        return suggestions

    def _get_learning_reinforcement_suggestions(self) -> List[Suggestion]:
        """Get learning reinforcement suggestions"""
        suggestions = []

        try:
            # Find learning type memories
            learning_memories = self.memory_engine.memory_repo.search_memories(
                memory_type='learning', limit=50
            )

            review_cutoff = datetime.now() - timedelta(
                days=self.prompt_rules['learning_reinforcement']['review_interval_days']
            )

            for memory in learning_memories:
                if (memory.last_accessed and memory.last_accessed < review_cutoff and
                    memory.importance > 0.5):

                    days_since_review = (datetime.now() - memory.last_accessed).days

                    suggestion = Suggestion(
                        type='reminder',
                        title=f"Learning review reminder ({days_since_review} days ago)",
                        content=f"Suggest reviewing this learning content: {memory.content[:80]}...",
                        confidence=memory.importance * 0.8,  # Slightly lower confidence
                        priority=3,
                        metadata={
                            'memory_id': memory.id,
                            'days_since_review': days_since_review,
                            'learning_type': True
                        }
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get learning reinforcement suggestions: {e}")

        return suggestions

    def get_memory_optimization_suggestions(self) -> List[Suggestion]:
        """Get memory optimization suggestions"""
        suggestions = []

        try:
            # Get system statistics
            stats = self.memory_engine.get_statistics()

            # Storage space suggestions
            total_size_mb = stats['db_size_mb'] + stats['fs_size_mb']
            max_size_mb = self.config.storage.max_size_gb * 1024

            if total_size_mb > max_size_mb * 0.8:  # Over 80%
                suggestion = Suggestion(
                    type='optimization',
                    title="Storage space warning",
                    content=f"Storage usage at {total_size_mb/max_size_mb*100:.1f}%, suggest cleaning old memories.",
                    confidence=1.0,
                    priority=9,
                    metadata={
                        'current_size_mb': total_size_mb,
                        'max_size_mb': max_size_mb,
                        'usage_percent': total_size_mb/max_size_mb*100
                    }
                )
                suggestions.append(suggestion)

            # Forgetting suggestions
            if stats['total_memories'] > 0:
                memories = self.memory_engine.memory_repo.search_memories(limit=1000)
                forgetting_stats = self.forgetting_manager.get_forgetting_statistics(memories)

                if forgetting_stats['forgetting_candidates'] > 10:
                    suggestion = Suggestion(
                        type='optimization',
                        title="Memory cleanup suggestion",
                        content=f"Found {forgetting_stats['forgetting_candidates']} cleanable memories, suggest executing cleanup operation.",
                        confidence=0.9,
                        priority=6,
                        metadata={
                            'forgetting_candidates': forgetting_stats['forgetting_candidates'],
                            'total_memories': stats['total_memories']
                        }
                    )
                    suggestions.append(suggestion)

            # Backup suggestions
            if stats['total_memories'] > 100:
                suggestion = Suggestion(
                    type='optimization',
                    title="Data backup suggestion",
                    content=f"You have {stats['total_memories']} memories, suggest regular backup of important data.",
                    confidence=0.7,
                    priority=2,
                    metadata={
                        'total_memories': stats['total_memories']
                    }
                )
                suggestions.append(suggestion)

        except Exception as e:
            self.logger.warning(f"Failed to get optimization suggestions: {e}")

        return suggestions

    def format_suggestions_for_display(self, suggestions: List[Suggestion]) -> str:
        """
        Format suggestions for display

        Args:
            suggestions: List of suggestions

        Returns:
            Formatted suggestion text
        """
        if not suggestions:
            return "No intelligent suggestions available."

        lines = ["🤖 Intelligent Suggestions:"]

        # Group by type
        suggestion_groups = {}
        for suggestion in suggestions:
            if suggestion.type not in suggestion_groups:
                suggestion_groups[suggestion.type] = []
            suggestion_groups[suggestion.type].append(suggestion)

        # Type icon mapping
        type_icons = {
            'memory': '📚',
            'reminder': '⏰',
            'pattern': '🔍',
            'optimization': '⚡'
        }

        for suggestion_type, group_suggestions in suggestion_groups.items():
            icon = type_icons.get(suggestion_type, '💡')
            lines.append(f"\n{icon} {suggestion_type.title()}:")

            for i, suggestion in enumerate(group_suggestions[:3], 1):  # Max 3 per group
                confidence_bar = "●" * int(suggestion.confidence * 5)
                lines.append(f"  {i}. {suggestion.title}")
                lines.append(f"     {suggestion.content}")
                lines.append(f"     Confidence: {confidence_bar} ({suggestion.confidence:.2f})")

        return "\n".join(lines)

    def get_quick_suggestions(self, query: str) -> List[str]:
        """
        Get quick suggestions (for autocomplete, etc.)

        Args:
            query: Query string

        Returns:
            List of quick suggestions
        """
        suggestions = []

        try:
            # Quick suggestions based on historical memories
            memories = self.memory_engine.recall_memories(query, limit=5)

            for memory in memories:
                # Extract key phrases as suggestions
                content_words = memory.content.split()
                if len(content_words) > 3:
                    # Take first few words as suggestion
                    suggestion = " ".join(content_words[:5])
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)

            # Limit number of suggestions
            return suggestions[:10]

        except Exception as e:
            self.logger.warning(f"Failed to get quick suggestions: {e}")
            return []

    def update_prompt_rules(self, rule_name: str, enabled: bool) -> bool:
        """
        Update prompt rules

        Args:
            rule_name: Rule name
            enabled: Whether to enable

        Returns:
            Whether update was successful
        """
        try:
            if rule_name in self.prompt_rules:
                self.prompt_rules[rule_name]['enabled'] = enabled
                self.logger.info(f"Prompt rule '{rule_name}' {'enabled' if enabled else 'disabled'}")
                return True
            else:
                self.logger.warning(f"Prompt rule not found: {rule_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to update prompt rule: {e}")
            return False

    def get_prompt_stats(self) -> Dict[str, Any]:
        """
        Get prompt system statistics

        Returns:
            Statistics information
        """
        return {
            'total_rules': len(self.prompt_rules),
            'enabled_rules': len([r for r in self.prompt_rules.values() if r['enabled']]),
            'rules_detail': self.prompt_rules.copy()
        }