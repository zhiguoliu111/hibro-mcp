#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Injector Module
Intelligently injects relevant memories into Claude conversations to provide contextual support
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ..core.memory_engine import MemoryEngine
from ..intelligence import SimilarityCalculator, SemanticSearchEngine, ImportanceScorer
from ..utils.config import Config


class MemoryInjector:
    """Memory Injector"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize Memory Injector

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.memory_injector')

        # Initialize components
        self.similarity_calc = SimilarityCalculator()
        self.search_engine = SemanticSearchEngine(self.similarity_calc)
        self.importance_scorer = ImportanceScorer(config)

        # Injection strategy configuration
        self.injection_config = {
            'max_context_kb': config.ide_integration.context_limit_kb,
            'auto_inject': config.ide_integration.auto_inject,
            'relevance_threshold': 0.4,
            'importance_threshold': 0.3,
            'max_memories_per_type': {
                'preference': 5,
                'decision': 8,
                'project': 10,
                'important': 3,
                'learning': 6,
                'conversation': 15
            },
            'context_layers': {
                'essential': {'size_kb': 50, 'min_importance': 0.7},
                'relevant': {'size_kb': 100, 'min_importance': 0.4},
                'supplementary': {'size_kb': 50, 'min_importance': 0.2}
            }
        }

    def inject_contextual_memories(self, current_input: str,
                                 context_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject relevant memory context for current input

        Args:
            current_input: Current user input
            context_info: Context information (project path, file type, etc.)

        Returns:
            Complete input with memory context
        """
        if not self.injection_config['auto_inject']:
            return current_input

        try:
            # Get relevant memories
            relevant_memories = self._get_relevant_memories(current_input, context_info)

            if not relevant_memories:
                return current_input

            # Build context
            context_text = self._build_context_text(relevant_memories, current_input)

            # Check size limit
            if self._calculate_text_size_kb(context_text) > self.injection_config['max_context_kb']:
                context_text = self._optimize_context_size(context_text, relevant_memories)

            # Combine final input
            injected_input = self._combine_context_and_input(context_text, current_input)

            self.logger.info(f"Injected {len(relevant_memories)} relevant memories")
            return injected_input

        except Exception as e:
            self.logger.error(f"Memory injection failed: {e}")
            return current_input

    def _get_relevant_memories(self, current_input: str,
                             context_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get relevant memories

        Args:
            current_input: Current input
            context_info: Context information

        Returns:
            List of relevant memories
        """
        all_memories = []

        try:
            # 1. Semantic similarity search
            semantic_memories = self._get_semantic_similar_memories(current_input)
            all_memories.extend(semantic_memories)

            # 2. Project-related memories
            if context_info and context_info.get('project_path'):
                project_memories = self._get_project_memories(context_info['project_path'])
                all_memories.extend(project_memories)

            # 3. Important preference memories
            preference_memories = self._get_preference_memories()
            all_memories.extend(preference_memories)

            # 4. Recent decision memories
            recent_decisions = self._get_recent_decisions()
            all_memories.extend(recent_decisions)

            # Deduplicate and sort by importance
            unique_memories = self._deduplicate_memories(all_memories)
            sorted_memories = self._sort_memories_by_relevance(unique_memories, current_input)

            # Filter by type limits
            filtered_memories = self._filter_memories_by_type_limits(sorted_memories)

            return filtered_memories

        except Exception as e:
            self.logger.error(f"Failed to get relevant memories: {e}")
            return []

    def _get_semantic_similar_memories(self, query: str) -> List[Dict[str, Any]]:
        """Get semantically similar memories"""
        try:
            memories = self.memory_engine.recall_memories(query, limit=50)
            memory_dicts = [memory.to_dict() for memory in memories]

            # Use semantic search
            similar_results = self.search_engine.search_memories(
                query, memory_dicts,
                top_k=20,
                min_similarity=self.injection_config['relevance_threshold']
            )

            # Add similarity information
            for memory_dict, similarity in similar_results:
                memory_dict['_similarity'] = similarity
                memory_dict['_source'] = 'semantic'

            return [memory_dict for memory_dict, _ in similar_results]

        except Exception as e:
            self.logger.warning(f"Semantic search failed: {e}")
            return []

    def _get_project_memories(self, project_path: str) -> List[Dict[str, Any]]:
        """Get project-related memories"""
        try:
            project_memories = self.memory_engine.get_project_memories(project_path, limit=30)

            memory_dicts = []
            for memory in project_memories:
                memory_dict = memory.to_dict()
                memory_dict['_source'] = 'project'
                memory_dict['_relevance'] = memory.importance
                memory_dicts.append(memory_dict)

            return memory_dicts

        except Exception as e:
            self.logger.warning(f"Failed to get project memories: {e}")
            return []

    def _get_preference_memories(self) -> List[Dict[str, Any]]:
        """Get preference memories"""
        try:
            preference_memories = self.memory_engine.memory_repo.search_memories(
                memory_type='preference',
                min_importance=self.injection_config['importance_threshold'],
                limit=10
            )

            memory_dicts = []
            for memory in preference_memories:
                memory_dict = memory.to_dict()
                memory_dict['_source'] = 'preference'
                memory_dict['_relevance'] = memory.importance
                memory_dicts.append(memory_dict)

            return memory_dicts

        except Exception as e:
            self.logger.warning(f"Failed to get preference memories: {e}")
            return []

    def _get_recent_decisions(self) -> List[Dict[str, Any]]:
        """Get recent decision memories"""
        try:
            # Get decision memories from the last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)

            decision_memories = self.memory_engine.memory_repo.search_memories(
                memory_type='decision',
                min_importance=0.5,
                limit=15
            )

            # Filter recent memories
            recent_memories = [
                memory for memory in decision_memories
                if memory.created_at and memory.created_at >= cutoff_date
            ]

            memory_dicts = []
            for memory in recent_memories:
                memory_dict = memory.to_dict()
                memory_dict['_source'] = 'recent_decision'
                memory_dict['_relevance'] = memory.importance
                memory_dicts.append(memory_dict)

            return memory_dicts

        except Exception as e:
            self.logger.warning(f"Failed to get recent decisions: {e}")
            return []

    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate memories"""
        seen_ids = set()
        unique_memories = []

        for memory in memories:
            memory_id = memory.get('id')
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_memories.append(memory)

        return unique_memories

    def _sort_memories_by_relevance(self, memories: List[Dict[str, Any]],
                                  query: str) -> List[Dict[str, Any]]:
        """Sort memories by relevance"""
        def calculate_relevance_score(memory: Dict[str, Any]) -> float:
            # Base importance
            importance = memory.get('importance', 0.0)

            # Similarity bonus
            similarity = memory.get('_similarity', 0.0)

            # Source weight
            source_weights = {
                'semantic': 1.0,
                'project': 0.8,
                'preference': 0.9,
                'recent_decision': 0.7
            }
            source_weight = source_weights.get(memory.get('_source', ''), 0.5)

            # Time decay
            time_factor = self._calculate_time_factor(memory)

            # Composite score
            return (importance * 0.4 + similarity * 0.3 + source_weight * 0.2 + time_factor * 0.1)

        return sorted(memories, key=calculate_relevance_score, reverse=True)

    def _calculate_time_factor(self, memory: Dict[str, Any]) -> float:
        """Calculate time factor"""
        try:
            if 'last_accessed' in memory and memory['last_accessed']:
                last_accessed = datetime.fromisoformat(memory['last_accessed'])
                days_ago = (datetime.now() - last_accessed).days

                # Recently accessed memories have higher weight
                if days_ago <= 1:
                    return 1.0
                elif days_ago <= 7:
                    return 0.8
                elif days_ago <= 30:
                    return 0.6
                else:
                    return 0.4

            return 0.5

        except Exception:
            return 0.5

    def _filter_memories_by_type_limits(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter memories by type limits"""
        type_counts = {}
        filtered_memories = []

        for memory in memories:
            memory_type = memory.get('memory_type', 'conversation')
            current_count = type_counts.get(memory_type, 0)
            max_count = self.injection_config['max_memories_per_type'].get(memory_type, 5)

            if current_count < max_count:
                filtered_memories.append(memory)
                type_counts[memory_type] = current_count + 1

        return filtered_memories

    def _build_context_text(self, memories: List[Dict[str, Any]], current_input: str) -> str:
        """Build context text"""
        if not memories:
            return ""

        context_parts = []
        context_parts.append("# Relevant Memory Context")
        context_parts.append("")

        # Group by type
        memory_groups = {}
        for memory in memories:
            memory_type = memory.get('memory_type', 'conversation')
            if memory_type not in memory_groups:
                memory_groups[memory_type] = []
            memory_groups[memory_type].append(memory)

        # Type display name mapping
        type_names = {
            'preference': 'Personal Preferences',
            'decision': 'Technical Decisions',
            'project': 'Project Related',
            'important': 'Important Information',
            'learning': 'Learning Content',
            'conversation': 'Historical Conversations'
        }

        # Sort types by priority
        type_priority = ['preference', 'important', 'decision', 'project', 'learning', 'conversation']

        for memory_type in type_priority:
            if memory_type in memory_groups:
                group_memories = memory_groups[memory_type]
                type_name = type_names.get(memory_type, memory_type)

                context_parts.append(f"## {type_name}")

                for i, memory in enumerate(group_memories[:5], 1):  # Max 5 per group
                    content = memory.get('content', '')
                    importance = memory.get('importance', 0.0)

                    # Truncate overly long content
                    if len(content) > 150:
                        content = content[:150] + "..."

                    context_parts.append(f"{i}. [{importance:.2f}] {content}")

                context_parts.append("")

        context_parts.append("---")
        context_parts.append("")

        return "\n".join(context_parts)

    def _calculate_text_size_kb(self, text: str) -> float:
        """Calculate text size (KB)"""
        return len(text.encode('utf-8')) / 1024

    def _optimize_context_size(self, context_text: str, memories: List[Dict[str, Any]]) -> str:
        """Optimize context size"""
        # Layered strategy: Keep most important memories
        layers = self.injection_config['context_layers']

        # Layer by importance
        essential_memories = []
        relevant_memories = []
        supplementary_memories = []

        for memory in memories:
            importance = memory.get('importance', 0.0)

            if importance >= layers['essential']['min_importance']:
                essential_memories.append(memory)
            elif importance >= layers['relevant']['min_importance']:
                relevant_memories.append(memory)
            else:
                supplementary_memories.append(memory)

        # Build context layer by layer
        optimized_context = ""
        current_size = 0
        max_size = self.injection_config['max_context_kb']

        # Essential layer
        essential_text = self._build_context_text(essential_memories[:10], "")
        essential_size = self._calculate_text_size_kb(essential_text)

        if essential_size <= layers['essential']['size_kb']:
            optimized_context += essential_text
            current_size += essential_size

        # Relevant layer
        if current_size < max_size * 0.8:  # Keep 20% buffer
            remaining_size = max_size - current_size
            relevant_count = min(len(relevant_memories), int(remaining_size / 5))  # Estimate 5KB per memory

            relevant_text = self._build_context_text(relevant_memories[:relevant_count], "")
            relevant_size = self._calculate_text_size_kb(relevant_text)

            if current_size + relevant_size <= max_size:
                optimized_context += relevant_text
                current_size += relevant_size

        return optimized_context

    def _combine_context_and_input(self, context_text: str, current_input: str) -> str:
        """Combine context and user input"""
        if not context_text:
            return current_input

        combined_parts = []

        # Add context description
        combined_parts.append("<!-- hibro Smart Memory Context -->")
        combined_parts.append(context_text)

        # Add user input
        combined_parts.append("# Current Request")
        combined_parts.append(current_input)

        return "\n".join(combined_parts)

    def create_project_context_snapshot(self, project_path: str) -> str:
        """
        Create context snapshot for project

        Args:
            project_path: Project path

        Returns:
            Project context snapshot text
        """
        try:
            # Get project-related memories
            project_memories = self.memory_engine.get_project_memories(project_path, limit=50)

            if not project_memories:
                return f"# {Path(project_path).name} Project Context\n\nNo relevant memories yet."

            # Group by type
            memory_groups = {}
            for memory in project_memories:
                memory_type = memory.memory_type
                if memory_type not in memory_groups:
                    memory_groups[memory_type] = []
                memory_groups[memory_type].append(memory)

            # Build snapshot
            snapshot_parts = []
            snapshot_parts.append(f"# {Path(project_path).name} Project Context Snapshot")
            snapshot_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            snapshot_parts.append(f"Project Path: {project_path}")
            snapshot_parts.append(f"Total Memories: {len(project_memories)}")
            snapshot_parts.append("")

            # Statistics
            avg_importance = sum(m.importance for m in project_memories) / len(project_memories)
            snapshot_parts.append(f"Average Importance: {avg_importance:.2f}")
            snapshot_parts.append("")

            # Display memories by type
            type_names = {
                'preference': 'Preferences',
                'decision': 'Technical Decisions',
                'project': 'Project Information',
                'important': 'Important Records',
                'learning': 'Learning Notes',
                'conversation': 'Discussion Records'
            }

            for memory_type, memories in memory_groups.items():
                type_name = type_names.get(memory_type, memory_type)
                snapshot_parts.append(f"## {type_name} ({len(memories)} items)")

                # Sort by importance
                sorted_memories = sorted(memories, key=lambda x: x.importance, reverse=True)

                for i, memory in enumerate(sorted_memories[:10], 1):  # Max 10 per type
                    content = memory.content
                    if len(content) > 100:
                        content = content[:100] + "..."

                    snapshot_parts.append(f"{i}. [{memory.importance:.2f}] {content}")

                snapshot_parts.append("")

            return "\n".join(snapshot_parts)

        except Exception as e:
            self.logger.error(f"Failed to create project context snapshot: {e}")
            return f"# Project Context Snapshot\n\nCreation failed: {e}"

    def get_injection_stats(self) -> Dict[str, Any]:
        """
        Get injection statistics

        Returns:
            Injection statistics
        """
        try:
            stats = self.memory_engine.get_statistics()

            return {
                'auto_inject_enabled': self.injection_config['auto_inject'],
                'max_context_kb': self.injection_config['max_context_kb'],
                'total_memories': stats['total_memories'],
                'relevance_threshold': self.injection_config['relevance_threshold'],
                'importance_threshold': self.injection_config['importance_threshold'],
                'type_limits': self.injection_config['max_memories_per_type']
            }

        except Exception as e:
            self.logger.error(f"Failed to get injection stats: {e}")
            return {}

    def update_injection_config(self, **kwargs) -> bool:
        """
        Update injection configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.injection_config:
                    self.injection_config[key] = value
                    self.logger.info(f"Injection config updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update injection config: {e}")
            return False