#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Optimizer Module
Implements intelligent context optimization algorithms to improve memory retrieval and injection efficiency
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

from ..core.memory_engine import MemoryEngine
from ..intelligence import SimilarityCalculator, ForgettingManager, ImportanceScorer
from ..storage.models import Memory
from ..utils.config import Config


@dataclass
class ContextWindow:
    """Context Window"""
    memories: List[Memory]
    total_size_kb: float
    relevance_score: float
    diversity_score: float
    freshness_score: float
    overall_score: float


@dataclass
class OptimizationMetrics:
    """Optimization Metrics"""
    compression_ratio: float
    relevance_improvement: float
    diversity_index: float
    processing_time_ms: float
    memory_count_reduction: int


class ContextOptimizer:
    """Context Optimizer"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize Context Optimizer

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.context_optimizer')

        # Initialize components
        self.similarity_calc = SimilarityCalculator()
        self.forgetting_manager = ForgettingManager(config)
        self.importance_scorer = ImportanceScorer(config)

        # Optimization configuration
        self.optimization_config = {
            'max_context_kb': config.ide_integration.context_limit_kb,
            'target_compression_ratio': 0.7,  # Target compression ratio
            'min_relevance_threshold': 0.3,
            'diversity_weight': 0.2,
            'freshness_weight': 0.15,
            'importance_weight': 0.4,
            'relevance_weight': 0.25,
            'max_similar_memories': 3,  # Maximum similar memories
            'clustering_threshold': 0.8,  # Clustering threshold
            'adaptive_threshold': True,  # Adaptive threshold
        }

        # Cache
        self._similarity_cache = {}
        self._cluster_cache = {}

    def optimize_context(self, memories: List[Memory], query: str,
                        target_size_kb: Optional[float] = None) -> ContextWindow:
        """
        Optimize context window

        Args:
            memories: List of candidate memories
            query: Query content
            target_size_kb: Target size (KB)

        Returns:
            Optimized context window
        """
        start_time = datetime.now()

        try:
            if not memories:
                return ContextWindow([], 0.0, 0.0, 0.0, 0.0, 0.0)

            # Set target size
            if target_size_kb is None:
                target_size_kb = self.optimization_config['max_context_kb']

            # 1. Calculate memory relevance
            memories_with_scores = self._calculate_relevance_scores(memories, query)

            # 2. Deduplicate and cluster
            clustered_memories = self._cluster_similar_memories(memories_with_scores)

            # 3. Multi-objective optimization selection
            selected_memories = self._multi_objective_selection(
                clustered_memories, query, target_size_kb
            )

            # 4. Build optimized context window
            context_window = self._build_context_window(selected_memories, query)

            # 5. Log optimization metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_optimization_metrics(memories, context_window, processing_time)

            return context_window

        except Exception as e:
            self.logger.error(f"Context optimization failed: {e}")
            # Return simple truncated version
            return self._fallback_optimization(memories, target_size_kb or self.optimization_config['max_context_kb'])

    def _calculate_relevance_scores(self, memories: List[Memory], query: str) -> List[Tuple[Memory, Dict[str, float]]]:
        """
        Calculate memory relevance scores

        Args:
            memories: Memory list
            query: Query content

        Returns:
            Memory list with scores
        """
        memories_with_scores = []

        # Get query embedding
        query_embedding = self.similarity_calc.get_embedding(query)

        for memory in memories:
            scores = {}

            # 1. Semantic similarity
            memory_embedding = self.similarity_calc.get_embedding(memory.content)
            scores['similarity'] = self.similarity_calc._cosine_similarity(query_embedding, memory_embedding)

            # 2. Importance score
            scores['importance'] = memory.importance

            # 3. Freshness score
            scores['freshness'] = self._calculate_freshness_score(memory)

            # 4. Access frequency score
            scores['frequency'] = self._calculate_frequency_score(memory)

            # 5. Type weight
            scores['type_weight'] = self._get_type_weight(memory.memory_type)

            # 6. Composite relevance score
            scores['relevance'] = self._calculate_composite_relevance(scores)

            memories_with_scores.append((memory, scores))

        # Sort by relevance
        memories_with_scores.sort(key=lambda x: x[1]['relevance'], reverse=True)

        return memories_with_scores

    def _calculate_freshness_score(self, memory: Memory) -> float:
        """Calculate freshness score"""
        if not memory.last_accessed:
            return 0.5

        days_since_access = (datetime.now() - memory.last_accessed).days

        # Use exponential decay
        return math.exp(-0.1 * days_since_access)

    def _calculate_frequency_score(self, memory: Memory) -> float:
        """Calculate access frequency score"""
        if not memory.created_at:
            return 0.5

        days_since_creation = max(1, (datetime.now() - memory.created_at).days)
        frequency = memory.access_count / days_since_creation

        # Use logarithmic function for smoothing
        return math.log(1 + frequency * 10) / math.log(11)

    def _get_type_weight(self, memory_type: str) -> float:
        """Get memory type weight"""
        type_weights = {
            'important': 1.0,
            'preference': 0.9,
            'decision': 0.8,
            'project': 0.7,
            'learning': 0.6,
            'conversation': 0.4
        }
        return type_weights.get(memory_type, 0.5)

    def _calculate_composite_relevance(self, scores: Dict[str, float]) -> float:
        """Calculate composite relevance score"""
        config = self.optimization_config

        return (
            scores['similarity'] * config['relevance_weight'] +
            scores['importance'] * config['importance_weight'] +
            scores['freshness'] * config['freshness_weight'] +
            scores['frequency'] * 0.1 +
            scores['type_weight'] * 0.1
        )

    def _cluster_similar_memories(self, memories_with_scores: List[Tuple[Memory, Dict[str, float]]]) -> List[List[Tuple[Memory, Dict[str, float]]]]:
        """
        Cluster similar memories

        Args:
            memories_with_scores: List of memories with scores

        Returns:
            Clustered memory groups
        """
        if len(memories_with_scores) <= 1:
            return [memories_with_scores]

        clusters = []
        used_indices = set()

        for i, (memory1, scores1) in enumerate(memories_with_scores):
            if i in used_indices:
                continue

            # Create new cluster
            cluster = [(memory1, scores1)]
            used_indices.add(i)

            # Find similar memories
            for j, (memory2, scores2) in enumerate(memories_with_scores[i+1:], i+1):
                if j in used_indices:
                    continue

                # Calculate similarity
                similarity = self._get_memory_similarity(memory1, memory2)

                if similarity >= self.optimization_config['clustering_threshold']:
                    cluster.append((memory2, scores2))
                    used_indices.add(j)

                    # Limit cluster size
                    if len(cluster) >= self.optimization_config['max_similar_memories']:
                        break

            clusters.append(cluster)

        self.logger.debug(f"Clustering result: {len(memories_with_scores)} memories -> {len(clusters)} clusters")
        return clusters

    def _get_memory_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """Get similarity between two memories"""
        # Use cache
        cache_key = (memory1.id, memory2.id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Calculate similarity
        similarity = self.similarity_calc.calculate_similarity(memory1.content, memory2.content)

        # Cache result
        self._similarity_cache[cache_key] = similarity
        return similarity

    def _multi_objective_selection(self, clusters: List[List[Tuple[Memory, Dict[str, float]]]],
                                 query: str, target_size_kb: float) -> List[Memory]:
        """
        Multi-objective optimization memory selection

        Args:
            clusters: Clustered memory groups
            query: Query content
            target_size_kb: Target size

        Returns:
            Selected memory list
        """
        selected_memories = []
        current_size_kb = 0.0

        # Select best representative for each cluster
        cluster_representatives = []
        for cluster in clusters:
            # Select most relevant memory in cluster as representative
            best_memory, best_scores = max(cluster, key=lambda x: x[1]['relevance'])
            cluster_representatives.append((best_memory, best_scores, len(cluster)))

        # Sort by relevance
        cluster_representatives.sort(key=lambda x: x[1]['relevance'], reverse=True)

        # Greedy selection considering multiple objectives
        for memory, scores, cluster_size in cluster_representatives:
            memory_size_kb = self._estimate_memory_size_kb(memory)

            # Check size constraint
            if current_size_kb + memory_size_kb > target_size_kb:
                # Try adaptive threshold
                if self.optimization_config['adaptive_threshold']:
                    if self._should_include_despite_size(memory, scores, selected_memories):
                        selected_memories.append(memory)
                        current_size_kb += memory_size_kb
                break

            # Check relevance threshold
            if scores['relevance'] < self.optimization_config['min_relevance_threshold']:
                break

            # Check diversity
            if self._maintains_diversity(memory, selected_memories):
                selected_memories.append(memory)
                current_size_kb += memory_size_kb

        return selected_memories

    def _estimate_memory_size_kb(self, memory: Memory) -> float:
        """Estimate memory size (KB)"""
        # Simple estimate: content length + metadata
        content_size = len(memory.content.encode('utf-8'))
        metadata_size = 200  # Estimate metadata size

        return (content_size + metadata_size) / 1024

    def _should_include_despite_size(self, memory: Memory, scores: Dict[str, float],
                                   selected_memories: List[Memory]) -> bool:
        """Determine if memory should be included (even if exceeds size limit)"""
        # If very important memory, can exceed limit
        if scores['importance'] >= 0.9 and scores['relevance'] >= 0.8:
            return True

        # If only memory of certain type
        memory_types = {m.memory_type for m in selected_memories}
        if memory.memory_type not in memory_types and memory.memory_type in ['important', 'preference']:
            return True

        return False

    def _maintains_diversity(self, candidate_memory: Memory, selected_memories: List[Memory]) -> bool:
        """Check if maintains diversity"""
        if not selected_memories:
            return True

        # Check type diversity
        selected_types = [m.memory_type for m in selected_memories]
        type_count = selected_types.count(candidate_memory.memory_type)

        # Limit same type memory count
        type_limits = {
            'conversation': 5,
            'learning': 3,
            'project': 4,
            'decision': 3,
            'preference': 2,
            'important': 2
        }

        max_count = type_limits.get(candidate_memory.memory_type, 3)
        if type_count >= max_count:
            return False

        # Check content diversity
        for selected_memory in selected_memories[-3:]:  # Only check recent ones
            similarity = self._get_memory_similarity(candidate_memory, selected_memory)
            if similarity > 0.9:  # Too similar content
                return False

        return True

    def _build_context_window(self, memories: List[Memory], query: str) -> ContextWindow:
        """Build context window"""
        if not memories:
            return ContextWindow([], 0.0, 0.0, 0.0, 0.0, 0.0)

        # Calculate total size
        total_size_kb = sum(self._estimate_memory_size_kb(m) for m in memories)

        # Calculate various scores
        relevance_score = self._calculate_window_relevance(memories, query)
        diversity_score = self._calculate_window_diversity(memories)
        freshness_score = self._calculate_window_freshness(memories)

        # Calculate overall score
        overall_score = (
            relevance_score * self.optimization_config['relevance_weight'] +
            diversity_score * self.optimization_config['diversity_weight'] +
            freshness_score * self.optimization_config['freshness_weight'] +
            self._calculate_window_importance(memories) * self.optimization_config['importance_weight']
        )

        return ContextWindow(
            memories=memories,
            total_size_kb=total_size_kb,
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            freshness_score=freshness_score,
            overall_score=overall_score
        )

    def _calculate_window_relevance(self, memories: List[Memory], query: str) -> float:
        """Calculate window relevance score"""
        if not memories:
            return 0.0

        query_embedding = self.similarity_calc.get_embedding(query)
        similarities = []

        for memory in memories:
            memory_embedding = self.similarity_calc.get_embedding(memory.content)
            similarity = self.similarity_calc._cosine_similarity(query_embedding, memory_embedding)
            similarities.append(similarity)

        return sum(similarities) / len(similarities)

    def _calculate_window_diversity(self, memories: List[Memory]) -> float:
        """Calculate window diversity score"""
        if len(memories) <= 1:
            return 1.0

        # Type diversity
        types = [m.memory_type for m in memories]
        type_diversity = len(set(types)) / len(types)

        # Content diversity
        total_similarity = 0.0
        pair_count = 0

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                similarity = self._get_memory_similarity(memories[i], memories[j])
                total_similarity += similarity
                pair_count += 1

        content_diversity = 1.0 - (total_similarity / pair_count) if pair_count > 0 else 1.0

        return (type_diversity + content_diversity) / 2

    def _calculate_window_freshness(self, memories: List[Memory]) -> float:
        """Calculate window freshness score"""
        if not memories:
            return 0.0

        freshness_scores = [self._calculate_freshness_score(m) for m in memories]
        return sum(freshness_scores) / len(freshness_scores)

    def _calculate_window_importance(self, memories: List[Memory]) -> float:
        """Calculate window importance score"""
        if not memories:
            return 0.0

        return sum(m.importance for m in memories) / len(memories)

    def _fallback_optimization(self, memories: List[Memory], target_size_kb: float) -> ContextWindow:
        """Fallback optimization (simple truncation)"""
        sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)

        selected_memories = []
        current_size_kb = 0.0

        for memory in sorted_memories:
            memory_size_kb = self._estimate_memory_size_kb(memory)
            if current_size_kb + memory_size_kb <= target_size_kb:
                selected_memories.append(memory)
                current_size_kb += memory_size_kb
            else:
                break

        return ContextWindow(
            memories=selected_memories,
            total_size_kb=current_size_kb,
            relevance_score=0.5,
            diversity_score=0.5,
            freshness_score=0.5,
            overall_score=0.5
        )

    def _log_optimization_metrics(self, original_memories: List[Memory],
                                context_window: ContextWindow, processing_time_ms: float):
        """Log optimization metrics"""
        if not original_memories:
            return

        original_size_kb = sum(self._estimate_memory_size_kb(m) for m in original_memories)
        compression_ratio = context_window.total_size_kb / original_size_kb if original_size_kb > 0 else 0

        metrics = OptimizationMetrics(
            compression_ratio=compression_ratio,
            relevance_improvement=context_window.relevance_score,
            diversity_index=context_window.diversity_score,
            processing_time_ms=processing_time_ms,
            memory_count_reduction=len(original_memories) - len(context_window.memories)
        )

        self.logger.info(
            f"Context optimization complete: Compression={compression_ratio:.2f}, "
            f"Relevance={context_window.relevance_score:.2f}, "
            f"Diversity={context_window.diversity_score:.2f}, "
            f"Processing time={processing_time_ms:.1f}ms"
        )

    def adaptive_threshold_adjustment(self, query_history: List[str],
                                    performance_feedback: List[float]):
        """
        Adaptive threshold adjustment

        Args:
            query_history: Query history
            performance_feedback: Performance feedback scores
        """
        if len(performance_feedback) < 5:
            return

        try:
            # Calculate average performance
            avg_performance = sum(performance_feedback[-10:]) / len(performance_feedback[-10:])

            # Adjust thresholds based on performance
            if avg_performance < 0.6:
                # Poor performance, lower threshold to include more memories
                self.optimization_config['min_relevance_threshold'] *= 0.9
                self.optimization_config['clustering_threshold'] *= 0.95
            elif avg_performance > 0.8:
                # Good performance, raise threshold to improve precision
                self.optimization_config['min_relevance_threshold'] *= 1.05
                self.optimization_config['clustering_threshold'] *= 1.02

            # Limit threshold range
            self.optimization_config['min_relevance_threshold'] = max(0.1, min(0.7, self.optimization_config['min_relevance_threshold']))
            self.optimization_config['clustering_threshold'] = max(0.6, min(0.95, self.optimization_config['clustering_threshold']))

            self.logger.debug(f"Adaptive threshold adjustment: Relevance threshold={self.optimization_config['min_relevance_threshold']:.3f}")

        except Exception as e:
            self.logger.error(f"Adaptive threshold adjustment failed: {e}")

    def clear_cache(self):
        """Clear cache"""
        self._similarity_cache.clear()
        self._cluster_cache.clear()
        self.logger.info("Optimizer cache cleared")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'config': self.optimization_config.copy(),
            'cache_size': {
                'similarity_cache': len(self._similarity_cache),
                'cluster_cache': len(self._cluster_cache)
            }
        }

    def update_optimization_config(self, **kwargs) -> bool:
        """Update optimization configuration"""
        try:
            for key, value in kwargs.items():
                if key in self.optimization_config:
                    self.optimization_config[key] = value
                    self.logger.info(f"Optimization config updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update optimization config: {e}")
            return False