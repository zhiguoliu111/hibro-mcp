#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Relationship Analysis Engine
Implements identification, analysis, and reasoning of causal relationships between memories
"""

import re
import logging
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from datetime import datetime, timedelta

from .similarity import SimilarityCalculator
from ..storage.models import Memory, MemoryRepository
from .bilingual_patterns import BILINGUAL_CAUSAL_PATTERNS


class CausalType(Enum):
    """Causal relationship types"""
    EXPLICIT = "explicit"      # Explicit causal expressions ("because...so...")
    TEMPORAL = "temporal"      # Temporal sequence causality (sequential occurrence)
    SEMANTIC = "semantic"      # Semantic association causality (concept related)
    DECISION = "decision"      # Decision causality (choice leads to result)


@dataclass
class CausalRelation:
    """Causal relationship"""
    cause_memory_id: int
    effect_memory_id: int
    causal_type: CausalType
    strength: float           # Causal strength (0.0-1.0)
    confidence: float         # Confidence level (0.0-1.0)
    evidence: List[str]       # Evidence list
    pattern_matched: str      # Matched pattern
    created_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'cause_memory_id': self.cause_memory_id,
            'effect_memory_id': self.effect_memory_id,
            'causal_type': self.causal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'pattern_matched': self.pattern_matched,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class CausalChain:
    """Causal chain"""
    chain_id: str
    relations: List[CausalRelation]
    total_strength: float
    chain_length: int
    root_cause: int           # Root cause memory ID
    final_effect: int         # Final effect memory ID

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'chain_id': self.chain_id,
            'relations': [r.to_dict() for r in self.relations],
            'total_strength': self.total_strength,
            'chain_length': self.chain_length,
            'root_cause': self.root_cause,
            'final_effect': self.final_effect
        }


class CausalAnalyzer:
    """Causal relationship analysis engine"""

    def __init__(self, similarity_calc: SimilarityCalculator, memory_repo: MemoryRepository):
        """
        Initialize causal analysis engine

        Args:
            similarity_calc: Semantic similarity calculator
            memory_repo: Memory repository
        """
        self.similarity_calc = similarity_calc
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.causal_analyzer')

        # Causal pattern library (bilingual: supports Chinese and English)
        self.causal_patterns = BILINGUAL_CAUSAL_PATTERNS

        # Causal strength weights
        self.strength_weights = {
            CausalType.EXPLICIT: 0.9,    # Explicitly expressed causal relationships are strongest (direct statements)
            CausalType.DECISION: 0.8,    # Decision causal relationships are strong (choice-based)
            CausalType.TEMPORAL: 0.6,    # Temporal sequence causality is medium (time-based)
            CausalType.SEMANTIC: 0.4     # Semantic association causality is weak (inference-based)
        }

    def analyze_causal_chain(self, memories: List[Memory]) -> List[CausalChain]:
        """
        Analyze causal chains between memories

        Args:
            memories: List of memories

        Returns:
            List of causal chains
        """
        self.logger.info(f"Starting to analyze causal chains for {len(memories)} memories")

        # 1. Extract all causal relationships
        relations = []
        for memory in memories:
            memory_relations = self.extract_causal_relations(memory, memories)
            relations.extend(memory_relations)

        self.logger.info(f"Extracted {len(relations)} causal relationships")

        # 2. Build causal graph
        causal_graph = self._build_causal_graph(relations)

        # 3. Identify causal chains
        chains = self._identify_causal_chains(causal_graph, relations)

        self.logger.info(f"Identified {len(chains)} causal chains")
        return chains

    def extract_causal_relations(self, target_memory: Memory, candidate_memories: List[Memory]) -> List[CausalRelation]:
        """
        Extract causal relationships between target memory and candidate memories

        Args:
            target_memory: Target memory
            candidate_memories: List of candidate memories

        Returns:
            List of causal relationships
        """
        relations = []

        # 1. Explicit causal pattern recognition
        explicit_relations = self._extract_explicit_causal_relations(target_memory, candidate_memories)
        relations.extend(explicit_relations)

        # 2. Temporal sequence causal analysis
        temporal_relations = self._extract_temporal_causal_relations(target_memory, candidate_memories)
        relations.extend(temporal_relations)

        # 3. Semantic association causal analysis
        semantic_relations = self._extract_semantic_causal_relations(target_memory, candidate_memories)
        relations.extend(semantic_relations)

        return relations

    def _extract_explicit_causal_relations(self, target_memory: Memory, candidates: List[Memory]) -> List[CausalRelation]:
        """Extract explicit causal relationships"""
        relations = []
        content = target_memory.content

        for pattern_type, patterns in self.causal_patterns.items():
            if pattern_type not in ['explicit', 'decision']:
                continue

            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Extract causal description
                        cause_desc = match.group(1).strip()
                        effect_desc = match.group(2).strip()

                        # Find matching causal objects in candidate memories
                        cause_memory = self._find_matching_memory(cause_desc, candidates)
                        effect_memory = self._find_matching_memory(effect_desc, candidates)

                        if cause_memory and effect_memory and cause_memory.id != effect_memory.id:
                            causal_type = CausalType.EXPLICIT if pattern_type == 'explicit' else CausalType.DECISION
                            strength = self.strength_weights[causal_type]

                            relation = CausalRelation(
                                cause_memory_id=cause_memory.id,
                                effect_memory_id=effect_memory.id,
                                causal_type=causal_type,
                                strength=strength,
                                confidence=0.8,
                                evidence=[match.group(0)],
                                pattern_matched=pattern,
                                created_at=datetime.now()
                            )
                            relations.append(relation)

                except re.error as e:
                    self.logger.warning(f"Regex error: {pattern} - {e}")
                    continue

        return relations

    def _extract_temporal_causal_relations(self, target_memory: Memory, candidates: List[Memory]) -> List[CausalRelation]:
        """Extract temporal sequence causal relationships"""
        relations = []

        # Sort candidate memories by time
        sorted_candidates = sorted(candidates, key=lambda m: m.created_at or datetime.min)
        target_index = next((i for i, m in enumerate(sorted_candidates) if m.id == target_memory.id), -1)

        if target_index == -1:
            return relations

        # Analyze temporal relationships within time window
        time_window = timedelta(hours=24)  # Memories within 24 hours may have causal relationships

        for i, candidate in enumerate(sorted_candidates):
            if i == target_index or not candidate.created_at or not target_memory.created_at:
                continue

            time_diff = abs((candidate.created_at - target_memory.created_at).total_seconds())
            if time_diff > time_window.total_seconds():
                continue

            # Check semantic similarity
            similarity = self.similarity_calc.calculate_similarity(target_memory.content, candidate.content)
            if similarity < 0.3:  # Too low similarity, unlikely to have causal relationship
                continue

            # Determine causal direction (earlier time is cause, later time is effect)
            if candidate.created_at < target_memory.created_at:
                cause_id, effect_id = candidate.id, target_memory.id
            else:
                cause_id, effect_id = target_memory.id, candidate.id

            # Calculate causal strength (based on time interval and similarity)
            time_factor = max(0.1, 1.0 - time_diff / time_window.total_seconds())
            strength = self.strength_weights[CausalType.TEMPORAL] * similarity * time_factor

            relation = CausalRelation(
                cause_memory_id=cause_id,
                effect_memory_id=effect_id,
                causal_type=CausalType.TEMPORAL,
                strength=strength,
                confidence=0.6,
                evidence=[f"Time interval: {time_diff/3600:.1f} hours, Similarity: {similarity:.2f}"],
                pattern_matched="temporal_sequence",
                created_at=datetime.now()
            )
            relations.append(relation)

        return relations

    def _extract_semantic_causal_relations(self, target_memory: Memory, candidates: List[Memory]) -> List[CausalRelation]:
        """Extract semantic association causal relationships"""
        relations = []

        # Causal inference based on memory types
        causal_type_pairs = [
            ('decision', 'project'),      # Decisions lead to project changes
            ('preference', 'decision'),   # Preferences affect decisions
            ('important', 'decision'),    # Important information affects decisions
            ('learning', 'preference')    # Learning changes preferences
        ]

        for cause_type, effect_type in causal_type_pairs:
            if target_memory.memory_type == cause_type:
                # Find possible effect memories
                effect_candidates = [m for m in candidates if m.memory_type == effect_type]
                for effect_memory in effect_candidates:
                    similarity = self.similarity_calc.calculate_similarity(
                        target_memory.content, effect_memory.content
                    )

                    if similarity > 0.5:  # Semantic similarity threshold
                        strength = self.strength_weights[CausalType.SEMANTIC] * similarity

                        relation = CausalRelation(
                            cause_memory_id=target_memory.id,
                            effect_memory_id=effect_memory.id,
                            causal_type=CausalType.SEMANTIC,
                            strength=strength,
                            confidence=0.5,
                            evidence=[f"Type association: {cause_type}->{effect_type}, Similarity: {similarity:.2f}"],
                            pattern_matched=f"{cause_type}_to_{effect_type}",
                            created_at=datetime.now()
                        )
                        relations.append(relation)

        return relations

    def _find_matching_memory(self, description: str, candidates: List[Memory]) -> Optional[Memory]:
        """Find the memory that best matches the description from candidates"""
        if not description or len(description) < 3:
            return None

        best_match = None
        best_similarity = 0.0

        for memory in candidates:
            similarity = self.similarity_calc.calculate_similarity(description, memory.content)
            if similarity > best_similarity and similarity > 0.4:  # Minimum similarity threshold
                best_similarity = similarity
                best_match = memory

        return best_match

    def _build_causal_graph(self, relations: List[CausalRelation]) -> nx.DiGraph:
        """Build causal relationship graph"""
        graph = nx.DiGraph()

        for relation in relations:
            graph.add_edge(
                relation.cause_memory_id,
                relation.effect_memory_id,
                relation=relation,
                weight=relation.strength
            )

        return graph

    def _identify_causal_chains(self, graph: nx.DiGraph, relations: List[CausalRelation]) -> List[CausalChain]:
        """Identify causal chains"""
        chains = []

        # Find all simple paths (avoid cycles)
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        # Find all simple paths (length limited to 5 to avoid overly long chains)
                        paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))

                        for path in paths:
                            if len(path) < 2:  # At least 2 nodes needed to form a chain
                                continue

                            # Build causal chain
                            chain_relations = []
                            total_strength = 1.0

                            for i in range(len(path) - 1):
                                cause_id = path[i]
                                effect_id = path[i + 1]

                                # Find corresponding causal relationship
                                edge_data = graph.get_edge_data(cause_id, effect_id)
                                if edge_data:
                                    relation = edge_data['relation']
                                    chain_relations.append(relation)
                                    total_strength *= relation.strength

                            if chain_relations:
                                chain = CausalChain(
                                    chain_id=f"chain_{source}_{target}_{len(path)}",
                                    relations=chain_relations,
                                    total_strength=total_strength,
                                    chain_length=len(path),
                                    root_cause=path[0],
                                    final_effect=path[-1]
                                )
                                chains.append(chain)

                    except nx.NetworkXNoPath:
                        continue

        # Sort by strength, return strongest causal chains
        chains.sort(key=lambda c: c.total_strength, reverse=True)
        return chains[:50]  # Limit return count

    def find_root_causes(self, effect_memory_id: int, max_depth: int = 3) -> List[CausalRelation]:
        """
        Find root causes for specified effect

        Args:
            effect_memory_id: Effect memory ID
            max_depth: Maximum search depth

        Returns:
            List of root causes
        """
        # Get all memories for analysis
        all_memories = self.memory_repo.search_memories(limit=1000)
        all_relations = []

        for memory in all_memories:
            relations = self.extract_causal_relations(memory, all_memories)
            all_relations.extend(relations)

        # Build causal graph
        graph = self._build_causal_graph(all_relations)

        # Find all paths pointing to target memory
        root_causes = []
        for node in graph.nodes():
            if node != effect_memory_id:
                try:
                    paths = list(nx.all_simple_paths(graph, node, effect_memory_id, cutoff=max_depth))
                    for path in paths:
                        if len(path) > 1:
                            # Get first causal relationship as root cause
                            edge_data = graph.get_edge_data(path[0], path[1])
                            if edge_data:
                                root_causes.append(edge_data['relation'])
                except nx.NetworkXNoPath:
                    continue

        # Sort by strength
        root_causes.sort(key=lambda r: r.strength, reverse=True)
        return root_causes

    def predict_effects(self, cause_memory_id: int, max_depth: int = 3) -> List[CausalRelation]:
        """
        Predict possible effects from specified cause

        Args:
            cause_memory_id: Cause memory ID
            max_depth: Maximum prediction depth

        Returns:
            List of possible effects
        """
        # Get all memories for analysis
        all_memories = self.memory_repo.search_memories(limit=1000)
        all_relations = []

        for memory in all_memories:
            relations = self.extract_causal_relations(memory, all_memories)
            all_relations.extend(relations)

        graph = self._build_causal_graph(all_relations)

        predicted_effects = []
        for node in graph.nodes():
            if node != cause_memory_id:
                try:
                    paths = list(nx.all_simple_paths(graph, cause_memory_id, node, cutoff=max_depth))
                    for path in paths:
                        if len(path) > 1:
                            # Get last causal relationship as prediction result
                            edge_data = graph.get_edge_data(path[-2], path[-1])
                            if edge_data:
                                predicted_effects.append(edge_data['relation'])
                except nx.NetworkXNoPath:
                    continue

        predicted_effects.sort(key=lambda r: r.strength, reverse=True)
        return predicted_effects

    def get_causal_statistics(self, memories: List[Memory]) -> Dict[str, any]:
        """
        Get causal relationship statistics

        Args:
            memories: List of memories

        Returns:
            Statistics dictionary
        """
        relations = []
        for memory in memories:
            memory_relations = self.extract_causal_relations(memory, memories)
            relations.extend(memory_relations)

        stats = {
            'total_relations': len(relations),
            'by_type': {},
            'strength_distribution': {
                'high': 0,    # >= 0.7
                'medium': 0,  # 0.4-0.7
                'low': 0      # < 0.4
            },
            'avg_strength': 0.0,
            'most_common_patterns': {}
        }

        if not relations:
            return stats

        # Statistics by type
        for relation in relations:
            causal_type = relation.causal_type.value
            if causal_type not in stats['by_type']:
                stats['by_type'][causal_type] = 0
            stats['by_type'][causal_type] += 1

            # Strength distribution
            if relation.strength >= 0.7:
                stats['strength_distribution']['high'] += 1
            elif relation.strength >= 0.4:
                stats['strength_distribution']['medium'] += 1
            else:
                stats['strength_distribution']['low'] += 1

            # Pattern statistics
            pattern = relation.pattern_matched
            if pattern not in stats['most_common_patterns']:
                stats['most_common_patterns'][pattern] = 0
            stats['most_common_patterns'][pattern] += 1

        # Average strength
        stats['avg_strength'] = sum(r.strength for r in relations) / len(relations)

        return stats