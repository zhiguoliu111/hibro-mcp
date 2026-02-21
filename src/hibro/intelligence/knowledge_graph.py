#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Builder
Extracts concepts from memories and builds knowledge graphs, supports concept relationship analysis and graph traversal
"""

import re
import json
import logging
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter

from .similarity import SimilarityCalculator
from .causal_analyzer import CausalAnalyzer
from ..storage.models import Memory


class RelationType(Enum):
    """Relationship types"""
    SIMILAR = "similar"           # Semantic similarity
    CAUSAL = "causal"            # Causal relationship
    HIERARCHICAL = "hierarchical" # Hierarchical relationship
    TEMPORAL = "temporal"         # Temporal relationship
    CATEGORICAL = "categorical"   # Categorical relationship


@dataclass
class Concept:
    """Concept node"""
    concept_id: str
    name: str
    category: str
    frequency: int               # Occurrence frequency
    importance: float            # Importance score
    related_memories: List[int]  # Related memory IDs
    aliases: List[str]           # Alias list
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'concept_id': self.concept_id,
            'name': self.name,
            'category': self.category,
            'frequency': self.frequency,
            'importance': self.importance,
            'related_memories': self.related_memories,
            'aliases': self.aliases,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ConceptRelation:
    """Concept relationship"""
    relation_id: str
    concept1_id: str
    concept2_id: str
    relation_type: RelationType
    weight: float                # Relationship weight
    evidence_count: int          # Evidence count
    confidence: float            # Confidence level
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'relation_id': self.relation_id,
            'concept1_id': self.concept1_id,
            'concept2_id': self.concept2_id,
            'relation_type': self.relation_type.value,
            'weight': self.weight,
            'evidence_count': self.evidence_count,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }


class KnowledgeGraph:
    """Knowledge graph builder"""

    def __init__(self, similarity_calc: SimilarityCalculator, causal_analyzer: CausalAnalyzer):
        """
        Initialize knowledge graph builder

        Args:
            similarity_calc: Semantic similarity calculator
            causal_analyzer: Causal relationship analyzer
        """
        self.similarity_calc = similarity_calc
        self.causal_analyzer = causal_analyzer
        self.logger = logging.getLogger('hibro.knowledge_graph')

        # Build graph using NetworkX
        self.graph = nx.Graph()  # Undirected graph for general relationships
        self.directed_graph = nx.DiGraph()  # Directed graph for causal relationships

        # Concept and relationship storage
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, ConceptRelation] = {}

        # Concept extraction rules
        self.concept_patterns = {
            'technology': [
                r'\b(python|javascript|react|vue|django|fastapi|postgresql|mongodb|docker|kubernetes)\b',
                r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b',  # CamelCase
            ],
            'methodology': [
                r'\b(agile|waterfall|test-driven|domain-driven|microservices|monolithic)\b',
                r'\b(TDD|DDD|CI/CD|DevOps)\b'
            ],
            'domain': [
                r'\b(user management|permission control|data analysis|machine learning|artificial intelligence)\b',
                r'\b(e-commerce|finance|education|healthcare|gaming)\b'
            ]
        }

    def build_concept_graph(self, memories: List[Memory]) -> nx.Graph:
        """
        Build concept graph from memory list

        Args:
            memories: List of memories

        Returns:
            Concept graph
        """
        self.logger.info(f"Starting to build knowledge graph based on {len(memories)} memories")

        # 1. Extract concepts
        extracted_concepts = self._extract_concepts(memories)

        # 2. Calculate concept relationships
        concept_relations = self._calculate_concept_relations(extracted_concepts, memories)

        # 3. Build graph
        self._build_graph(extracted_concepts, concept_relations)

        # 4. Optimize graph structure
        self._optimize_graph_structure()

        self.logger.info(f"Built knowledge graph with {len(self.concepts)} concepts and {len(self.relations)} relationships")
        return self.graph

    def _extract_concepts(self, memories: List[Memory]) -> List[Concept]:
        """Extract concepts from memories"""
        concept_candidates = defaultdict(list)

        for memory in memories:
            content = memory.content
            memory_id = memory.id if memory.id else 0

            # Use regex patterns to extract concepts
            for category, patterns in self.concept_patterns.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            concept_name = match.group(0).lower().strip()
                            if len(concept_name) > 2:  # Filter too short concepts
                                concept_candidates[concept_name].append((memory_id, category))
                    except re.error as e:
                        self.logger.warning(f"Regex error: {pattern} - {e}")
                        continue

            # Extract Chinese concepts (based on word segmentation)
            chinese_concepts = self._extract_chinese_concepts(content)
            for concept_name in chinese_concepts:
                concept_candidates[concept_name].append((memory_id, 'general'))

        # Convert to Concept objects
        concepts = []
        for concept_name, occurrences in concept_candidates.items():
            if len(occurrences) >= 2:  # Valid concept only if appears at least 2 times
                memory_ids = [occ[0] for occ in occurrences]
                categories = [occ[1] for occ in occurrences]
                most_common_category = Counter(categories).most_common(1)[0][0]

                concept = Concept(
                    concept_id=f"concept_{len(concepts)}",
                    name=concept_name,
                    category=most_common_category,
                    frequency=len(occurrences),
                    importance=min(len(occurrences) / 10.0, 1.0),  # Calculate importance based on frequency
                    related_memories=memory_ids,
                    aliases=[],
                    created_at=datetime.now()
                )
                concepts.append(concept)
                self.concepts[concept.concept_id] = concept

        self.logger.info(f"Extracted {len(concepts)} valid concepts")
        return concepts

    def _extract_chinese_concepts(self, content: str) -> List[str]:
        """Extract Chinese concepts (simplified version)"""
        concepts = []

        # Simple Chinese concept extraction (based on common patterns)
        # These patterns match Chinese characters combined with technical suffixes:
        # - 系统, 平台, 框架, 工具
        # - 方法, 模式, 架构
        # - 管理, 控制, 分析, 处理, 优化
        # - Prefixes: 基于, 使用, 采用, 选择
        chinese_patterns = [
            r'[\u4e00-\u9fff]{2,4}(?:系统|平台|框架|工具|方法|模式|架构)',
            r'[\u4e00-\u9fff]{2,6}(?:管理|控制|分析|处理|优化)',
            r'(?:基于|使用|采用|选择)[\u4e00-\u9fff]{2,6}'
        ]

        for pattern in chinese_patterns:
            try:
                matches = re.finditer(pattern, content)
                for match in matches:
                    concept = match.group(0)
                    # Clean concept name (remove prefix particles)
                    concept = re.sub(r'^(基于|使用|采用|选择)', '', concept)
                    if len(concept) >= 2:
                        concepts.append(concept)
            except re.error as e:
                self.logger.warning(f"Chinese concept extraction error: {pattern} - {e}")
                continue

        return concepts

    def _calculate_concept_relations(self, concepts: List[Concept], memories: List[Memory]) -> List[ConceptRelation]:
        """Calculate relationships between concepts"""
        relations = []

        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                # Calculate multiple types of relationships

                # 1. Semantic similarity relationship
                similarity = self.similarity_calc.calculate_similarity(concept1.name, concept2.name)
                if similarity > 0.6:
                    relation = ConceptRelation(
                        relation_id=f"rel_{len(relations)}",
                        concept1_id=concept1.concept_id,
                        concept2_id=concept2.concept_id,
                        relation_type=RelationType.SIMILAR,
                        weight=similarity,
                        evidence_count=1,
                        confidence=similarity,
                        created_at=datetime.now()
                    )
                    relations.append(relation)

                # 2. Co-occurrence relationship
                cooccurrence_weight = self._calculate_cooccurrence(concept1, concept2, memories)
                if cooccurrence_weight > 0.3:
                    relation = ConceptRelation(
                        relation_id=f"rel_{len(relations)}",
                        concept1_id=concept1.concept_id,
                        concept2_id=concept2.concept_id,
                        relation_type=RelationType.CATEGORICAL,
                        weight=cooccurrence_weight,
                        evidence_count=len(set(concept1.related_memories) & set(concept2.related_memories)),
                        confidence=cooccurrence_weight,
                        created_at=datetime.now()
                    )
                    relations.append(relation)

                # 3. Hierarchical relationship
                hierarchical_weight = self._detect_hierarchical_relation(concept1, concept2)
                if hierarchical_weight > 0.5:
                    relation = ConceptRelation(
                        relation_id=f"rel_{len(relations)}",
                        concept1_id=concept1.concept_id,
                        concept2_id=concept2.concept_id,
                        relation_type=RelationType.HIERARCHICAL,
                        weight=hierarchical_weight,
                        evidence_count=1,
                        confidence=hierarchical_weight,
                        created_at=datetime.now()
                    )
                    relations.append(relation)

        self.logger.info(f"Calculated {len(relations)} concept relationships")
        return relations

    def _calculate_cooccurrence(self, concept1: Concept, concept2: Concept, memories: List[Memory]) -> float:
        """Calculate co-occurrence weight between two concepts"""
        memory_ids1 = set(concept1.related_memories)
        memory_ids2 = set(concept2.related_memories)

        # Count co-occurring memories
        cooccurrence_count = len(memory_ids1 & memory_ids2)
        total_occurrences = len(memory_ids1 | memory_ids2)

        if total_occurrences == 0:
            return 0.0

        # Jaccard similarity
        jaccard_similarity = cooccurrence_count / total_occurrences

        # Consider frequency influence
        frequency_factor = min(concept1.frequency, concept2.frequency) / max(concept1.frequency, concept2.frequency)

        return jaccard_similarity * frequency_factor

    def _detect_hierarchical_relation(self, concept1: Concept, concept2: Concept) -> float:
        """Detect hierarchical relationship"""
        name1, name2 = concept1.name.lower(), concept2.name.lower()

        # Check inclusion relationship
        if name1 in name2 or name2 in name1:
            return 0.8

        # Check common hierarchical relationship patterns (Chinese-specific)
        # Patterns: X system/framework/platform/tool -> X
        # Remove common suffixes (Chinese technical suffixes)
        normalization_rules = [
            (r'(.+)系统', r'\1'),  # X系统 -> X
            (r'(.+)框架', r'\1'),  # X框架 -> X
            (r'(.+)平台', r'\1'),  # X平台 -> X
            (r'(.+)工具', r'\1')   # X工具 -> X
        ]

        for parent_pattern, child_pattern in normalization_rules:
            try:
                if re.match(parent_pattern, name1) and re.match(child_pattern, name2):
                    return 0.7
                if re.match(parent_pattern, name2) and re.match(child_pattern, name1):
                    return 0.7
            except re.error:
                continue

        return 0.0

    def _build_graph(self, concepts: List[Concept], relations: List[ConceptRelation]):
        """Build graph structure"""
        # Add concept nodes
        for concept in concepts:
            self.graph.add_node(
                concept.concept_id,
                name=concept.name,
                category=concept.category,
                frequency=concept.frequency,
                importance=concept.importance
            )

        # Add relationship edges
        for relation in relations:
            self.relations[relation.relation_id] = relation

            # Select graph based on relationship type
            if relation.relation_type == RelationType.CAUSAL:
                # Use directed graph for causal relationships
                self.directed_graph.add_edge(
                    relation.concept1_id,
                    relation.concept2_id,
                    relation_type=relation.relation_type.value,
                    weight=relation.weight,
                    confidence=relation.confidence
                )
            else:
                # Use undirected graph for other relationships
                self.graph.add_edge(
                    relation.concept1_id,
                    relation.concept2_id,
                    relation_type=relation.relation_type.value,
                    weight=relation.weight,
                    confidence=relation.confidence
                )

    def _optimize_graph_structure(self):
        """Optimize graph structure"""
        # 1. Remove edges with too low weight
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 0) < 0.3:
                edges_to_remove.append((u, v))

        self.graph.remove_edges_from(edges_to_remove)

        # 2. Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)

        # 3. Merge similar concepts
        self._merge_similar_concepts()

        self.logger.info(f"Graph optimization completed, removed {len(edges_to_remove)} low-weight edges and {len(isolated_nodes)} isolated nodes")

    def _merge_similar_concepts(self):
        """Merge similar concepts"""
        concepts_to_merge = []

        for concept1_id in self.graph.nodes():
            for concept2_id in self.graph.nodes():
                if concept1_id >= concept2_id:  # Avoid duplicate comparison
                    continue

                if concept1_id in self.concepts and concept2_id in self.concepts:
                    concept1 = self.concepts[concept1_id]
                    concept2 = self.concepts[concept2_id]

                    # Check if should merge
                    if self._should_merge_concepts(concept1, concept2):
                        concepts_to_merge.append((concept1_id, concept2_id))

        # Execute merge
        for concept1_id, concept2_id in concepts_to_merge:
            self._merge_concept_pair(concept1_id, concept2_id)

        if concepts_to_merge:
            self.logger.info(f"Merged {len(concepts_to_merge)} pairs of similar concepts")

    def _should_merge_concepts(self, concept1: Concept, concept2: Concept) -> bool:
        """Determine if two concepts should be merged"""
        # Names highly similar
        similarity = self.similarity_calc.calculate_similarity(concept1.name, concept2.name)
        if similarity > 0.9:
            return True

        # One is alias of the other
        if concept1.name in concept2.aliases or concept2.name in concept1.aliases:
            return True

        # Same category and name inclusion relationship
        if (concept1.category == concept2.category and
            (concept1.name in concept2.name or concept2.name in concept1.name)):
            return True

        return False

    def _merge_concept_pair(self, concept1_id: str, concept2_id: str):
        """Merge two concepts"""
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            return

        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]

        # Select higher frequency concept as main concept
        if concept1.frequency >= concept2.frequency:
            main_concept, merge_concept = concept1, concept2
            main_id, merge_id = concept1_id, concept2_id
        else:
            main_concept, merge_concept = concept2, concept1
            main_id, merge_id = concept2_id, concept1_id

        # Merge attributes
        main_concept.frequency += merge_concept.frequency
        main_concept.importance = max(main_concept.importance, merge_concept.importance)
        main_concept.related_memories.extend(merge_concept.related_memories)
        main_concept.related_memories = list(set(main_concept.related_memories))  # Remove duplicates
        main_concept.aliases.append(merge_concept.name)

        # Update graph structure
        # Redirect edges pointing to merge_concept to main_concept
        edges_to_update = []
        for u, v, data in self.graph.edges(data=True):
            if u == merge_id:
                edges_to_update.append((main_id, v, data))
            elif v == merge_id:
                edges_to_update.append((u, main_id, data))

        # Remove old edges and node
        if self.graph.has_node(merge_id):
            self.graph.remove_node(merge_id)

        # Add new edges
        for u, v, data in edges_to_update:
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, **data)

        # Delete merged concept
        del self.concepts[merge_id]

    def find_concept_clusters(self) -> List[List[str]]:
        """Discover concept clusters"""
        # Use community detection algorithm
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(self.graph)
            return [list(community) for community in communities]
        except ImportError:
            # If no community detection algorithm, use simple connected components
            return [list(component) for component in nx.connected_components(self.graph)]

    def traverse_relation_path(self, start_concept: str, end_concept: str, max_depth: int = 3) -> List[List[str]]:
        """Traverse relationship paths between concepts"""
        if start_concept not in self.graph or end_concept not in self.graph:
            return []

        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.graph, start_concept, end_concept, cutoff=max_depth))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_concept_neighbors(self, concept_id: str, relation_types: Optional[List[RelationType]] = None) -> List[Tuple[str, float]]:
        """Get neighbor nodes of a concept"""
        if concept_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in self.graph.neighbors(concept_id):
            edge_data = self.graph.get_edge_data(concept_id, neighbor_id)

            # Filter by relation type
            if relation_types:
                edge_relation_type = edge_data.get('relation_type')
                if edge_relation_type not in [rt.value for rt in relation_types]:
                    continue

            weight = edge_data.get('weight', 0.0)
            neighbors.append((neighbor_id, weight))

        # Sort by weight
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def get_central_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most central concepts"""
        if len(self.graph.nodes()) == 0:
            return []

        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)

        # Sort by centrality
        sorted_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:top_k]

    def search_concepts(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search related concepts"""
        results = []

        for concept_id, concept in self.concepts.items():
            # Calculate similarity between query and concept name
            similarity = self.similarity_calc.calculate_similarity(query, concept.name)

            # Also check aliases
            for alias in concept.aliases:
                alias_similarity = self.similarity_calc.calculate_similarity(query, alias)
                similarity = max(similarity, alias_similarity)

            if similarity > 0.3:  # Similarity threshold
                results.append((concept_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'graph_density': nx.density(self.graph) if len(self.graph.nodes()) > 1 else 0.0,
            'connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph) if len(self.graph.nodes()) > 0 else 0.0,
            'concept_categories': {},
            'relation_types': {},
            'top_concepts': self.get_central_concepts(5)
        }

        # Statistics by category
        for concept in self.concepts.values():
            category = concept.category
            if category not in stats['concept_categories']:
                stats['concept_categories'][category] = 0
            stats['concept_categories'][category] += 1

        # Statistics by type
        for relation in self.relations.values():
            rel_type = relation.relation_type.value
            if rel_type not in stats['relation_types']:
                stats['relation_types'][rel_type] = 0
            stats['relation_types'][rel_type] += 1

        return stats

    def export_graph(self, format: str = 'json') -> str:
        """Export graph"""
        if format == 'gexf':
            # Export as GEXF format (for visualization in tools like Gephi)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
                nx.write_gexf(self.graph, f.name)
                return f.name
        elif format == 'json':
            # Export as JSON format
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(self.graph)
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_concept_by_id(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID"""
        return self.concepts.get(concept_id)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        for concept in self.concepts.values():
            if concept.name.lower() == name.lower():
                return concept
            if name.lower() in [alias.lower() for alias in concept.aliases]:
                return concept
        return None

    def get_relation_by_id(self, relation_id: str) -> Optional[ConceptRelation]:
        """Get relation by ID"""
        return self.relations.get(relation_id)

    def clear_graph(self):
        """Clear graph"""
        self.graph.clear()
        self.directed_graph.clear()
        self.concepts.clear()
        self.relations.clear()
        self.logger.info("Knowledge graph cleared")