#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine Unified Interface
Integrates three core reasoning modules: causal analysis, predictive reasoning, and knowledge graph
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .causal_analyzer import CausalAnalyzer, CausalRelation, CausalChain
from .predictive_engine import PredictiveEngine, Prediction, DecisionPattern
from .knowledge_graph import KnowledgeGraph, Concept, ConceptRelation
from .similarity import SimilarityCalculator
from .scorer import ImportanceScorer
from ..storage.models import Memory, MemoryRepository
from .bilingual_patterns import BILINGUAL_DECISION_KEYWORDS


class ReasoningType(Enum):
    """Reasoning types"""
    CAUSAL = "causal"           # Causal reasoning
    PREDICTIVE = "predictive"   # Predictive reasoning
    CONCEPTUAL = "conceptual"   # Conceptual reasoning
    INTEGRATED = "integrated"   # Integrated reasoning


@dataclass
class ReasoningResult:
    """Reasoning result"""
    reasoning_type: ReasoningType
    content: str
    confidence: float
    evidence: List[str]
    related_memories: List[int]
    metadata: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'reasoning_type': self.reasoning_type.value,
            'content': self.content,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'related_memories': self.related_memories,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class IntegratedInsight:
    """Integrated insight"""
    insight_id: str
    title: str
    description: str
    causal_relations: List[CausalRelation]
    predictions: List[Prediction]
    key_concepts: List[Concept]
    confidence: float
    importance: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'insight_id': self.insight_id,
            'title': self.title,
            'description': self.description,
            'causal_relations': [r.to_dict() for r in self.causal_relations],
            'predictions': [p.to_dict() for p in self.predictions],
            'key_concepts': [c.to_dict() for c in self.key_concepts],
            'confidence': self.confidence,
            'importance': self.importance,
            'created_at': self.created_at.isoformat()
        }


class ReasoningEngine:
    """Reasoning engine unified interface"""

    def __init__(self, similarity_calc: SimilarityCalculator, importance_scorer: ImportanceScorer,
                 memory_repo: MemoryRepository):
        """
        Initialize reasoning engine

        Args:
            similarity_calc: Semantic similarity calculator
            importance_scorer: Importance scorer
            memory_repo: Memory repository
        """
        self.similarity_calc = similarity_calc
        self.importance_scorer = importance_scorer
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.reasoning_engine')

        # Initialize three core reasoning modules
        self.causal_analyzer = CausalAnalyzer(similarity_calc, memory_repo)
        self.predictive_engine = PredictiveEngine(
            self.causal_analyzer, importance_scorer, similarity_calc, memory_repo
        )
        self.knowledge_graph = KnowledgeGraph(similarity_calc, self.causal_analyzer)

        # Reasoning result cache
        self._reasoning_cache = {}
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour

    def comprehensive_reasoning(self, memories: List[Memory],
                              project_context: Optional[Dict[str, Any]] = None) -> List[ReasoningResult]:
        """
        Comprehensive reasoning analysis

        Args:
            memories: List of memories
            project_context: Project context

        Returns:
            List of reasoning results
        """
        self.logger.info(f"Starting comprehensive reasoning analysis based on {len(memories)} memories")

        results = []

        # 1. Causal reasoning analysis
        causal_results = self._perform_causal_reasoning(memories)
        results.extend(causal_results)

        # 2. Predictive reasoning analysis
        predictive_results = self._perform_predictive_reasoning(memories, project_context)
        results.extend(predictive_results)

        # 3. Conceptual reasoning analysis
        conceptual_results = self._perform_conceptual_reasoning(memories)
        results.extend(conceptual_results)

        # 4. Integrated reasoning analysis
        integrated_results = self._perform_integrated_reasoning(memories, project_context)
        results.extend(integrated_results)

        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)

        self.logger.info(f"Comprehensive reasoning completed, generated {len(results)} reasoning results")
        return results

    def _perform_causal_reasoning(self, memories: List[Memory]) -> List[ReasoningResult]:
        """Perform causal reasoning"""
        results = []

        # Analyze causal chains
        causal_chains = self.causal_analyzer.analyze_causal_chain(memories)

        for chain in causal_chains[:5]:  # Take top 5 strongest causal chains
            result = ReasoningResult(
                reasoning_type=ReasoningType.CAUSAL,
                content=f"Found causal chain: from memory {chain.root_cause} to memory {chain.final_effect}, "
                       f"contains {chain.chain_length} links, total strength {chain.total_strength:.2f}",
                confidence=chain.total_strength,
                evidence=[f"Causal chain length: {chain.chain_length}", f"Relation count: {len(chain.relations)}"],
                related_memories=[chain.root_cause, chain.final_effect],
                metadata={
                    'chain_id': chain.chain_id,
                    'chain_length': chain.chain_length,
                    'total_strength': chain.total_strength
                },
                created_at=datetime.now()
            )
            results.append(result)

        return results

    def _perform_predictive_reasoning(self, memories: List[Memory],
                                    project_context: Optional[Dict[str, Any]]) -> List[ReasoningResult]:
        """Perform predictive reasoning"""
        results = []

        # Predict next needs
        if project_context:
            predictions = self.predictive_engine.predict_next_needs(project_context)
        else:
            # Use generic context
            predictions = self.predictive_engine.predict_next_needs({'project_path': None})

        for prediction in predictions[:3]:  # Take top 3 most likely predictions
            result = ReasoningResult(
                reasoning_type=ReasoningType.PREDICTIVE,
                content=f"Prediction: {prediction.content}",
                confidence=prediction.confidence,
                evidence=prediction.evidence,
                related_memories=prediction.related_memories,
                metadata={
                    'prediction_type': prediction.prediction_type.value,
                    'probability': prediction.probability,
                    'time_horizon': prediction.time_horizon.total_seconds() if prediction.time_horizon else None
                },
                created_at=datetime.now()
            )
            results.append(result)

        return results

    def _perform_conceptual_reasoning(self, memories: List[Memory]) -> List[ReasoningResult]:
        """Perform conceptual reasoning"""
        results = []

        # Build knowledge graph
        self.knowledge_graph.build_concept_graph(memories)

        # Get central concepts
        central_concepts = self.knowledge_graph.get_central_concepts(top_k=5)

        for concept_id, centrality in central_concepts:
            concept = self.knowledge_graph.get_concept_by_id(concept_id)
            if concept:
                result = ReasoningResult(
                    reasoning_type=ReasoningType.CONCEPTUAL,
                    content=f"Core concept: {concept.name} (category: {concept.category}), "
                           f"frequency {concept.frequency}, centrality {centrality:.2f}",
                    confidence=centrality,
                    evidence=[f"Frequency: {concept.frequency}", f"Importance: {concept.importance}"],
                    related_memories=concept.related_memories,
                    metadata={
                        'concept_id': concept.concept_id,
                        'category': concept.category,
                        'frequency': concept.frequency,
                        'centrality': centrality
                    },
                    created_at=datetime.now()
                )
                results.append(result)

        return results

    def _perform_integrated_reasoning(self, memories: List[Memory],
                                    project_context: Optional[Dict[str, Any]]) -> List[ReasoningResult]:
        """Perform integrated reasoning"""
        results = []

        # Generate integrated insights
        insights = self.generate_integrated_insights(memories, project_context)

        for insight in insights[:2]:  # Take top 2 most important insights
            result = ReasoningResult(
                reasoning_type=ReasoningType.INTEGRATED,
                content=f"Integrated insight: {insight.title} - {insight.description}",
                confidence=insight.confidence,
                evidence=[
                    f"Causal relationships: {len(insight.causal_relations)}",
                    f"Predictions: {len(insight.predictions)}",
                    f"Key concepts: {len(insight.key_concepts)}"
                ],
                related_memories=[],  # Integrated insights involve multiple memories
                metadata={
                    'insight_id': insight.insight_id,
                    'importance': insight.importance,
                    'components': {
                        'causal_relations': len(insight.causal_relations),
                        'predictions': len(insight.predictions),
                        'key_concepts': len(insight.key_concepts)
                    }
                },
                created_at=datetime.now()
            )
            results.append(result)

        return results

    def generate_integrated_insights(self, memories: List[Memory],
                                   project_context: Optional[Dict[str, Any]] = None) -> List[IntegratedInsight]:
        """
        Generate integrated insights

        Args:
            memories: List of memories
            project_context: Project context

        Returns:
            List of integrated insights
        """
        insights = []

        # Get analysis results from each module
        causal_chains = self.causal_analyzer.analyze_causal_chain(memories)

        if project_context:
            predictions = self.predictive_engine.predict_next_needs(project_context)
        else:
            predictions = self.predictive_engine.predict_next_needs({'project_path': None})

        self.knowledge_graph.build_concept_graph(memories)
        central_concepts = self.knowledge_graph.get_central_concepts(top_k=10)

        # Generate tech stack-based integrated insights
        tech_insight = self._generate_tech_stack_insight(
            causal_chains, predictions, central_concepts, memories
        )
        if tech_insight:
            insights.append(tech_insight)

        # Generate decision pattern-based integrated insights
        decision_insight = self._generate_decision_pattern_insight(
            causal_chains, predictions, central_concepts, memories
        )
        if decision_insight:
            insights.append(decision_insight)

        # Sort by importance
        insights.sort(key=lambda i: i.importance, reverse=True)

        self.logger.info(f"Generated {len(insights)} integrated insights")
        return insights

    def _generate_tech_stack_insight(self, causal_chains: List[CausalChain],
                                   predictions: List[Prediction],
                                   central_concepts: List[Tuple[str, float]],
                                   memories: List[Memory]) -> Optional[IntegratedInsight]:
        """Generate tech stack-related integrated insights"""

        # Identify technology-related concepts
        tech_concepts = []
        for concept_id, centrality in central_concepts:
            concept = self.knowledge_graph.get_concept_by_id(concept_id)
            if concept and concept.category == 'technology':
                tech_concepts.append(concept)

        if not tech_concepts:
            return None

        # Find technology-related causal relationships
        tech_causal_relations = []
        for chain in causal_chains:
            for relation in chain.relations:
                if relation.causal_type.value in ['explicit', 'decision']:
                    tech_causal_relations.append(relation)

        # Find technology-related predictions
        tech_predictions = [p for p in predictions if p.prediction_type.value == 'tech_choice']

        if tech_concepts:
            insight = IntegratedInsight(
                insight_id=f"tech_insight_{datetime.now().timestamp()}",
                title="Technology Stack Evolution Insight",
                description=f"Analysis based on {len(tech_concepts)} core technology concepts, "
                          f"found {len(tech_causal_relations)} technology decision causal relationships, "
                          f"predicting {len(tech_predictions)} technology selection recommendations",
                causal_relations=tech_causal_relations[:3],
                predictions=tech_predictions[:3],
                key_concepts=tech_concepts[:5],
                confidence=0.8,
                importance=0.9,
                created_at=datetime.now()
            )
            return insight

        return None

    def _generate_decision_pattern_insight(self, causal_chains: List[CausalChain],
                                         predictions: List[Prediction],
                                         central_concepts: List[Tuple[str, float]],
                                         memories: List[Memory]) -> Optional[IntegratedInsight]:
        """Generate decision pattern-related integrated insights"""

        # Analyze decision patterns
        decision_patterns = self.predictive_engine.analyze_decision_patterns(memories)

        if not decision_patterns:
            return None

        # Find decision-related causal relationships
        decision_causal_relations = []
        for chain in causal_chains:
            for relation in chain.relations:
                if relation.causal_type.value == 'decision':
                    decision_causal_relations.append(relation)

        # Find decision-related predictions
        decision_predictions = [p for p in predictions if 'decision' in p.content.lower()]

        # Find decision-related concepts
        decision_concepts = []
        for concept_id, centrality in central_concepts:
            concept = self.knowledge_graph.get_concept_by_id(concept_id)
            if concept and any(keyword in concept.name.lower()
                             for keyword in ['决策', 'decision', '选择', 'choice']):  # Chinese and English keywords for decision
                decision_concepts.append(concept)

        if decision_patterns:
            insight = IntegratedInsight(
                insight_id=f"decision_insight_{datetime.now().timestamp()}",
                title="Decision Pattern Insight",
                description=f"Identified {len(decision_patterns)} decision patterns, "
                          f"found {len(decision_causal_relations)} decision causal relationships, "
                          f"generated {len(decision_predictions)} decision recommendations",
                causal_relations=decision_causal_relations[:3],
                predictions=decision_predictions[:3],
                key_concepts=decision_concepts[:3],
                confidence=0.7,
                importance=0.8,
                created_at=datetime.now()
            )
            return insight

        return None

    def query_reasoning(self, query: str, memories: List[Memory],
                       reasoning_types: Optional[List[ReasoningType]] = None) -> List[ReasoningResult]:
        """
        Query-based reasoning

        Args:
            query: Query content
            memories: List of memories
            reasoning_types: Reasoning type filter

        Returns:
            List of reasoning results
        """
        results = []

        # If no reasoning type specified, use all types
        if reasoning_types is None:
            reasoning_types = list(ReasoningType)

        # Filter relevant memories based on query content
        relevant_memories = self._filter_relevant_memories(query, memories)

        for reasoning_type in reasoning_types:
            if reasoning_type == ReasoningType.CAUSAL:
                causal_results = self._query_causal_reasoning(query, relevant_memories)
                results.extend(causal_results)
            elif reasoning_type == ReasoningType.PREDICTIVE:
                predictive_results = self._query_predictive_reasoning(query, relevant_memories)
                results.extend(predictive_results)
            elif reasoning_type == ReasoningType.CONCEPTUAL:
                conceptual_results = self._query_conceptual_reasoning(query, relevant_memories)
                results.extend(conceptual_results)

        # Sort by relevance and confidence
        results.sort(key=lambda r: r.confidence, reverse=True)

        self.logger.info(f"Query reasoning '{query}' returned {len(results)} results")
        return results

    def _filter_relevant_memories(self, query: str, memories: List[Memory]) -> List[Memory]:
        """Filter relevant memories"""
        relevant_memories = []

        for memory in memories:
            similarity = self.similarity_calc.calculate_similarity(query, memory.content)
            if similarity > 0.3:  # Similarity threshold
                relevant_memories.append(memory)

        # Sort by similarity, take top 20 most relevant
        relevant_memories.sort(
            key=lambda m: self.similarity_calc.calculate_similarity(query, m.content),
            reverse=True
        )
        return relevant_memories[:20]

    def _query_causal_reasoning(self, query: str, memories: List[Memory]) -> List[ReasoningResult]:
        """Query causal reasoning"""
        results = []

        # Analyze causal relationships
        causal_chains = self.causal_analyzer.analyze_causal_chain(memories)

        for chain in causal_chains[:3]:
            # Check if causal chain is relevant to query
            chain_relevance = 0.0
            for relation in chain.relations:
                # Simplified: find relevance based on memory ID
                for memory in memories:
                    if memory.id in [relation.cause_memory_id, relation.effect_memory_id]:
                        similarity = self.similarity_calc.calculate_similarity(query, memory.content)
                        chain_relevance = max(chain_relevance, similarity)

            if chain_relevance > 0.4:
                result = ReasoningResult(
                    reasoning_type=ReasoningType.CAUSAL,
                    content=f"Relevant causal chain: {chain.chain_id}, strength {chain.total_strength:.2f}",
                    confidence=chain.total_strength * chain_relevance,
                    evidence=[f"Query relevance: {chain_relevance:.2f}"],
                    related_memories=[chain.root_cause, chain.final_effect],
                    metadata={'chain_id': chain.chain_id},
                    created_at=datetime.now()
                )
                results.append(result)

        return results

    def _query_predictive_reasoning(self, query: str, memories: List[Memory]) -> List[ReasoningResult]:
        """Query predictive reasoning"""
        results = []

        # Generate predictions based on query
        predictions = self.predictive_engine.recommend_proactive_info(query)

        for prediction in predictions[:3]:
            result = ReasoningResult(
                reasoning_type=ReasoningType.PREDICTIVE,
                content=f"Predictive recommendation: {prediction.content}",
                confidence=prediction.confidence,
                evidence=prediction.evidence,
                related_memories=prediction.related_memories,
                metadata={'prediction_type': prediction.prediction_type.value},
                created_at=datetime.now()
            )
            results.append(result)

        return results

    def _query_conceptual_reasoning(self, query: str, memories: List[Memory]) -> List[ReasoningResult]:
        """Query conceptual reasoning"""
        results = []

        # Build knowledge graph
        self.knowledge_graph.build_concept_graph(memories)

        # Search relevant concepts
        concept_results = self.knowledge_graph.search_concepts(query, top_k=5)

        for concept_id, similarity in concept_results:
            concept = self.knowledge_graph.get_concept_by_id(concept_id)
            if concept:
                result = ReasoningResult(
                    reasoning_type=ReasoningType.CONCEPTUAL,
                    content=f"Relevant concept: {concept.name} ({concept.category}), similarity {similarity:.2f}",
                    confidence=similarity,
                    evidence=[f"Concept frequency: {concept.frequency}"],
                    related_memories=concept.related_memories,
                    metadata={'concept_id': concept.concept_id},
                    created_at=datetime.now()
                )
                results.append(result)

        return results

    def get_reasoning_statistics(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Get reasoning statistics

        Args:
            memories: List of memories

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_memories': len(memories),
            'causal_analysis': {},
            'predictive_analysis': {},
            'conceptual_analysis': {},
            'integrated_analysis': {}
        }

        # Causal analysis statistics
        causal_stats = self.causal_analyzer.get_causal_statistics(memories)
        stats['causal_analysis'] = causal_stats

        # Predictive analysis statistics
        predictions = self.predictive_engine.predict_next_needs({'project_path': None})
        predictive_stats = self.predictive_engine.get_prediction_statistics(predictions)
        stats['predictive_analysis'] = predictive_stats

        # Conceptual analysis statistics
        self.knowledge_graph.build_concept_graph(memories)
        conceptual_stats = self.knowledge_graph.get_graph_statistics()
        stats['conceptual_analysis'] = conceptual_stats

        # Integrated analysis statistics
        insights = self.generate_integrated_insights(memories)
        stats['integrated_analysis'] = {
            'total_insights': len(insights),
            'avg_confidence': sum(i.confidence for i in insights) / len(insights) if insights else 0.0,
            'avg_importance': sum(i.importance for i in insights) / len(insights) if insights else 0.0
        }

        return stats

    def clear_cache(self):
        """Clear reasoning cache"""
        self._reasoning_cache.clear()
        self.logger.info("Reasoning cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._reasoning_cache),
            'cache_ttl_hours': self._cache_ttl.total_seconds() / 3600
        }