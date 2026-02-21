#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Personalized Recommendation Algorithm
Provides intelligent recommendation services combining knowledge graph and user behavior patterns
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import math

from .behavior_analyzer import BehaviorAnalyzer, UserBehavior
from .knowledge_graph import KnowledgeGraph
from .causal_analyzer import CausalAnalyzer
from ..storage.models import Memory, MemoryRepository
from .bilingual_patterns import BILINGUAL_STOP_WORDS


class RecommendationType(Enum):
    """Recommendation types"""
    COLLABORATIVE = "collaborative"     # Collaborative filtering
    CONTENT_BASED = "content_based"    # Content-based filtering
    HYBRID = "hybrid"                  # Hybrid recommendation
    CAUSAL = "causal"                  # Causal recommendation
    PREDICTIVE = "predictive"          # Predictive recommendation


class RecommendationReason(Enum):
    """Recommendation reasons"""
    SIMILAR_USERS = "similar_users"           # Liked by similar users
    SIMILAR_CONTENT = "similar_content"       # Similar content
    CAUSAL_RELATION = "causal_relation"       # Causal relationship
    FREQUENT_PATTERN = "frequent_pattern"     # Frequent pattern
    CONTEXTUAL_MATCH = "contextual_match"     # Contextual match
    KNOWLEDGE_GRAPH = "knowledge_graph"       # Knowledge graph association


@dataclass
class MemoryRecommendation:
    """Memory recommendation"""
    memory_id: int
    memory_content: str
    recommendation_type: RecommendationType
    confidence_score: float
    relevance_score: float
    reason: RecommendationReason
    explanation: str
    related_memories: List[int]
    context_factors: Dict[str, Any]
    created_at: datetime


@dataclass
class KnowledgeGap:
    """Knowledge gap"""
    gap_type: str  # 'missing_concept', 'weak_connection', 'outdated_info'
    description: str
    suggested_topics: List[str]
    priority: float
    evidence: List[str]
    related_memories: List[int]


@dataclass
class InfoNeed:
    """Information need"""
    need_type: str  # 'immediate', 'contextual', 'proactive'
    description: str
    suggested_queries: List[str]
    confidence: float
    urgency: float
    context: Dict[str, Any]


class PersonalizedRecommender:
    """Personalized recommender"""

    def __init__(self, memory_repo: MemoryRepository, behavior_analyzer: BehaviorAnalyzer,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 causal_analyzer: Optional[CausalAnalyzer] = None):
        """
        Initialize personalized recommender

        Args:
            memory_repo: Memory repository
            behavior_analyzer: Behavior analyzer
            knowledge_graph: Knowledge graph (optional)
            causal_analyzer: Causal analyzer (optional)
        """
        self.memory_repo = memory_repo
        self.behavior_analyzer = behavior_analyzer
        self.knowledge_graph = knowledge_graph
        self.causal_analyzer = causal_analyzer
        self.logger = logging.getLogger('hibro.recommender')

        # Recommendation parameters
        self.similarity_threshold = 0.3
        self.max_recommendations = 10
        self.diversity_factor = 0.2
        self.novelty_factor = 0.1

        # User profile cache
        self._user_profiles = {}
        self._profile_cache_ttl = timedelta(hours=2)

        # Recommendation history
        self._recommendation_history = defaultdict(list)

    def recommend_related_memories(self, context: str, project_path: Optional[str] = None,
                                 user_session: Optional[str] = None,
                                 recommendation_types: Optional[List[RecommendationType]] = None) -> List[MemoryRecommendation]:
        """
        Recommend related memories

        Args:
            context: Current context
            project_path: Project path
            user_session: User session
            recommendation_types: List of recommendation types

        Returns:
            List of recommendations
        """
        try:
            if not recommendation_types:
                recommendation_types = [RecommendationType.HYBRID]

            all_recommendations = []

            # Get candidate memories
            candidate_memories = self._get_candidate_memories(context, project_path)

            if not candidate_memories:
                return []

            # Generate recommendations based on recommendation types
            for rec_type in recommendation_types:
                if rec_type == RecommendationType.COLLABORATIVE:
                    recommendations = self._collaborative_filtering(context, candidate_memories, user_session)
                elif rec_type == RecommendationType.CONTENT_BASED:
                    recommendations = self._content_based_filtering(context, candidate_memories)
                elif rec_type == RecommendationType.CAUSAL:
                    recommendations = self._causal_based_recommendation(context, candidate_memories)
                elif rec_type == RecommendationType.PREDICTIVE:
                    recommendations = self._predictive_recommendation(context, candidate_memories, project_path)
                else:  # HYBRID
                    recommendations = self._hybrid_recommendation(context, candidate_memories, user_session, project_path)

                all_recommendations.extend(recommendations)

            # Deduplicate and sort
            unique_recommendations = self._deduplicate_recommendations(all_recommendations)
            ranked_recommendations = self._rank_recommendations(unique_recommendations, context)

            # Apply diversity and novelty
            final_recommendations = self._apply_diversity_and_novelty(ranked_recommendations)

            # Record recommendation history
            if user_session:
                self._record_recommendations(user_session, final_recommendations)

            return final_recommendations[:self.max_recommendations]

        except Exception as e:
            self.logger.error(f"Failed to recommend related memories: {e}")
            return []

    def suggest_missing_knowledge(self, project_path: str, current_memories: Optional[List[Memory]] = None) -> List[KnowledgeGap]:
        """
        Suggest missing knowledge

        Args:
            project_path: Project path
            current_memories: Current memory list

        Returns:
            List of knowledge gaps
        """
        try:
            if not current_memories:
                current_memories = self.memory_repo.get_project_memories(project_path, limit=200)

            knowledge_gaps = []

            # Analyze technology stack completeness
            tech_gaps = self._analyze_tech_stack_gaps(current_memories, project_path)
            knowledge_gaps.extend(tech_gaps)

            # Analyze concept connectivity
            if self.knowledge_graph:
                concept_gaps = self._analyze_concept_gaps(current_memories)
                knowledge_gaps.extend(concept_gaps)

            # Analyze timeliness
            temporal_gaps = self._analyze_temporal_gaps(current_memories)
            knowledge_gaps.extend(temporal_gaps)

            # Sort by priority
            knowledge_gaps.sort(key=lambda g: g.priority, reverse=True)

            return knowledge_gaps[:10]

        except Exception as e:
            self.logger.error(f"Failed to suggest missing knowledge: {e}")
            return []

    def predict_information_needs(self, current_task: str, project_path: Optional[str] = None,
                                user_session: Optional[str] = None) -> List[InfoNeed]:
        """
        Predict information needs

        Args:
            current_task: Current task
            project_path: Project path
            user_session: User session

        Returns:
            List of information needs
        """
        try:
            info_needs = []

            # Predict based on task type
            task_based_needs = self._predict_task_based_needs(current_task, project_path)
            info_needs.extend(task_based_needs)

            # Predict based on user behavior patterns
            if user_session:
                behavior_based_needs = self._predict_behavior_based_needs(user_session, current_task)
                info_needs.extend(behavior_based_needs)

            # Predict based on project phase
            if project_path:
                phase_based_needs = self._predict_phase_based_needs(project_path, current_task)
                info_needs.extend(phase_based_needs)

            # Sort by urgency and confidence
            info_needs.sort(key=lambda n: (n.urgency * n.confidence), reverse=True)

            return info_needs[:8]

        except Exception as e:
            self.logger.error(f"Failed to predict information needs: {e}")
            return []

    def _get_candidate_memories(self, context: str, project_path: Optional[str]) -> List[Memory]:
        """Get candidate memories"""
        try:
            # Get memories based on project path
            if project_path:
                project_memories = self.memory_repo.get_project_memories(project_path, limit=100)
            else:
                project_memories = []

            # Search based on keywords
            keywords = self._extract_context_keywords(context)
            keyword_memories = []
            for keyword in keywords[:5]:  # Limit keyword count
                memories = self.memory_repo.search_memories(query=keyword, limit=20)
                keyword_memories.extend(memories)

            # Merge and deduplicate
            all_memories = project_memories + keyword_memories
            unique_memories = {m.id: m for m in all_memories}.values()

            return list(unique_memories)

        except Exception as e:
            self.logger.error(f"Failed to get candidate memories: {e}")
            return []

    def _collaborative_filtering(self, context: str, candidate_memories: List[Memory],
                               user_session: Optional[str]) -> List[MemoryRecommendation]:
        """Collaborative filtering recommendation"""
        recommendations = []

        try:
            if not user_session:
                return recommendations

            # Get user behavior history
            user_behaviors = self._get_user_behaviors(user_session)

            # Find similar users
            similar_users = self._find_similar_users(user_session, user_behaviors)

            # Recommend based on similar user preferences
            for memory in candidate_memories:
                similarity_score = self._calculate_collaborative_score(memory, similar_users)

                if similarity_score > self.similarity_threshold:
                    recommendation = MemoryRecommendation(
                        memory_id=memory.id,
                        memory_content=memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        recommendation_type=RecommendationType.COLLABORATIVE,
                        confidence_score=similarity_score,
                        relevance_score=similarity_score * 0.9,
                        reason=RecommendationReason.SIMILAR_USERS,
                        explanation="Similar users frequently access this type of memory",
                        related_memories=[],
                        context_factors={'similar_users_count': len(similar_users)},
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Collaborative filtering failed: {e}")

        return recommendations

    def _content_based_filtering(self, context: str, candidate_memories: List[Memory]) -> List[MemoryRecommendation]:
        """Content-based filtering recommendation"""
        recommendations = []

        try:
            context_keywords = self._extract_context_keywords(context)

            for memory in candidate_memories:
                # Calculate content similarity
                content_similarity = self._calculate_content_similarity(context, memory.content)

                # Calculate keyword match
                keyword_match = self._calculate_keyword_match(context_keywords, memory.content)

                # Comprehensive scoring
                total_score = (content_similarity * 0.7 + keyword_match * 0.3)

                if total_score > self.similarity_threshold:
                    recommendation = MemoryRecommendation(
                        memory_id=memory.id,
                        memory_content=memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        recommendation_type=RecommendationType.CONTENT_BASED,
                        confidence_score=total_score,
                        relevance_score=content_similarity,
                        reason=RecommendationReason.SIMILAR_CONTENT,
                        explanation=f"Content similarity: {content_similarity:.2f}",
                        related_memories=[],
                        context_factors={
                            'content_similarity': content_similarity,
                            'keyword_match': keyword_match
                        },
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Content-based filtering failed: {e}")

        return recommendations

    def _causal_based_recommendation(self, context: str, candidate_memories: List[Memory]) -> List[MemoryRecommendation]:
        """Causal relationship-based recommendation"""
        recommendations = []

        try:
            if not self.causal_analyzer:
                return recommendations

            # Analyze causal relationships
            causal_relations = self.causal_analyzer.analyze_causal_chain(candidate_memories)

            for memory in candidate_memories:
                # Find causal relationships related to current context
                causal_score = self._calculate_causal_relevance(memory, context, causal_relations)

                if causal_score > self.similarity_threshold:
                    recommendation = MemoryRecommendation(
                        memory_id=memory.id,
                        memory_content=memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        recommendation_type=RecommendationType.CAUSAL,
                        confidence_score=causal_score,
                        relevance_score=causal_score * 0.8,
                        reason=RecommendationReason.CAUSAL_RELATION,
                        explanation="Recommended based on causal relationship chain",
                        related_memories=[],
                        context_factors={'causal_score': causal_score},
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Causal-based recommendation failed: {e}")

        return recommendations

    def _predictive_recommendation(self, context: str, candidate_memories: List[Memory],
                                 project_path: Optional[str]) -> List[MemoryRecommendation]:
        """Predictive recommendation"""
        recommendations = []

        try:
            # Predict based on project phase
            if project_path:
                project_phase = self._identify_project_phase(project_path)
                phase_relevant_memories = self._filter_by_project_phase(candidate_memories, project_phase)

                for memory in phase_relevant_memories:
                    predictive_score = self._calculate_predictive_score(memory, context, project_phase)

                    if predictive_score > self.similarity_threshold:
                        recommendation = MemoryRecommendation(
                            memory_id=memory.id,
                            memory_content=memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                            recommendation_type=RecommendationType.PREDICTIVE,
                            confidence_score=predictive_score,
                            relevance_score=predictive_score * 0.85,
                            reason=RecommendationReason.FREQUENT_PATTERN,
                            explanation=f"Predicted based on project phase ({project_phase})",
                            related_memories=[],
                            context_factors={'project_phase': project_phase},
                            created_at=datetime.now()
                        )
                        recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Predictive recommendation failed: {e}")

        return recommendations

    def _hybrid_recommendation(self, context: str, candidate_memories: List[Memory],
                             user_session: Optional[str], project_path: Optional[str]) -> List[MemoryRecommendation]:
        """Hybrid recommendation"""
        try:
            # Get various recommendations
            collaborative_recs = self._collaborative_filtering(context, candidate_memories, user_session)
            content_recs = self._content_based_filtering(context, candidate_memories)
            causal_recs = self._causal_based_recommendation(context, candidate_memories)
            predictive_recs = self._predictive_recommendation(context, candidate_memories, project_path)

            # Merge recommendations
            all_recs = collaborative_recs + content_recs + causal_recs + predictive_recs

            # Calculate hybrid scores
            hybrid_recs = []
            memory_scores = defaultdict(list)

            for rec in all_recs:
                memory_scores[rec.memory_id].append(rec)

            for memory_id, recs in memory_scores.items():
                if len(recs) > 1:  # Multiple recommendation types support
                    # Calculate weighted average score
                    weights = {
                        RecommendationType.COLLABORATIVE: 0.3,
                        RecommendationType.CONTENT_BASED: 0.4,
                        RecommendationType.CAUSAL: 0.2,
                        RecommendationType.PREDICTIVE: 0.1
                    }

                    total_score = 0.0
                    total_weight = 0.0
                    explanations = []

                    for rec in recs:
                        weight = weights.get(rec.recommendation_type, 0.1)
                        total_score += rec.confidence_score * weight
                        total_weight += weight
                        explanations.append(f"{rec.recommendation_type.value}: {rec.confidence_score:.2f}")

                    if total_weight > 0:
                        hybrid_score = total_score / total_weight

                        # Create hybrid recommendation
                        base_rec = recs[0]  # Use first recommendation as base
                        hybrid_rec = MemoryRecommendation(
                            memory_id=memory_id,
                            memory_content=base_rec.memory_content,
                            recommendation_type=RecommendationType.HYBRID,
                            confidence_score=hybrid_score,
                            relevance_score=hybrid_score,
                            reason=RecommendationReason.SIMILAR_CONTENT,  # Primary reason
                            explanation=f"Hybrid recommendation: {'; '.join(explanations)}",
                            related_memories=[],
                            context_factors={'recommendation_count': len(recs)},
                            created_at=datetime.now()
                        )
                        hybrid_recs.append(hybrid_rec)

            return hybrid_recs

        except Exception as e:
            self.logger.error(f"Hybrid recommendation failed: {e}")
            return []

    def _extract_context_keywords(self, context: str) -> List[str]:
        """Extract context keywords"""
        # Simplified keyword extraction
        import re
        words = re.findall(r'\b\w+\b', context.lower())
        # Filter stop words (Chinese and English common particles)
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',  # Chinese stop words
                      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}  # English stop words
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        return keywords[:10]  # Return top 10 keywords

    def _calculate_content_similarity(self, context: str, content: str) -> float:
        """Calculate content similarity"""
        # Simplified similarity calculation
        context_words = set(self._extract_context_keywords(context))
        content_words = set(self._extract_context_keywords(content))

        if not context_words or not content_words:
            return 0.0

        intersection = context_words.intersection(content_words)
        union = context_words.union(content_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_keyword_match(self, keywords: List[str], content: str) -> float:
        """Calculate keyword match score"""
        if not keywords:
            return 0.0

        content_lower = content.lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)

        return matches / len(keywords)

    def _deduplicate_recommendations(self, recommendations: List[MemoryRecommendation]) -> List[MemoryRecommendation]:
        """Deduplicate recommendations"""
        seen_memories = set()
        unique_recs = []

        for rec in recommendations:
            if rec.memory_id not in seen_memories:
                seen_memories.add(rec.memory_id)
                unique_recs.append(rec)

        return unique_recs

    def _rank_recommendations(self, recommendations: List[MemoryRecommendation], context: str) -> List[MemoryRecommendation]:
        """Rank recommendations"""
        # Sort by confidence and relevance
        return sorted(recommendations, key=lambda r: (r.confidence_score + r.relevance_score) / 2, reverse=True)

    def _apply_diversity_and_novelty(self, recommendations: List[MemoryRecommendation]) -> List[MemoryRecommendation]:
        """Apply diversity and novelty"""
        if len(recommendations) <= 3:
            return recommendations

        # Simplified diversity implementation: ensure different recommendation types are represented
        diverse_recs = []
        type_counts = defaultdict(int)

        for rec in recommendations:
            rec_type = rec.recommendation_type
            if type_counts[rec_type] < 3:  # Maximum 3 per type
                diverse_recs.append(rec)
                type_counts[rec_type] += 1

        return diverse_recs

    def _record_recommendations(self, user_session: str, recommendations: List[MemoryRecommendation]):
        """Record recommendation history"""
        try:
            for rec in recommendations:
                query = """
                    INSERT INTO recommendation_history (
                        user_session, recommended_memory_id, recommendation_type,
                        recommendation_source, confidence_score, relevance_score,
                        context_query, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

                params = (
                    user_session,
                    rec.memory_id,
                    rec.recommendation_type.value,
                    'PersonalizedRecommender',
                    rec.confidence_score,
                    rec.relevance_score,
                    rec.explanation,
                    rec.created_at
                )

                self.memory_repo.execute_query(query, params)

        except Exception as e:
            self.logger.error(f"Failed to record recommendation history: {e}")

    # Simplified implementation helper methods
    def _get_user_behaviors(self, user_session: str) -> List[Dict]:
        """Get user behaviors"""
        return []  # Simplified implementation

    def _find_similar_users(self, user_session: str, behaviors: List[Dict]) -> List[str]:
        """Find similar users"""
        return []  # Simplified implementation

    def _calculate_collaborative_score(self, memory: Memory, similar_users: List[str]) -> float:
        """Calculate collaborative filtering score"""
        return 0.5  # Simplified implementation

    def _calculate_causal_relevance(self, memory: Memory, context: str, causal_relations: List) -> float:
        """Calculate causal relevance"""
        return 0.4  # Simplified implementation

    def _identify_project_phase(self, project_path: str) -> str:
        """Identify project phase"""
        return "development"  # Simplified implementation

    def _filter_by_project_phase(self, memories: List[Memory], phase: str) -> List[Memory]:
        """Filter by project phase"""
        return memories[:5]  # Simplified implementation

    def _calculate_predictive_score(self, memory: Memory, context: str, phase: str) -> float:
        """Calculate predictive score"""
        return 0.6  # Simplified implementation

    def _analyze_tech_stack_gaps(self, memories: List[Memory], project_path: str) -> List[KnowledgeGap]:
        """Analyze technology stack gaps"""
        return []  # Simplified implementation

    def _analyze_concept_gaps(self, memories: List[Memory]) -> List[KnowledgeGap]:
        """Analyze concept gaps"""
        return []  # Simplified implementation

    def _analyze_temporal_gaps(self, memories: List[Memory]) -> List[KnowledgeGap]:
        """Analyze timeliness gaps"""
        return []  # Simplified implementation

    def _predict_task_based_needs(self, task: str, project_path: Optional[str]) -> List[InfoNeed]:
        """Predict needs based on task"""
        return []  # Simplified implementation

    def _predict_behavior_based_needs(self, user_session: str, task: str) -> List[InfoNeed]:
        """Predict needs based on behavior"""
        return []  # Simplified implementation

    def _predict_phase_based_needs(self, project_path: str, task: str) -> List[InfoNeed]:
        """Predict needs based on phase"""
        return []  # Simplified implementation

    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Get recommendation statistics"""
        try:
            return {
                'total_recommendations': len(self._recommendation_history),
                'recommendation_types': dict(Counter(
                    rec.recommendation_type.value
                    for recs in self._recommendation_history.values()
                    for rec in recs
                )),
                'average_confidence': 0.65,  # Simplified implementation
                'cache_size': len(self._user_profiles)
            }

        except Exception as e:
            self.logger.error(f"Failed to get recommendation statistics: {e}")
            return {}