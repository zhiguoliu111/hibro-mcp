#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Reasoning System
Predicts user needs and project development trends based on historical patterns and causal relationships
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from .causal_analyzer import CausalAnalyzer
from .similarity import SimilarityCalculator
from ..storage.models import Memory, MemoryRepository
from .scorer import ImportanceScorer


class PredictionType(Enum):
    """Prediction types"""
    NEXT_NEED = "next_need"           # Next need prediction
    TECH_CHOICE = "tech_choice"       # Technology choice prediction
    PROJECT_PHASE = "project_phase"   # Project phase prediction
    IMPORTANCE_TREND = "importance_trend"  # Importance trend prediction


@dataclass
class Prediction:
    """Prediction result"""
    prediction_type: PredictionType
    content: str                      # Prediction content description
    confidence: float                 # Confidence level (0.0-1.0)
    probability: float                # Occurrence probability (0.0-1.0)
    evidence: List[str]               # Supporting evidence
    related_memories: List[int]       # Related memory IDs
    time_horizon: Optional[timedelta] # Prediction time range
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'prediction_type': self.prediction_type.value,
            'content': self.content,
            'confidence': self.confidence,
            'probability': self.probability,
            'evidence': self.evidence,
            'related_memories': self.related_memories,
            'time_horizon': self.time_horizon.total_seconds() if self.time_horizon else None,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class DecisionPattern:
    """Decision pattern"""
    pattern_id: str
    pattern_name: str
    trigger_conditions: List[str]     # Trigger conditions
    typical_sequence: List[str]       # Typical decision sequence
    success_rate: float               # Historical success rate
    usage_count: int                  # Usage count
    last_used: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'trigger_conditions': self.trigger_conditions,
            'typical_sequence': self.typical_sequence,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat()
        }


@dataclass
class ProjectPhase:
    """Project phase"""
    phase_name: str
    typical_duration: timedelta
    key_activities: List[str]
    required_decisions: List[str]
    next_phase: Optional[str]
    completion_indicators: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'phase_name': self.phase_name,
            'typical_duration': self.typical_duration.total_seconds(),
            'key_activities': self.key_activities,
            'required_decisions': self.required_decisions,
            'next_phase': self.next_phase,
            'completion_indicators': self.completion_indicators
        }


class PredictiveEngine:
    """Predictive reasoning system"""

    def __init__(self, causal_analyzer: CausalAnalyzer, scorer: ImportanceScorer,
                 similarity_calc: SimilarityCalculator, memory_repo: MemoryRepository):
        """
        Initialize prediction engine

        Args:
            causal_analyzer: Causal relationship analyzer
            scorer: Importance scorer
            similarity_calc: Semantic similarity calculator
            memory_repo: Memory repository
        """
        self.causal_analyzer = causal_analyzer
        self.scorer = scorer
        self.similarity_calc = similarity_calc
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.predictive_engine')

        # Project lifecycle phase definitions
        self.project_phases = {
            'initialization': ProjectPhase(
                phase_name='Project Initialization',
                typical_duration=timedelta(days=3),
                key_activities=['Requirements Analysis', 'Technology Selection', 'Architecture Design'],
                required_decisions=['Framework Selection', 'Database Selection', 'Deployment Method'],
                next_phase='development',
                completion_indicators=['Project Structure Created', 'Dependencies Installed', 'Basic Configuration']
            ),
            'development': ProjectPhase(
                phase_name='Development Phase',
                typical_duration=timedelta(days=21),
                key_activities=['Feature Development', 'Unit Testing', 'Code Review'],
                required_decisions=['API Design', 'Data Model', 'Business Logic'],
                next_phase='testing',
                completion_indicators=['Core Features Completed', 'Test Coverage Meets Requirements']
            ),
            'testing': ProjectPhase(
                phase_name='Testing Phase',
                typical_duration=timedelta(days=7),
                key_activities=['Integration Testing', 'Performance Testing', 'User Testing'],
                required_decisions=['Testing Strategy', 'Performance Optimization', 'User Feedback Handling'],
                next_phase='deployment',
                completion_indicators=['Tests Passed', 'Performance Meets Requirements', 'User Acceptance']
            ),
            'deployment': ProjectPhase(
                phase_name='Deployment Phase',
                typical_duration=timedelta(days=3),
                key_activities=['Environment Configuration', 'Deployment Scripts', 'Monitoring Setup'],
                required_decisions=['Deployment Strategy', 'Monitoring Solution', 'Backup Strategy'],
                next_phase='maintenance',
                completion_indicators=['Successful Deployment', 'Monitoring Operational', 'Documentation Complete']
            ),
            'maintenance': ProjectPhase(
                phase_name='Maintenance Phase',
                typical_duration=timedelta(days=365),
                key_activities=['Bug Fixes', 'Feature Enhancements', 'Performance Optimization'],
                required_decisions=['Version Planning', 'Technology Upgrades', 'Architecture Evolution'],
                next_phase=None,
                completion_indicators=['Stable Operation', 'User Satisfaction', 'Technical Debt Controlled']
            )
        }

        # Tech stack association patterns
        self.tech_associations = {
            'web_frontend': {
                'react': ['typescript', 'webpack', 'babel', 'jest'],
                'vue': ['typescript', 'vite', 'vitest', 'pinia'],
                'angular': ['typescript', 'rxjs', 'jasmine', 'karma']
            },
            'web_backend': {
                'fastapi': ['python', 'pydantic', 'sqlalchemy', 'pytest'],
                'django': ['python', 'django-rest-framework', 'celery', 'pytest'],
                'express': ['nodejs', 'typescript', 'mongoose', 'jest'],
                'spring': ['java', 'maven', 'hibernate', 'junit']
            },
            'database': {
                'postgresql': ['sqlalchemy', 'alembic', 'pgadmin'],
                'mongodb': ['mongoose', 'compass', 'atlas'],
                'redis': ['redis-py', 'redisinsight', 'cluster']
            },
            'deployment': {
                'docker': ['dockerfile', 'docker-compose', 'kubernetes'],
                'aws': ['ec2', 's3', 'rds', 'lambda'],
                'vercel': ['nextjs', 'serverless', 'edge-functions']
            }
        }

    def predict_next_needs(self, project_context: Dict[str, Any]) -> List[Prediction]:
        """
        Predict project's next needs

        Args:
            project_context: Project context information

        Returns:
            List of predicted needs
        """
        predictions = []
        project_path = project_context.get('project_path')

        if not project_path:
            return predictions

        # Get project-related memories
        project_memories = self.memory_repo.search_memories(limit=100)
        # Simplified: filter memories containing project path
        project_memories = [m for m in project_memories if project_path in str(m.metadata.get('project_path', ''))]

        self.logger.info(f"Predicting needs for project {project_path} based on {len(project_memories)} memories")

        # 1. Predict based on project phase
        phase_predictions = self._predict_by_project_phase(project_memories, project_context)
        predictions.extend(phase_predictions)

        # 2. Predict based on tech stack
        tech_predictions = self._predict_by_tech_stack(project_memories, project_context)
        predictions.extend(tech_predictions)

        # 3. Predict based on decision patterns
        pattern_predictions = self._predict_by_decision_patterns(project_memories)
        predictions.extend(pattern_predictions)

        # 4. Predict based on causal relationships
        causal_predictions = self._predict_by_causal_analysis(project_memories)
        predictions.extend(causal_predictions)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:10]  # Return top 10 most likely predictions

    def _predict_by_project_phase(self, memories: List[Memory], context: Dict[str, Any]) -> List[Prediction]:
        """Predict based on project phase"""
        predictions = []

        # Identify current project phase
        current_phase = self._identify_current_phase(memories)

        if current_phase and current_phase in self.project_phases:
            phase_info = self.project_phases[current_phase]

            # Predict next phase needs
            if phase_info.next_phase:
                next_phase_info = self.project_phases[phase_info.next_phase]

                for activity in next_phase_info.key_activities:
                    prediction = Prediction(
                        prediction_type=PredictionType.NEXT_NEED,
                        content=f"About to enter {next_phase_info.phase_name}, need to focus on: {activity}",
                        confidence=0.8,
                        probability=0.7,
                        evidence=[f"Current phase: {phase_info.phase_name}", f"Typical process: {activity}"],
                        related_memories=[m.id for m in memories[-5:] if m.id],  # Recent 5 memories
                        time_horizon=phase_info.typical_duration,
                        created_at=datetime.now()
                    )
                    predictions.append(prediction)

                for decision in next_phase_info.required_decisions:
                    prediction = Prediction(
                        prediction_type=PredictionType.TECH_CHOICE,
                        content=f"Need to make decision: {decision}",
                        confidence=0.75,
                        probability=0.8,
                        evidence=[f"Next phase: {next_phase_info.phase_name}", f"Required decision: {decision}"],
                        related_memories=[m.id for m in memories if m.id and decision.lower() in m.content.lower()],
                        time_horizon=timedelta(days=7),
                        created_at=datetime.now()
                    )
                    predictions.append(prediction)

        return predictions

    def _predict_by_tech_stack(self, memories: List[Memory], context: Dict[str, Any]) -> List[Prediction]:
        """Predict based on tech stack associations"""
        predictions = []

        # Extract used technologies
        used_technologies = set()
        for memory in memories:
            content_lower = memory.content.lower()
            for category, tech_map in self.tech_associations.items():
                for tech, related_techs in tech_map.items():
                    if tech in content_lower:
                        used_technologies.add(tech)

        # Predict related technology needs based on used technologies
        for tech in used_technologies:
            for category, tech_map in self.tech_associations.items():
                if tech in tech_map:
                    related_techs = tech_map[tech]
                    for related_tech in related_techs:
                        # Check if already used
                        already_used = any(related_tech in m.content.lower() for m in memories)
                        if not already_used:
                            prediction = Prediction(
                                prediction_type=PredictionType.TECH_CHOICE,
                                content=f"Consider using {related_tech}, works with existing tech stack {tech}",
                                confidence=0.6,
                                probability=0.5,
                                evidence=[f"Used technology: {tech}", f"Common pairing: {related_tech}"],
                                related_memories=[m.id for m in memories if m.id and tech in m.content.lower()],
                                time_horizon=timedelta(days=14),
                                created_at=datetime.now()
                            )
                            predictions.append(prediction)

        return predictions

    def _predict_by_decision_patterns(self, memories: List[Memory]) -> List[Prediction]:
        """Predict based on historical decision patterns"""
        predictions = []

        # Analyze decision patterns
        decision_patterns = self.analyze_decision_patterns(memories)

        for pattern in decision_patterns:
            if pattern.success_rate > 0.7:  # Only consider high-success patterns
                # Check if trigger conditions are met
                conditions_met = 0
                for condition in pattern.trigger_conditions:
                    if any(condition.lower() in m.content.lower() for m in memories[-10:]):  # Check recent 10 memories
                        conditions_met += 1

                if conditions_met >= len(pattern.trigger_conditions) * 0.6:  # 60% of conditions met
                    # Predict next decision
                    for next_step in pattern.typical_sequence:
                        already_done = any(next_step.lower() in m.content.lower() for m in memories)
                        if not already_done:
                            prediction = Prediction(
                                prediction_type=PredictionType.NEXT_NEED,
                                content=f"Based on historical patterns, suggest: {next_step}",
                                confidence=pattern.success_rate * 0.8,
                                probability=pattern.success_rate,
                                evidence=[f"Pattern: {pattern.pattern_name}", f"Success rate: {pattern.success_rate:.1%}"],
                                related_memories=[m.id for m in memories[-5:] if m.id],
                                time_horizon=timedelta(days=7),
                                created_at=datetime.now()
                            )
                            predictions.append(prediction)
                            break  # Only predict next step

        return predictions

    def _predict_by_causal_analysis(self, memories: List[Memory]) -> List[Prediction]:
        """Predict based on causal relationship analysis"""
        predictions = []

        # Get recent decision memories
        recent_decisions = [m for m in memories if m.memory_type == 'decision' and
                          m.created_at and (datetime.now() - m.created_at).days <= 7]

        for decision_memory in recent_decisions:
            if not decision_memory.id:
                continue

            # Predict effects this decision might produce
            predicted_effects = self.causal_analyzer.predict_effects(decision_memory.id, max_depth=2)

            for effect_relation in predicted_effects[:3]:  # Only take top 3 most likely effects
                effect_memory = self.memory_repo.get_memory(effect_relation.effect_memory_id)
                if effect_memory:
                    prediction = Prediction(
                        prediction_type=PredictionType.NEXT_NEED,
                        content=f"Decision '{decision_memory.content[:50]}...' may lead to need for: {effect_memory.content[:100]}...",
                        confidence=effect_relation.strength,
                        probability=effect_relation.confidence,
                        evidence=[f"Causal strength: {effect_relation.strength:.2f}", f"Historical pattern: {effect_relation.causal_type.value}"],
                        related_memories=[decision_memory.id, effect_memory.id],
                        time_horizon=timedelta(days=14),
                        created_at=datetime.now()
                    )
                    predictions.append(prediction)

        return predictions

    def _identify_current_phase(self, memories: List[Memory]) -> Optional[str]:
        """Identify current project phase"""
        if not memories:
            return None

        # Identify phase based on keywords in recent memories
        recent_memories = sorted(memories, key=lambda m: m.created_at or datetime.min, reverse=True)[:20]
        recent_content = ' '.join([m.content for m in recent_memories]).lower()

        phase_keywords = {
            'initialization': ['initialization', 'create project', 'technology selection', 'architecture design', 'requirements analysis', 'framework selection'],
            'development': ['development', 'implementation', 'coding', 'feature', 'api', 'database', 'business logic'],
            'testing': ['testing', 'debugging', 'bug', 'validation', 'integration', 'performance'],
            'deployment': ['deployment', 'release', 'launch', 'environment', 'configuration', 'monitoring'],
            'maintenance': ['maintenance', 'optimization', 'fix', 'upgrade', 'refactoring', 'monitoring']
        }

        phase_scores = {}
        for phase, keywords in phase_keywords.items():
            score = sum(1 for keyword in keywords if keyword in recent_content)
            phase_scores[phase] = score

        # Return highest scoring phase
        if phase_scores:
            best_phase = max(phase_scores, key=phase_scores.get)
            if phase_scores[best_phase] > 0:
                return best_phase

        return 'development'  # Default to development phase

    def analyze_decision_patterns(self, memories: List[Memory]) -> List[DecisionPattern]:
        """
        Analyze user decision patterns

        Args:
            memories: List of memories

        Returns:
            List of decision patterns
        """
        patterns = []

        # Extract decision memories
        decision_memories = [m for m in memories if m.memory_type == 'decision']

        if len(decision_memories) < 3:  # Need sufficient decision samples
            return patterns

        # Group by project for analysis
        project_groups = defaultdict(list)
        for memory in decision_memories:
            project_path = memory.metadata.get('project_path', 'global') if memory.metadata else 'global'
            project_groups[project_path].append(memory)

        for project_path, project_memories in project_groups.items():
            if len(project_memories) < 2:
                continue

            # Sort by time
            sorted_memories = sorted(project_memories, key=lambda m: m.created_at or datetime.min)

            # Extract decision sequence
            decision_sequence = []
            for memory in sorted_memories:
                # Simplify decision content, extract keywords
                keywords = self._extract_decision_keywords(memory.content)
                decision_sequence.extend(keywords)

            # Identify common patterns
            if len(decision_sequence) >= 3:
                pattern = DecisionPattern(
                    pattern_id=f"pattern_{project_path}_{len(patterns)}",
                    pattern_name=f"{project_path} Project Decision Pattern",
                    trigger_conditions=decision_sequence[:2],  # First two as trigger conditions
                    typical_sequence=decision_sequence[2:],    # Rest as typical sequence
                    success_rate=0.8,  # Simplified: assume 80% success rate
                    usage_count=len(project_memories),
                    last_used=sorted_memories[-1].created_at or datetime.now()
                )
                patterns.append(pattern)

        return patterns

    def _extract_decision_keywords(self, content: str) -> List[str]:
        """Extract keywords from decision content"""
        keywords = []
        content_lower = content.lower()

        # Technology keywords
        tech_keywords = ['python', 'javascript', 'react', 'vue', 'django', 'fastapi', 'postgresql', 'mongodb', 'docker', 'aws']
        for keyword in tech_keywords:
            if keyword in content_lower:
                keywords.append(keyword)

        # Decision type keywords
        decision_keywords = ['framework', 'database', 'deployment', 'testing', 'architecture', 'design', 'optimization']
        for keyword in decision_keywords:
            if keyword in content:
                keywords.append(keyword)

        return keywords[:3]  # Limit keyword count

    def recommend_proactive_info(self, current_context: str) -> List[Prediction]:
        """
        Proactively recommend relevant information based on current context

        Args:
            current_context: Current context description

        Returns:
            List of recommended information
        """
        predictions = []

        # Get all memories for semantic search
        all_memories = self.memory_repo.search_memories(limit=500)

        # Find memories related to current context
        similar_memories = []
        for memory in all_memories:
            similarity = self.similarity_calc.calculate_similarity(current_context, memory.content)
            if similarity > 0.4:  # Similarity threshold
                similar_memories.append((memory, similarity))

        # Sort by similarity
        similar_memories.sort(key=lambda x: x[1], reverse=True)

        # Generate recommendations based on related memories
        for memory, similarity in similar_memories[:5]:
            # Find other memories related to this memory
            related_memories = []
            for other_memory in all_memories:
                if other_memory.id != memory.id:
                    other_similarity = self.similarity_calc.calculate_similarity(memory.content, other_memory.content)
                    if other_similarity > 0.5:
                        related_memories.append(other_memory)

            if related_memories:
                # Generate recommendation
                recommendation_content = f"Based on '{memory.content[:50]}...', you may also need to know:"
                for related in related_memories[:2]:
                    recommendation_content += f"\n- {related.content[:100]}..."

                prediction = Prediction(
                    prediction_type=PredictionType.NEXT_NEED,
                    content=recommendation_content,
                    confidence=similarity * 0.8,
                    probability=similarity,
                    evidence=[f"Context similarity: {similarity:.2f}", f"Related memory count: {len(related_memories)}"],
                    related_memories=[memory.id] + [r.id for r in related_memories[:2] if r.id],
                    time_horizon=timedelta(days=1),
                    created_at=datetime.now()
                )
                predictions.append(prediction)

        return predictions

    def predict_importance_evolution(self, memory: Memory, time_horizon: timedelta = timedelta(days=30)) -> Prediction:
        """
        Predict memory importance evolution trend

        Args:
            memory: Target memory
            time_horizon: Prediction time range

        Returns:
            Importance evolution prediction
        """
        current_importance = memory.importance

        # Predict decay trend based on memory type
        decay_rates = {
            'preference': 0.02,    # Preference memories decay very slowly
            'decision': 0.05,      # Decision memories decay moderately
            'project': 0.08,       # Project memories decay faster
            'important': 0.01,     # Important memories barely decay
            'learning': 0.06,      # Learning memories decay moderately
            'conversation': 0.15   # Conversation memories decay quickly
        }

        decay_rate = decay_rates.get(memory.memory_type, 0.1)
        days = time_horizon.days

        # Simplified exponential decay model
        predicted_importance = current_importance * (1 - decay_rate) ** (days / 30)

        # Consider access frequency impact
        if memory.access_count > 5:  # High-frequency access memories decay slower
            predicted_importance *= 1.2

        predicted_importance = max(0.0, min(1.0, predicted_importance))

        # Determine trend
        if predicted_importance > current_importance * 0.9:
            trend = "Remains stable"
            confidence = 0.8
        elif predicted_importance > current_importance * 0.7:
            trend = "Slow decline"
            confidence = 0.7
        else:
            trend = "Significant decline"
            confidence = 0.6

        prediction = Prediction(
            prediction_type=PredictionType.IMPORTANCE_TREND,
            content=f"Importance will evolve from {current_importance:.2f} {trend} to approximately {predicted_importance:.2f}",
            confidence=confidence,
            probability=0.7,
            evidence=[
                f"Memory type: {memory.memory_type}",
                f"Decay rate: {decay_rate}",
                f"Access count: {memory.access_count}"
            ],
            related_memories=[memory.id] if memory.id else [],
            time_horizon=time_horizon,
            created_at=datetime.now()
        )

        return prediction

    def get_prediction_statistics(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """
        Get prediction statistics

        Args:
            predictions: List of predictions

        Returns:
            Statistics dictionary
        """
        if not predictions:
            return {}

        stats = {
            'total_predictions': len(predictions),
            'by_type': {},
            'confidence_distribution': {
                'high': 0,    # >= 0.7
                'medium': 0,  # 0.4-0.7
                'low': 0      # < 0.4
            },
            'avg_confidence': 0.0,
            'avg_probability': 0.0,
            'time_horizons': {}
        }

        # Statistics by type
        for prediction in predictions:
            pred_type = prediction.prediction_type.value
            if pred_type not in stats['by_type']:
                stats['by_type'][pred_type] = 0
            stats['by_type'][pred_type] += 1

            # Confidence distribution
            if prediction.confidence >= 0.7:
                stats['confidence_distribution']['high'] += 1
            elif prediction.confidence >= 0.4:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1

            # Time range statistics
            if prediction.time_horizon:
                days = prediction.time_horizon.days
                if days <= 7:
                    horizon = 'short_term'
                elif days <= 30:
                    horizon = 'medium_term'
                else:
                    horizon = 'long_term'

                if horizon not in stats['time_horizons']:
                    stats['time_horizons'][horizon] = 0
                stats['time_horizons'][horizon] += 1

        # Averages
        stats['avg_confidence'] = sum(p.confidence for p in predictions) / len(predictions)
        stats['avg_probability'] = sum(p.probability for p in predictions) / len(predictions)

        return stats