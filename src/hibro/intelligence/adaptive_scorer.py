#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Weight Adjustment Engine
Dynamically adjusts importance scoring weights based on user feedback and behavior patterns
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .scorer import ImportanceScorer, ImportanceFactor, ImportanceFactorType
from .behavior_analyzer import BehaviorAnalyzer, UserBehavior, FeedbackType
from ..storage.models import Memory, MemoryRepository


class LearningAlgorithm(Enum):
    """Learning algorithm types"""
    GRADIENT_DESCENT = "gradient_descent"
    REINFORCEMENT = "reinforcement"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"


class ABTestStatus(Enum):
    """A/B testing status"""
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class WeightAdjustment:
    """Weight adjustment record"""
    adjustment_type: str
    target_key: str
    old_value: float
    new_value: float
    adjustment_reason: str
    performance_metric: float
    user_feedback_score: float
    algorithm_used: str
    confidence: float
    project_path: Optional[str]
    timestamp: datetime


@dataclass
class LearningConfig:
    """Learning configuration"""
    config_key: str
    config_value: Dict[str, Any]
    config_type: str
    description: str
    is_active: bool
    version: str


@dataclass
class ABTestConfig:
    """A/B testing configuration"""
    test_id: str
    test_name: str
    control_weights: Dict[str, float]
    variant_weights: Dict[str, float]
    traffic_split: float  # 0.0-1.0, variant traffic proportion
    start_date: datetime
    end_date: Optional[datetime]
    status: ABTestStatus
    success_metric: str
    minimum_sample_size: int


class AdaptiveScorer(ImportanceScorer):
    """Adaptive importance scorer"""

    def __init__(self, config, memory_repo: MemoryRepository, behavior_analyzer: BehaviorAnalyzer):
        """
        Initialize adaptive scorer

        Args:
            config: Configuration object
            memory_repo: Memory repository
            behavior_analyzer: Behavior analyzer
        """
        super().__init__(config)
        self.memory_repo = memory_repo
        self.behavior_analyzer = behavior_analyzer
        self.logger = logging.getLogger('hibro.adaptive_scorer')

        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.001
        self.min_samples_for_adjustment = 10

        # Weight adjustment history
        self.adjustment_history = []

        # A/B test management
        self.active_ab_tests = {}

        # Performance metrics cache
        self._performance_cache = {}
        self._cache_ttl = timedelta(hours=1)

        # Initialize learning configuration
        self._initialize_learning_config()

    def _initialize_learning_config(self):
        """Initialize learning configuration"""
        try:
            # Default learning configuration
            default_configs = [
                LearningConfig(
                    config_key="gradient_descent_params",
                    config_value={
                        "learning_rate": 0.01,
                        "momentum": 0.9,
                        "weight_decay": 0.001,
                        "max_iterations": 100
                    },
                    config_type="learning_rate",
                    description="Gradient descent algorithm parameters",
                    is_active=True,
                    version="1.0"
                ),
                LearningConfig(
                    config_key="reinforcement_params",
                    config_value={
                        "exploration_rate": 0.1,
                        "discount_factor": 0.95,
                        "reward_decay": 0.9
                    },
                    config_type="reinforcement",
                    description="Reinforcement learning algorithm parameters",
                    is_active=True,
                    version="1.0"
                ),
                LearningConfig(
                    config_key="scoring_weights_adaptation",
                    config_value={
                        "adaptation_rate": 0.05,
                        "min_confidence": 0.6,
                        "max_weight_change": 0.2
                    },
                    config_type="scoring_weights",
                    description="Scoring weight adaptation parameters",
                    is_active=True,
                    version="1.0"
                )
            ]

            # Save configuration to database
            for config in default_configs:
                self._save_learning_config(config)

        except Exception as e:
            self.logger.error(f"Failed to initialize learning configuration: {e}")

    def adjust_factor_weights(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """
        Adjust factor weights based on user feedback

        Args:
            feedback_data: List of feedback data

        Returns:
            Whether adjustment was successful
        """
        try:
            if len(feedback_data) < self.min_samples_for_adjustment:
                self.logger.info(f"Insufficient feedback data, need at least {self.min_samples_for_adjustment} samples")
                return False

            # Analyze feedback patterns
            feedback_patterns = self._analyze_feedback_patterns(feedback_data)

            # Calculate weight adjustments
            weight_adjustments = self._calculate_weight_adjustments(feedback_patterns)

            # Apply weight adjustments
            for memory_type, adjustments in weight_adjustments.items():
                if memory_type in self.type_weights:
                    for factor_type, adjustment in adjustments.items():
                        old_weight = self.type_weights[memory_type].get(factor_type, 0.1)
                        new_weight = self._apply_weight_adjustment(old_weight, adjustment)

                        # Record adjustment
                        self._record_weight_adjustment(
                            adjustment_type="scoring_factor",
                            target_key=f"{memory_type}.{factor_type}",
                            old_value=old_weight,
                            new_value=new_weight,
                            adjustment_reason=f"Based on {len(feedback_data)} feedback samples",
                            performance_metric=feedback_patterns.get('avg_satisfaction', 0.0),
                            algorithm_used="gradient_descent"
                        )

                        # Update weights
                        self.type_weights[memory_type][factor_type] = new_weight

            self.logger.info(f"Adjusted weights based on {len(feedback_data)} feedback samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to adjust factor weights: {e}")
            return False

    def personalize_scoring(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Personalize scoring configuration

        Args:
            user_profile: User profile

        Returns:
            Personalized scoring configuration
        """
        try:
            # Base configuration
            personalized_config = {
                'type_weights': self.type_weights.copy(),
                'content_keywords': self.content_keywords.copy(),
                'personalization_factors': {}
            }

            # Adjust based on user preferences
            if 'preferred_memory_types' in user_profile:
                preferred_types = user_profile['preferred_memory_types']
                for memory_type in preferred_types:
                    if memory_type in personalized_config['type_weights']:
                        # Increase overall weight for preferred types
                        for factor_type in personalized_config['type_weights'][memory_type]:
                            personalized_config['type_weights'][memory_type][factor_type] *= 1.1

            # Adjust based on tech stack preferences
            if 'tech_stack_preferences' in user_profile:
                tech_prefs = user_profile['tech_stack_preferences']
                tech_keywords = []
                for tech, preference_score in tech_prefs.items():
                    if preference_score > 0.7:
                        tech_keywords.append(tech.lower())

                if tech_keywords:
                    personalized_config['content_keywords']['personal_tech'] = {
                        'keywords': tech_keywords,
                        'weight': 0.9
                    }

            # Adjust based on usage patterns
            if 'usage_patterns' in user_profile:
                patterns = user_profile['usage_patterns']

                # If user queries frequently, increase access pattern weight
                if patterns.get('query_frequency', 0) > 0.5:
                    for memory_type in personalized_config['type_weights']:
                        personalized_config['type_weights'][memory_type]['access_pattern'] *= 1.2

                # If user focuses on recency, increase temporal factor weight
                if patterns.get('recency_preference', 0) > 0.7:
                    for memory_type in personalized_config['type_weights']:
                        personalized_config['type_weights'][memory_type]['temporal'] *= 1.3

            personalized_config['personalization_factors'] = {
                'user_id': user_profile.get('user_id', 'default'),
                'personalization_strength': user_profile.get('personalization_strength', 0.5),
                'created_at': datetime.now().isoformat()
            }

            return personalized_config

        except Exception as e:
            self.logger.error(f"Failed to personalize scoring configuration: {e}")
            return {'type_weights': self.type_weights, 'content_keywords': self.content_keywords}

    def optimize_weights_online(self, performance_metrics: Dict[str, float]) -> bool:
        """
        Optimize weights online

        Args:
            performance_metrics: Performance metrics

        Returns:
            Whether optimization was successful
        """
        try:
            # Get current performance baseline
            baseline_metrics = self._get_performance_baseline()

            # Calculate performance improvements
            improvements = {}
            for metric, value in performance_metrics.items():
                baseline_value = baseline_metrics.get(metric, 0.5)
                improvement = (value - baseline_value) / max(baseline_value, 0.1)
                improvements[metric] = improvement

            # Adjust weights based on performance improvements
            if improvements.get('user_satisfaction', 0) > 0.05:  # User satisfaction improved by >5%
                self._apply_positive_reinforcement()
            elif improvements.get('user_satisfaction', 0) < -0.05:  # User satisfaction declined by >5%
                self._apply_negative_reinforcement()

            # Update performance baseline
            self._update_performance_baseline(performance_metrics)

            self.logger.info("Online weight optimization completed")
            return True

        except Exception as e:
            self.logger.error(f"Online weight optimization failed: {e}")
            return False

    def create_ab_test(self, test_config: ABTestConfig) -> bool:
        """
        Create A/B test

        Args:
            test_config: Test configuration

        Returns:
            Whether creation was successful
        """
        try:
            # Validate test configuration
            if not self._validate_ab_test_config(test_config):
                return False

            # Save test configuration
            self.active_ab_tests[test_config.test_id] = test_config

            # Log test start
            self.logger.info(f"A/B test '{test_config.test_name}' created, ID: {test_config.test_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            return False

    def get_ab_test_weights(self, test_id: str, user_session: str) -> Optional[Dict[str, float]]:
        """
        Get A/B test weights

        Args:
            test_id: Test ID
            user_session: User session

        Returns:
            Weight configuration
        """
        try:
            if test_id not in self.active_ab_tests:
                return None

            test_config = self.active_ab_tests[test_id]

            if test_config.status != ABTestStatus.RUNNING:
                return None

            # Determine group based on user session
            session_hash = hash(user_session) % 100
            use_variant = session_hash < (test_config.traffic_split * 100)

            if use_variant:
                return test_config.variant_weights
            else:
                return test_config.control_weights

        except Exception as e:
            self.logger.error(f"Failed to get A/B test weights: {e}")
            return None

    def evaluate_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Evaluate A/B test results

        Args:
            test_id: Test ID

        Returns:
            Test results
        """
        try:
            if test_id not in self.active_ab_tests:
                return {'error': 'Test not found'}

            test_config = self.active_ab_tests[test_id]

            # Collect test data
            control_metrics = self._collect_ab_test_metrics(test_id, 'control')
            variant_metrics = self._collect_ab_test_metrics(test_id, 'variant')

            # Calculate statistical significance
            significance = self._calculate_statistical_significance(control_metrics, variant_metrics)

            # Generate test report
            results = {
                'test_id': test_id,
                'test_name': test_config.test_name,
                'status': test_config.status.value,
                'control_metrics': control_metrics,
                'variant_metrics': variant_metrics,
                'statistical_significance': significance,
                'recommendation': self._generate_ab_test_recommendation(control_metrics, variant_metrics, significance),
                'evaluated_at': datetime.now().isoformat()
            }

            return results

        except Exception as e:
            self.logger.error(f"Failed to evaluate A/B test: {e}")
            return {'error': str(e)}

    def _analyze_feedback_patterns(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        patterns = {
            'total_feedback': len(feedback_data),
            'positive_feedback': 0,
            'negative_feedback': 0,
            'factor_performance': defaultdict(list),
            'memory_type_performance': defaultdict(list)
        }

        for feedback in feedback_data:
            # Count feedback types
            if feedback.get('user_feedback') in ['useful', 'very_useful']:
                patterns['positive_feedback'] += 1
            elif feedback.get('user_feedback') in ['not_useful']:
                patterns['negative_feedback'] += 1

            # Analyze factor performance
            if 'importance_factors' in feedback:
                for factor in feedback['importance_factors']:
                    factor_type = factor.get('factor_type')
                    factor_score = factor.get('score', 0)
                    patterns['factor_performance'][factor_type].append(factor_score)

            # Analyze memory type performance
            memory_type = feedback.get('memory_type', 'unknown')
            relevance = feedback.get('response_relevance', 0)
            patterns['memory_type_performance'][memory_type].append(relevance)

        # Calculate average performance
        patterns['avg_satisfaction'] = patterns['positive_feedback'] / max(patterns['total_feedback'], 1)

        return patterns

    def _calculate_weight_adjustments(self, feedback_patterns: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate weight adjustments"""
        adjustments = defaultdict(dict)

        # Adjust based on factor performance
        for factor_type, scores in feedback_patterns['factor_performance'].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                # If factor performs well, increase weight; if performs poorly, decrease weight
                adjustment = (avg_score - 0.5) * self.learning_rate

                # Apply to all memory types
                for memory_type in self.type_weights:
                    if factor_type in self.type_weights[memory_type]:
                        adjustments[memory_type][factor_type] = adjustment

        return adjustments

    def _apply_weight_adjustment(self, old_weight: float, adjustment: float) -> float:
        """Apply weight adjustment"""
        new_weight = old_weight + adjustment

        # Clamp weight range
        new_weight = max(0.01, min(1.0, new_weight))

        # Apply momentum and weight decay
        if hasattr(self, '_weight_momentum'):
            momentum_term = self.momentum * self._weight_momentum.get(f"{old_weight}", 0)
            new_weight += momentum_term

        # Weight decay
        new_weight *= (1 - self.weight_decay)

        return new_weight

    def _record_weight_adjustment(self, adjustment_type: str, target_key: str, old_value: float,
                                new_value: float, adjustment_reason: str, performance_metric: float,
                                algorithm_used: str, project_path: Optional[str] = None):
        """Record weight adjustment"""
        try:
            adjustment = WeightAdjustment(
                adjustment_type=adjustment_type,
                target_key=target_key,
                old_value=old_value,
                new_value=new_value,
                adjustment_reason=adjustment_reason,
                performance_metric=performance_metric,
                user_feedback_score=0.0,  # Can be calculated later
                algorithm_used=algorithm_used,
                confidence=0.8,
                project_path=project_path,
                timestamp=datetime.now()
            )

            # Save to database
            query = """
                INSERT INTO weight_adjustment_history (
                    adjustment_type, target_key, old_value, new_value,
                    adjustment_reason, performance_metric, user_feedback_score,
                    algorithm_used, confidence, project_path, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = (
                adjustment.adjustment_type,
                adjustment.target_key,
                adjustment.old_value,
                adjustment.new_value,
                adjustment.adjustment_reason,
                adjustment.performance_metric,
                adjustment.user_feedback_score,
                adjustment.algorithm_used,
                adjustment.confidence,
                adjustment.project_path,
                adjustment.timestamp
            )

            self.memory_repo.execute_query(query, params)
            self.adjustment_history.append(adjustment)

        except Exception as e:
            self.logger.error(f"Failed to record weight adjustment: {e}")

    def _save_learning_config(self, config: LearningConfig):
        """Save learning configuration"""
        try:
            query = """
                INSERT OR REPLACE INTO learning_config (
                    config_key, config_value, config_type, description, is_active, version
                ) VALUES (?, ?, ?, ?, ?, ?)
            """

            params = (
                config.config_key,
                json.dumps(config.config_value, ensure_ascii=False),
                config.config_type,
                config.description,
                config.is_active,
                config.version
            )

            self.memory_repo.execute_query(query, params)

        except Exception as e:
            self.logger.error(f"Failed to save learning configuration: {e}")

    def _get_performance_baseline(self) -> Dict[str, float]:
        """Get performance baseline"""
        # Simplified implementation, returns default baseline
        return {
            'user_satisfaction': 0.7,
            'relevance_score': 0.6,
            'click_through_rate': 0.3
        }

    def _update_performance_baseline(self, metrics: Dict[str, float]):
        """Update performance baseline"""
        # Simplified implementation, should actually save to database
        self._performance_cache.update(metrics)

    def _apply_positive_reinforcement(self):
        """Apply positive reinforcement"""
        # Increase weights for currently well-performing factors
        for memory_type in self.type_weights:
            for factor_type in self.type_weights[memory_type]:
                self.type_weights[memory_type][factor_type] *= 1.02

    def _apply_negative_reinforcement(self):
        """Apply negative reinforcement"""
        # Decrease weights for currently poorly-performing factors
        for memory_type in self.type_weights:
            for factor_type in self.type_weights[memory_type]:
                self.type_weights[memory_type][factor_type] *= 0.98

    def _validate_ab_test_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration"""
        if not config.test_id or not config.test_name:
            return False

        if not (0.0 <= config.traffic_split <= 1.0):
            return False

        if config.minimum_sample_size < 10:
            return False

        return True

    def _collect_ab_test_metrics(self, test_id: str, group: str) -> Dict[str, float]:
        """Collect A/B test metrics"""
        # Simplified implementation, should actually query from database
        return {
            'sample_size': 100,
            'user_satisfaction': 0.75,
            'relevance_score': 0.68,
            'click_through_rate': 0.32
        }

    def _calculate_statistical_significance(self, control_metrics: Dict[str, float],
                                          variant_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate statistical significance"""
        # Simplified implementation, should actually use statistical tests
        return {
            'p_value': 0.03,
            'confidence_level': 0.95,
            'is_significant': True,
            'effect_size': 0.15
        }

    def _generate_ab_test_recommendation(self, control_metrics: Dict[str, float],
                                       variant_metrics: Dict[str, float],
                                       significance: Dict[str, Any]) -> str:
        """Generate A/B test recommendation"""
        if significance.get('is_significant', False):
            if variant_metrics.get('user_satisfaction', 0) > control_metrics.get('user_satisfaction', 0):
                return "Recommend adopting variant configuration, user satisfaction significantly improved"
            else:
                return "Recommend keeping control configuration, variant configuration performs poorly"
        else:
            return "Test results have no statistical significance, recommend extending test time or increasing sample size"

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptive statistics"""
        try:
            return {
                'total_adjustments': len(self.adjustment_history),
                'active_ab_tests': len(self.active_ab_tests),
                'learning_rate': self.learning_rate,
                'recent_adjustments': [
                    {
                        'target': adj.target_key,
                        'change': adj.new_value - adj.old_value,
                        'reason': adj.adjustment_reason,
                        'timestamp': adj.timestamp.isoformat()
                    }
                    for adj in self.adjustment_history[-5:]
                ],
                'performance_cache': self._performance_cache
            }

        except Exception as e:
            self.logger.error(f"Failed to get adaptive statistics: {e}")
            return {}