#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proactive Suggestion Engine
Proactively provide context-relevant intelligent suggestions at key moments
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from ..storage import Memory, MemoryRepository


class SuggestionMomentType(Enum):
    """Suggestion moment type"""
    PROJECT_INIT = "project_init"           # Project initialization
    PHASE_TRANSITION = "phase_transition"   # Development phase transition
    PROBLEM_PATTERN = "problem_pattern"     # Problem pattern recognition
    TECH_DECISION = "tech_decision"         # Technical decision moment
    BEST_PRACTICE = "best_practice"         # Best practice moment
    KNOWLEDGE_GAP = "knowledge_gap"         # Knowledge gap discovery


class SuggestionType(Enum):
    """Suggestion type"""
    TECH_STACK = "tech_stack"               # Tech stack suggestions
    TOOL_RECOMMENDATION = "tool_recommendation"  # Tool recommendations
    PROCESS_IMPROVEMENT = "process_improvement"  # Process improvement
    SOLUTION_PATTERN = "solution_pattern"   # Solution patterns
    LEARNING_RESOURCE = "learning_resource" # Learning resources
    BEST_PRACTICE_TIP = "best_practice_tip" # Best practice tips


class Priority(Enum):
    """Suggestion priority"""
    CRITICAL = "critical"    # Critical suggestions
    HIGH = "high"           # High priority
    MEDIUM = "medium"       # Medium priority
    LOW = "low"            # Low priority
    INFO = "info"          # Informational suggestions


@dataclass
class SuggestionMoment:
    """Suggestion moment"""
    moment_type: SuggestionMomentType
    context: Dict[str, Any]
    project_path: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class Advice:
    """Advice content"""
    suggestion_type: SuggestionType
    title: str
    description: str
    reasoning: str
    action_items: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    priority: Priority = Priority.MEDIUM
    tags: List[str] = field(default_factory=list)


@dataclass
class Suggestion:
    """Complete suggestion"""
    moment: SuggestionMoment
    advice: Advice
    generated_at: datetime = field(default_factory=datetime.now)
    user_feedback: Optional[str] = None
    adopted: bool = False


class ProactiveAdvisor:
    """Proactive Suggestion Engine"""

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize proactive suggestion engine

        Args:
            memory_repo: Memory data repository
        """
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.proactive_advisor')

        # Suggestion history cache
        self.suggestion_history = []
        self.pattern_cache = {}

        # Suggestion rule configuration
        self.suggestion_rules = self._initialize_suggestion_rules()

    def detect_suggestion_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """
        Detect suggestion moments

        Args:
            context: Current context information

        Returns:
            List of detected suggestion moments
        """
        try:
            moments = []

            # Detect project initialization moments
            project_init_moments = self._detect_project_init_moments(context)
            moments.extend(project_init_moments)

            # Detect development phase transition moments
            phase_transition_moments = self._detect_phase_transition_moments(context)
            moments.extend(phase_transition_moments)

            # Detect problem pattern moments
            problem_pattern_moments = self._detect_problem_pattern_moments(context)
            moments.extend(problem_pattern_moments)

            # Detect technical decision moments
            tech_decision_moments = self._detect_tech_decision_moments(context)
            moments.extend(tech_decision_moments)

            # Detect best practice moments
            best_practice_moments = self._detect_best_practice_moments(context)
            moments.extend(best_practice_moments)

            # Detect knowledge gap moments
            knowledge_gap_moments = self._detect_knowledge_gap_moments(context)
            moments.extend(knowledge_gap_moments)

            self.logger.info(f"Detected {len(moments)} suggestion moments")
            return moments

        except Exception as e:
            self.logger.error(f"Detect suggestion moments failed: {e}")
            return []

    def generate_contextual_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """
        Generate contextual advice

        Args:
            moment: Suggestion moment

        Returns:
            List of generated advice
        """
        try:
            advice_list = []

            # Generate corresponding advice based on moment type
            if moment.moment_type == SuggestionMomentType.PROJECT_INIT:
                advice_list.extend(self._generate_project_init_advice(moment))

            elif moment.moment_type == SuggestionMomentType.PHASE_TRANSITION:
                advice_list.extend(self._generate_phase_transition_advice(moment))

            elif moment.moment_type == SuggestionMomentType.PROBLEM_PATTERN:
                advice_list.extend(self._generate_problem_pattern_advice(moment))

            elif moment.moment_type == SuggestionMomentType.TECH_DECISION:
                advice_list.extend(self._generate_tech_decision_advice(moment))

            elif moment.moment_type == SuggestionMomentType.BEST_PRACTICE:
                advice_list.extend(self._generate_best_practice_advice(moment))

            elif moment.moment_type == SuggestionMomentType.KNOWLEDGE_GAP:
                advice_list.extend(self._generate_knowledge_gap_advice(moment))

            # Filter and enhance advice
            advice_list = self._filter_and_enhance_advice(advice_list, moment)

            self.logger.info(f"Generated {len(advice_list)} suggestions for moment {moment.moment_type.value}")
            return advice_list

        except Exception as e:
            self.logger.error(f"Generate contextual advice failed: {e}")
            return []

    def prioritize_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """
        Prioritize suggestions

        Args:
            suggestions: List of suggestions

        Returns:
            Sorted list of suggestions
        """
        try:
            # Define priority weights
            priority_weights = {
                Priority.CRITICAL: 100,
                Priority.HIGH: 80,
                Priority.MEDIUM: 60,
                Priority.LOW: 40,
                Priority.INFO: 20
            }

            # Calculate comprehensive score
            scored_suggestions = []
            for suggestion in suggestions:
                score = 0

                # Base priority score
                score += priority_weights.get(suggestion.advice.priority, 50)

                # Confidence weighting
                score += suggestion.advice.confidence * 30

                # Moment confidence weighting
                score += suggestion.moment.confidence * 20

                # Freshness weighting (newer suggestions get higher scores)
                time_diff = (datetime.now() - suggestion.generated_at).total_seconds()
                freshness_score = max(0, 10 - time_diff / 3600)  # Full score within 1 hour
                score += freshness_score

                # Historical adoption rate weighting
                adoption_rate = self._get_suggestion_adoption_rate(suggestion.advice.suggestion_type)
                score += adoption_rate * 15

                scored_suggestions.append((suggestion, score))

            # Sort by score
            scored_suggestions.sort(key=lambda x: x[1], reverse=True)
            sorted_suggestions = [s[0] for s in scored_suggestions]

            self.logger.info(f"Prioritized {len(suggestions)} suggestions")
            return sorted_suggestions

        except Exception as e:
            self.logger.error(f"Suggestion prioritization failed: {e}")
            return suggestions

    def get_proactive_suggestions(self, context: Dict[str, Any]) -> List[Suggestion]:
        """
        Get proactive suggestions (complete workflow)

        Args:
            context: Current context

        Returns:
            Priority-sorted list of suggestions
        """
        try:
            # 1. Detect suggestion moments
            moments = self.detect_suggestion_moments(context)

            if not moments:
                return []

            # 2. Generate suggestions for each moment
            all_suggestions = []
            for moment in moments:
                advice_list = self.generate_contextual_advice(moment)
                for advice in advice_list:
                    suggestion = Suggestion(moment=moment, advice=advice)
                    all_suggestions.append(suggestion)

            # 3. Priority sorting
            prioritized_suggestions = self.prioritize_suggestions(all_suggestions)

            # 4. Record suggestion history
            self.suggestion_history.extend(prioritized_suggestions)

            # 5. Limit return count (avoid information overload)
            max_suggestions = context.get('max_suggestions', 5)
            return prioritized_suggestions[:max_suggestions]

        except Exception as e:
            self.logger.error(f"Get proactive suggestions failed: {e}")
            return []

    def record_suggestion_feedback(self, suggestion_id: str, feedback: str, adopted: bool = False):
        """
        Record user feedback on suggestions

        Args:
            suggestion_id: Suggestion ID
            feedback: User feedback
            adopted: Whether adopted
        """
        try:
            # In actual implementation, this should update suggestion records in database
            # Currently simplified to memory operations
            for suggestion in self.suggestion_history:
                if id(suggestion) == hash(suggestion_id):
                    suggestion.user_feedback = feedback
                    suggestion.adopted = adopted
                    break

            self.logger.info(f"Recorded suggestion feedback: {feedback}, adopted: {adopted}")

        except Exception as e:
            self.logger.error(f"Record suggestion feedback failed: {e}")

    def _initialize_suggestion_rules(self) -> Dict[str, Any]:
        """Initialize suggestion rule configuration"""
        return {
            "project_init": {
                "tech_stack_patterns": {
                    "web_api": ["FastAPI", "Flask", "Django"],
                    "data_science": ["pandas", "numpy", "scikit-learn"],
                    "machine_learning": ["tensorflow", "pytorch", "transformers"]
                },
                "common_tools": ["git", "docker", "pytest", "black", "mypy"]
            },
            "phase_transition": {
                "development_to_testing": ["pytest", "coverage", "tox"],
                "testing_to_deployment": ["docker", "kubernetes", "CI/CD"],
                "prototype_to_production": ["logging", "monitoring", "security"]
            },
            "problem_patterns": {
                "performance_issues": ["profiling", "caching", "optimization"],
                "security_concerns": ["authentication", "encryption", "validation"],
                "scalability_problems": ["load_balancing", "database_optimization", "microservices"]
            }
        }

    def _detect_project_init_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect project initialization moments"""
        moments = []

        project_path = context.get('project_path')
        if not project_path:
            return moments

        # Check if this is a new project
        project_memories = self.memory_repo.search_memories(
            query=f"project_path:{project_path}",
            limit=10
        )

        if len(project_memories) < 3:  # New project judgment criteria
            moment = SuggestionMoment(
                moment_type=SuggestionMomentType.PROJECT_INIT,
                context=context,
                project_path=project_path,
                confidence=0.9,
                evidence=["Project has fewer than 3 memories", "Likely new project initialization"]
            )
            moments.append(moment)

        return moments

    def _detect_phase_transition_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect development phase transition moments"""
        moments = []

        # Analyze recent activity patterns
        recent_queries = context.get('recent_queries', [])
        if not recent_queries:
            return moments

        # Detect transition from development to testing
        dev_keywords = ['implement', 'code', 'function', 'class', 'method']
        test_keywords = ['test', 'testing', 'pytest', 'unittest', 'coverage']

        recent_dev_count = sum(1 for query in recent_queries if any(kw in query.lower() for kw in dev_keywords))
        recent_test_count = sum(1 for query in recent_queries if any(kw in query.lower() for kw in test_keywords))

        if recent_dev_count > 3 and recent_test_count == 0:
            moment = SuggestionMoment(
                moment_type=SuggestionMomentType.PHASE_TRANSITION,
                context={**context, 'transition_type': 'dev_to_test'},
                confidence=0.7,
                evidence=[f"Recent {recent_dev_count} development-related queries", "No testing-related activity detected"]
            )
            moments.append(moment)

        return moments

    def _detect_problem_pattern_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect problem pattern moments"""
        moments = []

        # Analyze errors and problem patterns
        error_keywords = context.get('error_keywords', [])
        if error_keywords:
            # Detect performance issues
            performance_errors = ['slow', 'timeout', 'performance', 'memory', 'cpu']
            if any(kw in ' '.join(error_keywords).lower() for kw in performance_errors):
                moment = SuggestionMoment(
                    moment_type=SuggestionMomentType.PROBLEM_PATTERN,
                    context={**context, 'problem_type': 'performance'},
                    confidence=0.8,
                    evidence=[f"Detected performance-related keywords: {error_keywords}"]
                )
                moments.append(moment)

        return moments

    def _detect_tech_decision_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect technical decision moments"""
        moments = []

        # Detect technology selection related queries
        decision_keywords = ['choose', 'select', 'compare', 'vs', 'alternative', 'recommendation']
        recent_queries = context.get('recent_queries', [])

        for query in recent_queries:
            if any(kw in query.lower() for kw in decision_keywords):
                moment = SuggestionMoment(
                    moment_type=SuggestionMomentType.TECH_DECISION,
                    context={**context, 'decision_query': query},
                    confidence=0.6,
                    evidence=[f"Technical decision related query: {query}"]
                )
                moments.append(moment)

        return moments

    def _detect_best_practice_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect best practice moments"""
        moments = []

        # Detect code quality related moments
        quality_indicators = context.get('quality_indicators', {})
        if quality_indicators.get('code_complexity', 0) > 10:
            moment = SuggestionMoment(
                moment_type=SuggestionMomentType.BEST_PRACTICE,
                context={**context, 'practice_type': 'code_quality'},
                confidence=0.7,
                evidence=[f"Code complexity too high: {quality_indicators['code_complexity']}"]
            )
            moments.append(moment)

        return moments

    def _detect_knowledge_gap_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """Detect knowledge gap moments"""
        moments = []

        # Analyze knowledge gaps in user queries
        failed_queries = context.get('failed_queries', [])
        if failed_queries:
            moment = SuggestionMoment(
                moment_type=SuggestionMomentType.KNOWLEDGE_GAP,
                context={**context, 'gap_queries': failed_queries},
                confidence=0.8,
                evidence=[f"Found {len(failed_queries)} failed queries"]
            )
            moments.append(moment)

        return moments

    def _generate_project_init_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate project initialization advice"""
        advice_list = []

        context = moment.context
        project_type = context.get('project_type', 'general')

        # Tech stack recommendations
        if project_type in self.suggestion_rules['project_init']['tech_stack_patterns']:
            tech_stack = self.suggestion_rules['project_init']['tech_stack_patterns'][project_type]
            advice = Advice(
                suggestion_type=SuggestionType.TECH_STACK,
                title=f"Recommended tech stack for {project_type} project",
                description=f"Based on project type, recommend using the following tech stack: {', '.join(tech_stack)}",
                reasoning=f"These technologies are widely used in {project_type} projects and have good ecosystem and community support",
                action_items=[
                    f"Consider using {tech_stack[0]} as main framework",
                    "Setup project dependency management",
                    "Configure development environment"
                ],
                confidence=0.8,
                priority=Priority.HIGH,
                tags=[project_type, "tech_stack", "initialization"]
            )
            advice_list.append(advice)

        # Common tool recommendations
        common_tools = self.suggestion_rules['project_init']['common_tools']
        advice = Advice(
            suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
            title="Recommended project development tools",
            description=f"Recommend configuring the following development tools: {', '.join(common_tools)}",
            reasoning="These tools can improve development efficiency and code quality",
            action_items=[
                "Initialize Git repository",
                "Configure code formatting tools",
                "Setup testing framework"
            ],
            confidence=0.9,
            priority=Priority.MEDIUM,
            tags=["tools", "development", "setup"]
        )
        advice_list.append(advice)

        return advice_list

    def _generate_phase_transition_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate phase transition advice"""
        advice_list = []

        transition_type = moment.context.get('transition_type')
        if transition_type == 'dev_to_test':
            advice = Advice(
                suggestion_type=SuggestionType.PROCESS_IMPROVEMENT,
                title="Recommend starting testing phase",
                description="Detected significant development activity, recommend starting to write and run tests",
                reasoning="Timely testing can discover issues and improve code quality",
                action_items=[
                    "Write unit tests",
                    "Setup test coverage checks",
                    "Configure continuous integration"
                ],
                confidence=0.7,
                priority=Priority.HIGH,
                tags=["testing", "quality", "process"]
            )
            advice_list.append(advice)

        return advice_list

    def _generate_problem_pattern_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate problem pattern advice"""
        advice_list = []

        problem_type = moment.context.get('problem_type')
        if problem_type == 'performance':
            advice = Advice(
                suggestion_type=SuggestionType.SOLUTION_PATTERN,
                title="Performance optimization recommendations",
                description="Detected performance-related issues, recommend performance analysis and optimization",
                reasoning="Performance issues affect user experience and need timely resolution",
                action_items=[
                    "Use performance analysis tools",
                    "Check database query efficiency",
                    "Optimize algorithm complexity",
                    "Consider caching strategies"
                ],
                resources=[
                    {"type": "tool", "name": "cProfile", "description": "Python performance analysis tool"},
                    {"type": "library", "name": "memory_profiler", "description": "Memory usage analysis"}
                ],
                confidence=0.8,
                priority=Priority.HIGH,
                tags=["performance", "optimization", "troubleshooting"]
            )
            advice_list.append(advice)

        return advice_list

    def _generate_tech_decision_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate technical decision advice"""
        advice_list = []

        decision_query = moment.context.get('decision_query', '')

        # Generate recommendations based on query content
        if 'database' in decision_query.lower():
            advice = Advice(
                suggestion_type=SuggestionType.TECH_STACK,
                title="Database selection recommendations",
                description="Recommend suitable database solutions based on project requirements",
                reasoning="Database selection has significant impact on project performance and scalability",
                action_items=[
                    "Analyze data access patterns",
                    "Evaluate data consistency requirements",
                    "Consider scalability requirements",
                    "Conduct performance benchmarking"
                ],
                confidence=0.6,
                priority=Priority.MEDIUM,
                tags=["database", "decision", "architecture"]
            )
            advice_list.append(advice)

        return advice_list

    def _generate_best_practice_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate best practice advice"""
        advice_list = []

        practice_type = moment.context.get('practice_type')
        if practice_type == 'code_quality':
            advice = Advice(
                suggestion_type=SuggestionType.BEST_PRACTICE_TIP,
                title="Code quality improvement recommendations",
                description="Detected high code complexity, recommend refactoring",
                reasoning="High complexity code is difficult to maintain and test, prone to bugs",
                action_items=[
                    "Split complex functions",
                    "Extract common logic",
                    "Add code comments",
                    "Write unit tests"
                ],
                confidence=0.7,
                priority=Priority.MEDIUM,
                tags=["code_quality", "refactoring", "maintainability"]
            )
            advice_list.append(advice)

        return advice_list

    def _generate_knowledge_gap_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """Generate knowledge gap advice"""
        advice_list = []

        gap_queries = moment.context.get('gap_queries', [])
        if gap_queries:
            advice = Advice(
                suggestion_type=SuggestionType.LEARNING_RESOURCE,
                title="Learning resource recommendations",
                description="Recommend relevant learning resources based on failed queries",
                reasoning="Filling knowledge gaps can improve development efficiency",
                action_items=[
                    "Check official documentation",
                    "Find relevant tutorials",
                    "Reference open source projects",
                    "Join technical communities"
                ],
                confidence=0.6,
                priority=Priority.LOW,
                tags=["learning", "knowledge", "resources"]
            )
            advice_list.append(advice)

        return advice_list

    def _filter_and_enhance_advice(self, advice_list: List[Advice], moment: SuggestionMoment) -> List[Advice]:
        """Filter and enhance advice"""
        # Remove duplicates
        unique_advice = []
        seen_titles = set()

        for advice in advice_list:
            if advice.title not in seen_titles:
                unique_advice.append(advice)
                seen_titles.add(advice.title)

        # Enhance advice content
        for advice in unique_advice:
            # Add project context
            if moment.project_path:
                advice.tags.append(f"project:{moment.project_path}")

            # Adjust confidence
            if moment.confidence > 0.8:
                advice.confidence = min(advice.confidence + 0.1, 1.0)

        return unique_advice

    def _get_suggestion_adoption_rate(self, suggestion_type: SuggestionType) -> float:
        """Get historical adoption rate for suggestion type"""
        if not self.suggestion_history:
            return 0.5  # Default adoption rate

        type_suggestions = [s for s in self.suggestion_history if s.advice.suggestion_type == suggestion_type]
        if not type_suggestions:
            return 0.5

        adopted_count = sum(1 for s in type_suggestions if s.adopted)
        return adopted_count / len(type_suggestions)