#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Engine Core Module
Implements intelligent memory storage, retrieval and management functionality
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.config import Config
from ..utils.helpers import (
    sanitize_content,
    parse_memory_type,
    calculate_time_decay,
    format_timestamp
)
from ..storage import (
    DatabaseManager,
    Memory, Project, Preference,
    MemoryRepository, ProjectRepository,
    FileSystemManager
)
from ..intelligence import (
    SimilarityCalculator,
    ImportanceScorer
)
from ..intelligence.causal_analyzer import CausalAnalyzer, CausalRelation, CausalChain
from ..intelligence.predictive_engine import PredictiveEngine, Prediction, DecisionPattern
from ..intelligence.knowledge_graph import KnowledgeGraph, Concept, ConceptRelation
from ..intelligence.project_scanner import ProjectScanner
from ..assistant.proactive_advisor import ProactiveAdvisor, Suggestion, SuggestionMoment, Advice
from ..assistant.workflow_automator import WorkflowAutomator, Pattern, WorkflowTemplate, ExecutionResult
from ..assistant.reminder_system import ReminderSystem, Reminder, ReminderType, ReminderPriority
from ..assistant.intelligent_assistant import IntelligentAssistant, AssistantContext, AssistantResponse, AssistantConfig
from ..core.active_task import ActiveTaskManager
from ..core.lfu import LFUCalculator
from ..core.memory_cleaner import MemoryCleaner
from ..core.threshold_checker import ThresholdChecker
from ..core.cleanup_scheduler import CleanupScheduler
from ..security.security_monitoring_manager import SecurityMonitoringManager
from ..guidance.guidance_manager import GuidanceManager, UserLevel, GuidanceContext
from ..guidance.tool_recommender import ToolRecommender
from ..guidance.usage_hints import UsageHintProvider
from ..guidance.learning_path import LearningPathManager


class MemoryEngine:
    """Memory Engine Core Class"""

    def __init__(self, config: Config):
        """
        Initialize memory engine

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.memory_engine')

        # Initialize storage components
        self.db_manager = DatabaseManager(config)
        self.fs_manager = FileSystemManager(config)
        self.memory_repo = MemoryRepository(self.db_manager)
        self.project_repo = ProjectRepository(self.db_manager)

        # Initialize intelligent analysis components (lazy loading)
        self._similarity_calc = None
        self.importance_scorer = ImportanceScorer(config)
        self._causal_analyzer = None
        self._predictive_engine = None
        self._knowledge_graph = None

        # Initialize intelligent assistant components
        self.proactive_advisor = ProactiveAdvisor(self.memory_repo)
        self.workflow_automator = WorkflowAutomator(self.memory_repo)
        self.reminder_system = ReminderSystem(self.memory_repo)
        self.intelligent_assistant = IntelligentAssistant(self.memory_repo)

        # Initialize active task manager
        self.active_task_manager = ActiveTaskManager(self.memory_repo)

        # Initialize security monitoring manager
        self.security_monitoring_manager = SecurityMonitoringManager(config)

        # Initialize intelligent guidance system
        self.guidance_manager = GuidanceManager(config)
        self.tool_recommender = ToolRecommender(config)
        self.usage_hint_provider = UsageHintProvider(config)
        self.learning_path_manager = LearningPathManager(config)

        # Initialize project scanner
        self.project_scanner = ProjectScanner()

        # Initialize LFU calculator for cleanup
        self.lfu_calculator = LFUCalculator()

        # Initialize memory cleanup system
        self.cleaner = MemoryCleaner(self, self.lfu_calculator)
        self.threshold_checker = ThresholdChecker(
            memory_repo=self.memory_repo,
            cleaner=self.cleaner,
            max_memories=config.memory.max_memories
        )
        self.cleanup_scheduler = CleanupScheduler(
            cleaner=self.cleaner,
            config={
                'cleanup_time_of_day': getattr(config.forgetting, 'cleanup_time', '03:00'),
                'cleanup_enabled': getattr(config.forgetting, 'cleanup_enabled', True)
            }
        )

        # Initialize database watcher (for multi-session sync)
        self._db_watcher = None
        self._cache_version = 0  # Cache version number

        # Initialize event bus (for inter-component communication)
        from .event_bus import EventBus
        self._event_bus = EventBus(max_queue_size=1000, worker_threads=2)

        self._initialized = False

    @property
    def similarity_calc(self):
        """Lazy load similarity calculator"""
        if self._similarity_calc is None:
            self.logger.info("First time using semantic search, loading model...")
            self._similarity_calc = SimilarityCalculator()
        return self._similarity_calc

    @property
    def causal_analyzer(self):
        """Lazy load causal analyzer"""
        if self._causal_analyzer is None:
            self._causal_analyzer = CausalAnalyzer(self.similarity_calc, self.memory_repo)
        return self._causal_analyzer

    @property
    def predictive_engine(self):
        """Lazy load predictive engine"""
        if self._predictive_engine is None:
            self._predictive_engine = PredictiveEngine(
                self.causal_analyzer, self.importance_scorer,
                self.similarity_calc, self.memory_repo
            )
        return self._predictive_engine

    @property
    def knowledge_graph(self):
        """Lazy load knowledge graph"""
        if self._knowledge_graph is None:
            self._knowledge_graph = KnowledgeGraph(self.similarity_calc, self.causal_analyzer)
        return self._knowledge_graph

    def initialize(self):
        """Initialize memory engine"""
        try:
            # Initialize database
            self.db_manager.initialize_database()

            # Check database integrity
            if not self.db_manager.check_database_integrity():
                raise Exception("Database integrity check failed")

            # Start event bus
            self._event_bus.start()
            self.logger.info("Event bus started")

            # Start database watcher (for multi-session sync)
            self._start_database_watcher()

            # Start memory cleanup scheduler
            self.cleanup_scheduler.start()
            self.logger.info("Memory cleanup scheduler started")

            self._initialized = True
            self.logger.info("Memory engine initialization complete (database watcher, event bus, and cleanup scheduler enabled)")

        except Exception as e:
            self.logger.error(f"Memory engine initialization failed: {e}")
            raise

    def _start_database_watcher(self):
        """Start database watcher"""
        try:
            from .database_watcher import DatabaseWatcher

            db_path = self.db_manager.db_path
            self._db_watcher = DatabaseWatcher(
                db_path=db_path,
                on_change_callback=self._on_database_changed,
                debounce_seconds=1.0
            )
            self._db_watcher.start()
            self.logger.info(f"Database watcher started: {db_path}")

        except Exception as e:
            self.logger.warning(f"Failed to start database watcher (multi-session sync unavailable): {e}")
            self._db_watcher = None

    def _on_database_changed(self):
        """Database change callback - refresh cache and publish event"""
        self._cache_version += 1
        self.logger.info(
            f"Detected database change, cache refreshed (version: {self._cache_version})"
        )

        # Publish database change event
        try:
            from .event_bus import EventType
            self._event_bus.publish(
                event_type=EventType.DATABASE_CHANGED,
                data={
                    'cache_version': self._cache_version,
                    'timestamp': datetime.now().isoformat()
                },
                source='database_watcher'
            )
        except Exception as e:
            self.logger.warning(f"Failed to publish database change event: {e}")

    def refresh_cache(self):
        """Manually refresh cache"""
        self._cache_version += 1
        self.logger.info(f"Cache manually refreshed (version: {self._cache_version})")

    def store_memory(self, content: str, importance: float = 0.5,
                    category: Optional[str] = None,
                    memory_type: Optional[str] = None,
                    project_path: Optional[str] = None) -> int:
        """
        Store memory

        Args:
            content: Memory content
            importance: Importance score (0.0-1.0)
            category: Memory category
            memory_type: Memory type
            project_path: Associated project path

        Returns:
            Memory ID

        Raises:
            Exception: If storage is blocked due to memory capacity
        """
        if not self._initialized:
            self.initialize()

        # Check memory threshold before storing
        allow_store, message = self.threshold_checker.check_before_store()
        if not allow_store:
            raise Exception(message)
        if message:
            self.logger.warning(message)

        # Clean content
        cleaned_content = sanitize_content(content)

        # Auto-identify memory type
        if memory_type is None:
            memory_type = parse_memory_type(cleaned_content)

        # Create memory object
        memory = Memory(
            content=cleaned_content,
            memory_type=memory_type,
            importance=importance,
            category=category,
            metadata={
                'source': 'user_input',
                'project_path': project_path
            }
        )

        # Store to database
        memory_id = self.memory_repo.create_memory(memory)

        # If there's a project path, establish association
        if project_path:
            self._associate_memory_with_project(memory_id, project_path)

        # Save conversation records to file system
        if memory_type == 'conversation':
            project_name = Path(project_path).name if project_path else None
            self.fs_manager.save_conversation(cleaned_content, project_name)

        # Publish memory storage event
        try:
            from .event_bus import EventType

            # Publish general memory storage event
            self._event_bus.publish(
                event_type=EventType.MEMORY_STORED,
                data={
                    'memory_id': memory_id,
                    'memory_type': memory_type,
                    'category': category,
                    'importance': importance,
                    'project_path': project_path,
                    'content_preview': cleaned_content[:100]
                },
                source='memory_engine'
            )

            # If it's preference type, additionally publish preference change event
            if memory_type == 'preference' or category == 'code':
                self._event_bus.publish(
                    event_type=EventType.PREFERENCE_CHANGED,
                    data={
                        'memory_id': memory_id,
                        'category': category,
                        'content_preview': cleaned_content[:100]
                    },
                    source='memory_engine',
                    priority=1  # High priority
                )

        except Exception as e:
            self.logger.warning(f"Failed to publish memory storage event: {e}")

        self.logger.info(f"Memory stored: ID={memory_id}, type={memory_type}")
        return memory_id

    def recall_memories(self, query: str, limit: int = 5,
                       min_similarity: float = 0.3,
                       memory_type: Optional[str] = None,
                       category: Optional[str] = None,
                       project_path: Optional[str] = None) -> List[Memory]:
        """
        Recall related memories

        Args:
            query: Query content
            limit: Return quantity limit
            min_similarity: Minimum similarity threshold
            memory_type: Memory type filter
            category: Category filter
            project_path: Project path filter

        Returns:
            Related memory list
        """
        if not self._initialized:
            self.initialize()

        # Search memories from database
        memories = self.memory_repo.search_memories(
            query=query,
            memory_type=memory_type,
            category=category,
            limit=limit
        )

        # Update access information
        for memory in memories:
            memory.update_access()
            self.memory_repo.update_memory(memory)

        self.logger.info(f"Recall query: '{query}' -> {len(memories)} memories")
        return memories

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Get specified memory

        Args:
            memory_id: Memory ID

        Returns:
            Memory object, return None if not exists
        """
        if not self._initialized:
            self.initialize()

        memory = self.memory_repo.get_memory(memory_id)
        if memory:
            memory.update_access()
            self.memory_repo.update_memory(memory)

        return memory

    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete memory

        Args:
            memory_id: Memory ID

        Returns:
            Whether deletion was successful
        """
        if not self._initialized:
            self.initialize()

        return self.memory_repo.delete_memory(memory_id)

    def init_project(self, project_path: str, name: Optional[str] = None,
                    tech_stack: Optional[List[str]] = None,
                    description: Optional[str] = None):
        """
        Initialize project context

        Args:
            project_path: Project path
            name: Project name
            tech_stack: Technology stack
            description: Project description
        """
        if not self._initialized:
            self.initialize()

        project_path_obj = Path(project_path).resolve()

        # Check if project already exists
        existing_project = self.project_repo.get_project_by_path(str(project_path_obj))
        if existing_project:
            self.logger.info(f"Project already exists: {project_path_obj}")
            return existing_project.id

        # Create new project
        if name is None:
            name = project_path_obj.name

        project = Project(
            name=name,
            path=str(project_path_obj),
            tech_stack=tech_stack or [],
            description=description
        )

        project_id = self.project_repo.create_project(project)

        # Create project context snapshot
        self._create_project_snapshot(project_path_obj)

        self.logger.info(f"Project context initialized: {project_path_obj} (ID: {project_id})")
        return project_id

    def get_project_memories(self, project_path: str, limit: int = 50) -> List[Memory]:
        """
        Get project-related memories

        Args:
            project_path: Project path
            limit: Return quantity limit

        Returns:
            Project-related memory list
        """
        if not self._initialized:
            self.initialize()

        # Search for memories containing project path
        memories = self.memory_repo.search_memories(
            query=str(Path(project_path).name),
            limit=limit
        )

        return memories

    def export_memories(self, export_format: str = 'json',
                       memory_type: Optional[str] = None,
                       project_path: Optional[str] = None) -> Path:
        """
        Export memory data

        Args:
            export_format: Export format
            memory_type: Memory type filter
            project_path: Project path filter

        Returns:
            Export file path
        """
        if not self._initialized:
            self.initialize()

        # Get memories to export
        memories = self.memory_repo.search_memories(
            memory_type=memory_type,
            limit=10000  # Bulk export
        )

        # Convert to dictionary format
        memories_data = [memory.to_dict() for memory in memories]

        # Export using file system manager
        return self.fs_manager.export_memories(memories_data, export_format)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        # Get database statistics
        db_stats = self.db_manager.get_database_stats()

        # Get file system statistics
        fs_stats = self.fs_manager.get_storage_stats()

        # Merge statistics
        stats = {
            'total_memories': db_stats.get('memories_count', 0),
            'total_projects': db_stats.get('projects_count', 0),
            'total_preferences': db_stats.get('preferences_count', 0),
            'db_size_mb': db_stats.get('db_size_mb', 0.0),
            'fs_size_mb': fs_stats.get('total_size_mb', 0.0),
            'conversations_count': fs_stats.get('conversations', {}).get('count', 0),
            'contexts_count': fs_stats.get('contexts', {}).get('count', 0),
            'last_activity': db_stats.get('last_activity'),
            'last_updated': format_timestamp()
        }

        return stats

    def cleanup_old_memories(self, days_to_keep: int = 365,
                           min_importance: float = 0.1):
        """
        Clean up old memories

        Args:
            days_to_keep: Days to keep
            min_importance: Minimum importance threshold
        """
        if not self._initialized:
            self.initialize()

        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Search for memories to clean
            old_memories = self.memory_repo.search_memories(
                min_importance=0.0,  # Get all memories
                limit=10000
            )

            cleaned_count = 0
            for memory in old_memories:
                # Check if cleanup is needed
                if (memory.last_accessed and memory.last_accessed < cutoff_date and
                    memory.importance < min_importance):

                    if self.memory_repo.delete_memory(memory.id):
                        cleaned_count += 1

            # Clean old files in file system
            self.fs_manager.cleanup_old_files(days_to_keep)

            self.logger.info(f"Cleanup complete, cleaned {cleaned_count} memories")

        except Exception as e:
            self.logger.error(f"Failed to clean old memories: {e}")

    def _associate_memory_with_project(self, memory_id: int, project_path: str):
        """
        Associate memory with project

        Args:
            memory_id: Memory ID
            project_path: Project path
        """
        try:
            # Ensure project exists
            project = self.project_repo.get_project_by_path(project_path)
            if not project:
                # Auto-create project
                self.init_project(project_path)

        except Exception as e:
            self.logger.warning(f"Failed to associate memory with project: {e}")

    def _create_project_snapshot(self, project_path: Path):
        """
        Create project context snapshot

        Args:
            project_path: Project path
        """
        try:
            # Get project-related memories
            memories = self.get_project_memories(str(project_path))
            memories_data = [memory.to_dict() for memory in memories]

            # Save snapshot
            self.fs_manager.save_context_snapshot(str(project_path), memories_data)

        except Exception as e:
            self.logger.warning(f"Failed to create project snapshot: {e}")

    # ==================== Causal Analysis Features ====================

    def analyze_causal_relations(self, memories: Optional[List[Memory]] = None,
                                project_path: Optional[str] = None) -> List[CausalChain]:
        """
        Analyze causal relationship chains between memories

        Args:
            memories: List of memories to analyze, if None analyze all memories
            project_path: Project path filter, if specified only analyze memories of this project

        Returns:
            List of causal chains
        """
        if not self._initialized:
            self.initialize()

        # Get memories to analyze
        if memories is None:
            if project_path:
                memories = self.get_project_memories(project_path, limit=500)
            else:
                memories = self.memory_repo.search_memories(limit=500)

        # Use causal analyzer for analysis
        causal_chains = self.causal_analyzer.analyze_causal_chain(memories)

        self.logger.info(f"Causal analysis complete: found {len(causal_chains)} causal chains")
        return causal_chains

    def find_root_causes(self, effect_memory_id: int, max_depth: int = 3) -> List[CausalRelation]:
        """
        Find root causes of specified effect

        Args:
            effect_memory_id: Effect memory ID
            max_depth: Maximum search depth

        Returns:
            List of root causes
        """
        if not self._initialized:
            self.initialize()

        root_causes = self.causal_analyzer.find_root_causes(effect_memory_id, max_depth)

        self.logger.info(f"Root cause analysis complete: memory {effect_memory_id} found {len(root_causes)} root causes")
        return root_causes

    def predict_effects(self, cause_memory_id: int, max_depth: int = 3) -> List[CausalRelation]:
        """
        Predict possible effects of specified cause

        Args:
            cause_memory_id: Cause memory ID
            max_depth: Maximum prediction depth

        Returns:
            List of possible effects
        """
        if not self._initialized:
            self.initialize()

        predicted_effects = self.causal_analyzer.predict_effects(cause_memory_id, max_depth)

        self.logger.info(f"Effect prediction complete: memory {cause_memory_id} predicted {len(predicted_effects)} possible effects")
        return predicted_effects

    def get_causal_statistics(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get causal relationship statistics

        Args:
            project_path: Project path filter

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        # Get memories to analyze
        if project_path:
            memories = self.get_project_memories(project_path, limit=1000)
        else:
            memories = self.memory_repo.search_memories(limit=1000)

        # Get causal statistics
        causal_stats = self.causal_analyzer.get_causal_statistics(memories)

        return causal_stats

    def store_causal_relation(self, causal_relation: CausalRelation) -> bool:
        """
        Store causal relationship to database

        Args:
            causal_relation: Causal relationship object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO memory_relations
                    (memory_id, related_id, relation_type, strength, relation_strength,
                     causal_type, confidence_score, created_by, evidence, pattern_matched)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    causal_relation.cause_memory_id,
                    causal_relation.effect_memory_id,
                    'causal',
                    causal_relation.strength,
                    causal_relation.strength,
                    causal_relation.causal_type.value,
                    causal_relation.confidence,
                    'causal_analyzer',
                    str(causal_relation.evidence),
                    causal_relation.pattern_matched
                ))
                conn.commit()

            self.logger.info(f"Causal relationship stored: {causal_relation.cause_memory_id} -> {causal_relation.effect_memory_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store causal relationship: {e}")
            return False

    def store_causal_chain(self, causal_chain: CausalChain) -> bool:
        """
        Store causal chain to database

        Args:
            causal_chain: Causal chain object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            import json

            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO causal_chains
                    (chain_id, root_cause_memory_id, final_effect_memory_id,
                     chain_length, total_strength, relations_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    causal_chain.chain_id,
                    causal_chain.root_cause,
                    causal_chain.final_effect,
                    causal_chain.chain_length,
                    causal_chain.total_strength,
                    json.dumps(causal_chain.to_dict())
                ))
                conn.commit()

            self.logger.info(f"Causal chain stored: {causal_chain.chain_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store causal chain: {e}")
            return False

    # ==================== Predictive Analysis Features ====================

    def predict_next_needs(self, project_path: Optional[str] = None) -> List[Prediction]:
        """
        Predict next step requirements

        Args:
            project_path: Project path filter

        Returns:
            Predicted requirements list
        """
        if not self._initialized:
            self.initialize()

        project_context = {
            'project_path': project_path
        }

        predictions = self.predictive_engine.predict_next_needs(project_context)

        self.logger.info(f"Requirements prediction complete: generated {len(predictions)} predictions")
        return predictions

    def analyze_decision_patterns(self, project_path: Optional[str] = None) -> List[DecisionPattern]:
        """
        Analyze decision patterns

        Args:
            project_path: Project path filter

        Returns:
            Decision pattern list
        """
        if not self._initialized:
            self.initialize()

        # Get memories to analyze
        if project_path:
            memories = self.get_project_memories(project_path, limit=200)
        else:
            memories = self.memory_repo.search_memories(limit=200)

        patterns = self.predictive_engine.analyze_decision_patterns(memories)

        self.logger.info(f"Decision pattern analysis complete: found {len(patterns)} patterns")
        return patterns

    def recommend_proactive_info(self, current_context: str) -> List[Prediction]:
        """
        Proactively recommend relevant information based on current context

        Args:
            current_context: Current context description

        Returns:
            Recommended information list
        """
        if not self._initialized:
            self.initialize()

        recommendations = self.predictive_engine.recommend_proactive_info(current_context)

        self.logger.info(f"Proactive recommendation complete: generated {len(recommendations)} recommendations")
        return recommendations

    # ==================== Proactive Suggestion Features ====================

    def get_proactive_suggestions(self, context: Dict[str, Any]) -> List[Suggestion]:
        """
        Get proactive suggestions

        Args:
            context: Current context information, including:
                - project_path: Project path
                - recent_queries: Recent query list
                - error_keywords: Error keywords
                - project_type: Project type
                - quality_indicators: Code quality indicators
                - failed_queries: Failed query list
                - max_suggestions: Maximum suggestion count

        Returns:
            Priority-sorted suggestion list
        """
        if not self._initialized:
            self.initialize()

        suggestions = self.proactive_advisor.get_proactive_suggestions(context)

        self.logger.info(f"Proactive suggestion generation complete: obtained {len(suggestions)} suggestions")
        return suggestions

    def detect_suggestion_moments(self, context: Dict[str, Any]) -> List[SuggestionMoment]:
        """
        Detect suggestion moments

        Args:
            context: Current context information

        Returns:
            List of detected suggestion moments
        """
        if not self._initialized:
            self.initialize()

        moments = self.proactive_advisor.detect_suggestion_moments(context)

        self.logger.info(f"Suggestion moment detection complete: found {len(moments)} moments")
        return moments

    def generate_contextual_advice(self, moment: SuggestionMoment) -> List[Advice]:
        """
        Generate contextual advice for specific moments

        Args:
            moment: Suggestion moment

        Returns:
            Generated advice list
        """
        if not self._initialized:
            self.initialize()

        advice_list = self.proactive_advisor.generate_contextual_advice(moment)

        self.logger.info(f"Contextual advice generation complete: generated {len(advice_list)} suggestions for moment {moment.moment_type.value}")
        return advice_list

    def record_suggestion_feedback(self, suggestion_id: str, feedback: str, adopted: bool = False):
        """
        Record user feedback on suggestions

        Args:
            suggestion_id: Suggestion ID
            feedback: User feedback content
            adopted: Whether suggestion was adopted
        """
        if not self._initialized:
            self.initialize()

        self.proactive_advisor.record_suggestion_feedback(suggestion_id, feedback, adopted)

        self.logger.info(f"Suggestion feedback recorded: {feedback}, adoption status: {adopted}")

    def get_suggestion_statistics(self) -> Dict[str, Any]:
        """
        Get suggestion system statistics

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        history = self.proactive_advisor.suggestion_history

        if not history:
            return {
                "total_suggestions": 0,
                "adoption_rate": 0.0,
                "suggestion_types": {},
                "moment_types": {},
                "priority_distribution": {}
            }

        # Calculate statistics
        total_suggestions = len(history)
        adopted_count = sum(1 for s in history if s.adopted)
        adoption_rate = adopted_count / total_suggestions if total_suggestions > 0 else 0.0

        # Statistics by type
        suggestion_types = {}
        moment_types = {}
        priority_distribution = {}

        for suggestion in history:
            # Suggestion type statistics
            stype = suggestion.advice.suggestion_type.value
            suggestion_types[stype] = suggestion_types.get(stype, 0) + 1

            # Moment type statistics
            mtype = suggestion.moment.moment_type.value
            moment_types[mtype] = moment_types.get(mtype, 0) + 1

            # Priority statistics
            priority = suggestion.advice.priority.value
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

        return {
            "total_suggestions": total_suggestions,
            "adopted_suggestions": adopted_count,
            "adoption_rate": round(adoption_rate, 3),
            "suggestion_types": suggestion_types,
            "moment_types": moment_types,
            "priority_distribution": priority_distribution,
            "avg_confidence": round(
                sum(s.advice.confidence for s in history) / total_suggestions, 3
            ) if total_suggestions > 0 else 0.0
        }

    # ==================== Workflow Automator Interface ====================

    def detect_workflow_patterns(self, context: Dict[str, Any]) -> List[Pattern]:
        """
        Detect workflow patterns

        Args:
            context: Context information, including:
                - project_path: Project path
                - command_history: Command history
                - file_operations: File operation history
                - time_window: Time window (hours)

        Returns:
            List of detected patterns
        """
        if not self._initialized:
            self.initialize()

        patterns = self.workflow_automator.detect_patterns(context)

        self.logger.info(f"Detected {len(patterns)} workflow patterns")
        return patterns

    def get_workflow_recommendations(self, context: Dict[str, Any]) -> List[WorkflowTemplate]:
        """
        Get workflow recommendations

        Args:
            context: Context information, including:
                - project_path: Project path
                - project_type: Project type
                - detected_patterns: Detected patterns
                - max_recommendations: Maximum recommendation count

        Returns:
            Recommended workflow template list
        """
        if not self._initialized:
            self.initialize()

        recommendations = self.workflow_automator.recommend_workflows(context)

        self.logger.info(f"Generated {len(recommendations)} workflow recommendations")
        return recommendations

    def execute_workflow(self, template_id: str, variables: Dict[str, Any],
                        automation_level: str = "manual") -> ExecutionResult:
        """
        Execute workflow

        Args:
            template_id: Template ID
            variables: Variable dictionary
            automation_level: Automation level (manual/semi_auto/full_auto)

        Returns:
            Execution result
        """
        if not self._initialized:
            self.initialize()

        result = self.workflow_automator.execute_workflow(template_id, variables, automation_level)

        self.logger.info(f"Workflow execution complete: {template_id}, success: {result.success}")
        return result

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        Get workflow statistics

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        stats = self.workflow_automator.get_statistics()

        self.logger.info("Workflow statistics retrieved")
        return stats

    # ==================== Intelligent Reminder System Interface ====================

    def detect_reminder_moments(self, context: Dict[str, Any]) -> List[ReminderType]:
        """
        Detect reminder moments

        Args:
            context: Context information, including:
                - project_path: Project path
                - project_metrics: Project metrics
                - milestone_data: Milestone data
                - code_quality_metrics: Code quality metrics
                - security_issues: Security issue list

        Returns:
            List of reminder types that need to be triggered
        """
        if not self._initialized:
            self.initialize()

        reminder_types = self.reminder_system.detect_reminder_moments(context)

        self.logger.info(f"Detected {len(reminder_types)} reminder moment types")
        return reminder_types

    def generate_contextual_reminders(self, context: Dict[str, Any]) -> List[Reminder]:
        """
        Generate context-related reminders

        Args:
            context: Context information

        Returns:
            Generated reminder list
        """
        if not self._initialized:
            self.initialize()

        reminders = self.reminder_system.generate_contextual_reminders(context)

        self.logger.info(f"Generated {len(reminders)} contextual reminders")
        return reminders

    def schedule_reminder(self, reminder: Reminder) -> bool:
        """
        Schedule reminder

        Args:
            reminder: Reminder object

        Returns:
            Whether scheduling was successful
        """
        if not self._initialized:
            self.initialize()

        success = self.reminder_system.schedule_reminder(reminder)

        self.logger.info(f"Reminder scheduling {'successful' if success else 'failed'}: {reminder.title}")
        return success

    def get_active_reminders(self, project_path: Optional[str] = None,
                           reminder_type: Optional[ReminderType] = None,
                           priority: Optional[ReminderPriority] = None) -> List[Reminder]:
        """
        Get active reminders

        Args:
            project_path: Project path filter
            reminder_type: Reminder type filter
            priority: Priority filter

        Returns:
            Filtered reminder list
        """
        if not self._initialized:
            self.initialize()

        reminders = self.reminder_system.get_active_reminders(project_path, reminder_type, priority)

        self.logger.info(f"Retrieved {len(reminders)} active reminders")
        return reminders

    def dismiss_reminder(self, reminder_id: str) -> bool:
        """
        Dismiss reminder

        Args:
            reminder_id: Reminder ID

        Returns:
            Whether operation was successful
        """
        if not self._initialized:
            self.initialize()

        success = self.reminder_system.dismiss_reminder(reminder_id)

        self.logger.info(f"Reminder dismissal {'successful' if success else 'failed'}: {reminder_id}")
        return success

    def complete_reminder(self, reminder_id: str) -> bool:
        """
        Complete reminder

        Args:
            reminder_id: Reminder ID

        Returns:
            Whether operation was successful
        """
        if not self._initialized:
            self.initialize()

        success = self.reminder_system.complete_reminder(reminder_id)

        self.logger.info(f"Reminder completion {'successful' if success else 'failed'}: {reminder_id}")
        return success

    def get_reminder_statistics(self) -> Dict[str, Any]:
        """
        Get reminder system statistics

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        stats = self.reminder_system.get_statistics()

        self.logger.info("Reminder system statistics retrieved")
        return stats

    # ==================== Intelligent Assistant Unified Interface ====================

    def process_assistant_request(self, context: AssistantContext) -> AssistantResponse:
        """
        Process intelligent assistant request

        Args:
            context: Assistant context, including:
                - project_path: Project path
                - user_query: User query
                - recent_activities: Recent activities
                - project_metrics: Project metrics
                - error_context: Error context
                - time_context: Time context
                - user_preferences: User preferences

        Returns:
            Assistant response, containing suggestions, workflows, reminders and coordination actions
        """
        if not self._initialized:
            self.initialize()

        response = self.intelligent_assistant.process_request(context)

        self.logger.info(f"Intelligent assistant request processing complete: {len(response.suggestions)} suggestions, "
                        f"{len(response.workflows)} workflows, {len(response.reminders)} reminders")
        return response

    def update_assistant_config(self, config: AssistantConfig):
        """
        Update intelligent assistant configuration

        Args:
            config: New assistant configuration
        """
        if not self._initialized:
            self.initialize()

        self.intelligent_assistant.update_config(config)

        self.logger.info("Intelligent assistant configuration updated")

    def get_assistant_statistics(self) -> Dict[str, Any]:
        """
        Get intelligent assistant statistics

        Returns:
            Statistics dictionary, including performance metrics, configuration info and module statistics
        """
        if not self._initialized:
            self.initialize()

        stats = self.intelligent_assistant.get_statistics()

        self.logger.info("Intelligent assistant statistics retrieved")
        return stats

    def reset_assistant_statistics(self):
        """
        Reset intelligent assistant statistics
        """
        if not self._initialized:
            self.initialize()

        self.intelligent_assistant.reset_statistics()

        self.logger.info("Intelligent assistant statistics reset")

    def predict_importance_evolution(self, memory_id: int,
                                   time_horizon: timedelta = timedelta(days=30)) -> Optional[Prediction]:
        """
        Predict memory importance evolution

        Args:
            memory_id: Memory ID
            time_horizon: Prediction time range

        Returns:
            Importance evolution prediction
        """
        if not self._initialized:
            self.initialize()

        memory = self.get_memory(memory_id)
        if not memory:
            self.logger.warning(f"Memory {memory_id} does not exist")
            return None

        prediction = self.predictive_engine.predict_importance_evolution(memory, time_horizon)

        self.logger.info(f"Importance evolution prediction completed: memory {memory_id}")
        return prediction

    def get_prediction_statistics(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get prediction statistics

        Args:
            project_path: Project path filter

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        # Generate current predictions
        predictions = self.predict_next_needs(project_path)

        # Get statistics
        stats = self.predictive_engine.get_prediction_statistics(predictions)

        return stats

    def store_prediction(self, prediction: Prediction) -> bool:
        """
        Store prediction results to database

        Args:
            prediction: Prediction object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            import json

            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO prediction_patterns
                    (prediction_type, content, confidence, probability,
                     evidence, related_memories, time_horizon, prediction_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_type.value,
                    prediction.content,
                    prediction.confidence,
                    prediction.probability,
                    json.dumps(prediction.evidence),
                    json.dumps(prediction.related_memories),
                    prediction.time_horizon.total_seconds() if prediction.time_horizon else None,
                    json.dumps(prediction.to_dict())
                ))
                conn.commit()

            self.logger.info(f"Prediction result stored: {prediction.prediction_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store prediction result: {e}")
            return False

    def store_decision_pattern(self, pattern: DecisionPattern) -> bool:
        """
        Store decision pattern to database

        Args:
            pattern: Decision pattern object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            import json

            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO prediction_patterns
                    (pattern_id, pattern_name, trigger_conditions, typical_sequence,
                     success_rate, usage_count, last_used, prediction_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_name,
                    json.dumps(pattern.trigger_conditions),
                    json.dumps(pattern.typical_sequence),
                    pattern.success_rate,
                    pattern.usage_count,
                    pattern.last_used.isoformat(),
                    json.dumps(pattern.to_dict())
                ))
                conn.commit()

            self.logger.info(f"Decision pattern stored: {pattern.pattern_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store decision pattern: {e}")
            return False

    # ==================== Knowledge Graph Features ====================

    def build_knowledge_graph(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build knowledge graph

        Args:
            project_path: Project path filter

        Returns:
            Graph statistics
        """
        if not self._initialized:
            self.initialize()

        # Get memories to analyze
        if project_path:
            memories = self.get_project_memories(project_path, limit=500)
        else:
            memories = self.memory_repo.search_memories(limit=500)

        # Build knowledge graph
        graph = self.knowledge_graph.build_concept_graph(memories)

        # Get statistics
        stats = self.knowledge_graph.get_graph_statistics()

        self.logger.info(f"Knowledge graph construction completed: {stats['total_concepts']} concepts, {stats['total_relations']} relations")
        return stats

    def search_concepts(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search related concepts

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of concept search results
        """
        if not self._initialized:
            self.initialize()

        results = self.knowledge_graph.search_concepts(query, top_k)

        self.logger.info(f"Concept search '{query}' returned {len(results)} results")
        return results

    def get_concept_neighbors(self, concept_id: str) -> List[Tuple[str, float]]:
        """
        Get neighbor nodes of a concept

        Args:
            concept_id: Concept ID

        Returns:
            List of neighbor concepts
        """
        if not self._initialized:
            self.initialize()

        neighbors = self.knowledge_graph.get_concept_neighbors(concept_id)

        self.logger.info(f"Concept {concept_id} has {len(neighbors)} neighbors")
        return neighbors

    def find_concept_clusters(self) -> List[List[str]]:
        """
        Discover concept clusters

        Returns:
            List of concept clusters
        """
        if not self._initialized:
            self.initialize()

        clusters = self.knowledge_graph.find_concept_clusters()

        self.logger.info(f"Discovered {len(clusters)} concept clusters")
        return clusters

    def traverse_concept_path(self, start_concept: str, end_concept: str, max_depth: int = 3) -> List[List[str]]:
        """
        Traverse relationship paths between concepts

        Args:
            start_concept: Starting concept ID
            end_concept: Ending concept ID
            max_depth: Maximum search depth

        Returns:
            List of paths
        """
        if not self._initialized:
            self.initialize()

        paths = self.knowledge_graph.traverse_relation_path(start_concept, end_concept, max_depth)

        self.logger.info(f"Found {len(paths)} paths from {start_concept} to {end_concept}")
        return paths

    def get_central_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get concepts with highest centrality

        Args:
            top_k: Number of results to return

        Returns:
            List of central concepts
        """
        if not self._initialized:
            self.initialize()

        central_concepts = self.knowledge_graph.get_central_concepts(top_k)

        self.logger.info(f"Retrieved {len(central_concepts)} central concepts")
        return central_concepts

    def export_knowledge_graph(self, format: str = 'json') -> str:
        """
        Export knowledge graph

        Args:
            format: Export format ('json' or 'gexf')

        Returns:
            Export result (JSON string or file path)
        """
        if not self._initialized:
            self.initialize()

        result = self.knowledge_graph.export_graph(format)

        self.logger.info(f"Knowledge graph exported in {format} format")
        return result

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """
        Get concept by name

        Args:
            name: Concept name

        Returns:
            Concept object
        """
        if not self._initialized:
            self.initialize()

        concept = self.knowledge_graph.get_concept_by_name(name)

        if concept:
            self.logger.info(f"Found concept: {concept.name}")
        else:
            self.logger.info(f"Concept not found: {name}")

        return concept

    def store_concept(self, concept: Concept) -> bool:
        """
        Store concept to database

        Args:
            concept: Concept object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            import json

            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO concept_graph
                    (concept_id, name, category, frequency, importance,
                     related_memories, aliases, concept_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept.concept_id,
                    concept.name,
                    concept.category,
                    concept.frequency,
                    concept.importance,
                    json.dumps(concept.related_memories),
                    json.dumps(concept.aliases),
                    json.dumps(concept.to_dict())
                ))
                conn.commit()

            self.logger.info(f"Concept stored: {concept.concept_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store concept: {e}")
            return False

    def store_concept_relation(self, relation: ConceptRelation) -> bool:
        """
        Store concept relation to database

        Args:
            relation: Concept relation object

        Returns:
            Whether storage was successful
        """
        if not self._initialized:
            self.initialize()

        try:
            import json

            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO concept_graph
                    (relation_id, concept1_id, concept2_id, relation_type,
                     weight, evidence_count, confidence, relation_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.relation_id,
                    relation.concept1_id,
                    relation.concept2_id,
                    relation.relation_type.value,
                    relation.weight,
                    relation.evidence_count,
                    relation.confidence,
                    json.dumps(relation.to_dict())
                ))
                conn.commit()

            self.logger.info(f"Concept relation stored: {relation.relation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store concept relation: {e}")
            return False

    # ==================== Security Monitoring Management Interface ====================

    def get_security_status(self) -> Dict[str, Any]:
        """
        Get system security status

        Returns:
            Security status information dictionary, including encryption, access control, monitoring, backup and other module statuses
        """
        if not self._initialized:
            self.initialize()

        status = self.security_monitoring_manager.get_security_status()

        self.logger.info("Security status information retrieved")
        return status

    def apply_security_policy(self, policy_id: str) -> bool:
        """
        Apply security policy

        Args:
            policy_id: Security policy ID

        Returns:
            Whether application was successful
        """
        if not self._initialized:
            self.initialize()

        success = self.security_monitoring_manager.apply_security_policy(policy_id)

        self.logger.info(f"Security policy application {'successful' if success else 'failed'}: {policy_id}")
        return success

    def get_security_events(self, event_type: Optional[str] = None,
                          resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get security events

        Args:
            event_type: Event type filter
            resolved: Resolved status filter

        Returns:
            Security events list
        """
        if not self._initialized:
            self.initialize()

        events = self.security_monitoring_manager.get_security_events(event_type, resolved)

        # Convert to dictionary format
        event_dicts = []
        for event in events:
            event_dicts.append({
                'event_id': event.event_id,
                'event_type': event.event_type,
                'severity': event.severity.value,
                'source_module': event.source_module,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description,
                'details': event.details,
                'resolved': event.resolved,
                'resolution_notes': event.resolution_notes
            })

        self.logger.info(f"Retrieved {len(event_dicts)} security events")
        return event_dicts

    def resolve_security_event(self, event_id: str, resolution_notes: str) -> bool:
        """
        Resolve security event

        Args:
            event_id: Event ID
            resolution_notes: Resolution notes

        Returns:
            Whether resolution was successful
        """
        if not self._initialized:
            self.initialize()

        success = self.security_monitoring_manager.resolve_security_event(event_id, resolution_notes)

        self.logger.info(f"Security event resolution {'successful' if success else 'failed'}: {event_id}")
        return success

    def create_backup(self, backup_type: str = "incremental", description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create backup

        Args:
            backup_type: Backup type (full/incremental/differential)
            description: Backup description

        Returns:
            Backup information dictionary
        """
        if not self._initialized:
            self.initialize()

        backup_manager = self.security_monitoring_manager.backup_manager

        if backup_type == "full":
            backup_info = backup_manager.create_full_backup(description)
        elif backup_type == "incremental":
            backup_info = backup_manager.create_incremental_backup(description=description)
        elif backup_type == "differential":
            backup_info = backup_manager.create_differential_backup(description=description)
        else:
            self.logger.error(f"Unsupported backup type: {backup_type}")
            return None

        if backup_info:
            result = {
                'backup_id': backup_info.backup_id,
                'backup_type': backup_info.backup_type,
                'created_at': backup_info.created_at.isoformat(),
                'file_path': str(backup_info.file_path),
                'size_bytes': backup_info.size_bytes,
                'compressed': backup_info.compressed,
                'encrypted': backup_info.encrypted,
                'status': backup_info.status.value if hasattr(backup_info.status, 'value') else str(backup_info.status)
            }
            self.logger.info(f"Backup created successfully: {backup_info.backup_id}")
            return result

        return None

    def restore_backup(self, backup_id: str, target_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Restore backup

        Args:
            backup_id: Backup ID
            target_path: Target path

        Returns:
            Restore information dictionary
        """
        if not self._initialized:
            self.initialize()

        backup_manager = self.security_monitoring_manager.backup_manager
        target_path_obj = Path(target_path) if target_path else None

        restore_info = backup_manager.restore_backup(backup_id, target_path_obj)

        if restore_info:
            result = {
                'restore_id': restore_info.restore_id,
                'backup_id': restore_info.backup_id,
                'restore_type': restore_info.restore_type,
                'target_path': str(restore_info.target_path),
                'created_at': restore_info.created_at.isoformat(),
                'status': restore_info.status,
                'progress_percentage': restore_info.progress_percentage
            }
            self.logger.info(f"Backup restore started: {restore_info.restore_id}")
            return result

        return None

    def get_backup_statistics(self) -> Dict[str, Any]:
        """
        Get backup statistics

        Returns:
            Backup statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        backup_manager = self.security_monitoring_manager.backup_manager
        stats = backup_manager.get_enhanced_backup_statistics()

        self.logger.info("Backup statistics retrieved")
        return stats

    def register_sync_device(self, device_name: str, device_type: str, sync_path: Optional[str] = None) -> str:
        """
        Register sync device

        Args:
            device_name: Device name
            device_type: Device type
            sync_path: Sync path

        Returns:
            Device ID
        """
        if not self._initialized:
            self.initialize()

        migration_manager = self.security_monitoring_manager.migration_manager
        sync_path_obj = Path(sync_path) if sync_path else None

        device_id = migration_manager.register_sync_device(device_name, device_type, sync_path_obj)

        self.logger.info(f"Sync device registration {'successful' if device_id else 'failed'}: {device_name}")
        return device_id

    def start_device_migration(self, source_device_id: str, target_device_id: str) -> Optional[Dict[str, Any]]:
        """
        Start device migration

        Args:
            source_device_id: Source device ID
            target_device_id: Target device ID

        Returns:
            Migration information dictionary
        """
        if not self._initialized:
            self.initialize()

        migration_manager = self.security_monitoring_manager.migration_manager
        migration_info = migration_manager.start_cross_device_sync(source_device_id, target_device_id)

        if migration_info:
            result = {
                'migration_id': migration_info.migration_id,
                'source_device': migration_info.source_device,
                'target_device': migration_info.target_device,
                'migration_type': migration_info.migration_type,
                'created_at': migration_info.created_at.isoformat(),
                'status': migration_info.status,
                'progress_percentage': migration_info.progress_percentage
            }
            self.logger.info(f"Device migration started: {migration_info.migration_id}")
            return result

        return None

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health status

        Returns:
            System health status dictionary
        """
        if not self._initialized:
            self.initialize()

        health_checker = self.security_monitoring_manager.health_checker
        health_status = health_checker.get_system_health()

        self.logger.info("System health status retrieved")
        return health_status

    def perform_security_scan(self) -> Dict[str, Any]:
        """
        Perform security scan

        Returns:
            Security scan results dictionary
        """
        if not self._initialized:
            self.initialize()

        # Comprehensive security checks across modules
        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'encryption_status': self.security_monitoring_manager.encryption_manager.get_encryption_statistics(),
            'access_control_status': self.security_monitoring_manager.access_controller.get_security_statistics(),
            'backup_health': {
                'health_score': self.security_monitoring_manager.backup_manager._calculate_backup_health_score(),
                'statistics': self.security_monitoring_manager.backup_manager.get_enhanced_backup_statistics()
            },
            'system_health': self.security_monitoring_manager.health_checker.get_system_health(),
            'security_events': len([e for e in self.security_monitoring_manager.security_events if not e.resolved]),
            'recommendations': []
        }

        # Generate security recommendations
        if scan_results['backup_health']['health_score'] < 70:
            scan_results['recommendations'].append("Backup health score is low, recommend checking backup strategy")

        if scan_results['system_health'].get('overall_health', 0) < 0.8:
            scan_results['recommendations'].append("System health needs attention, recommend performing system maintenance")

        if scan_results['security_events'] > 5:
            scan_results['recommendations'].append("Multiple unresolved security events exist, recommend timely resolution")

        self.logger.info("Security scan completed")
        return scan_results

    # ===== Intelligent Guidance System Interface =====

    def create_user_session(self, session_id: str, user_level: str = "beginner",
                           project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create user session

        Args:
            session_id: Session ID
            user_level: User level
            project_path: Project path

        Returns:
            Session information dictionary
        """
        if not self._initialized:
            self.initialize()

        # Convert user level
        level_mapping = {
            "beginner": UserLevel.BEGINNER,
            "intermediate": UserLevel.INTERMEDIATE,
            "advanced": UserLevel.ADVANCED,
            "expert": UserLevel.EXPERT
        }

        user_level_enum = level_mapping.get(user_level, UserLevel.BEGINNER)

        session = self.guidance_manager.create_user_session(
            session_id, user_level_enum, project_path
        )

        self.logger.info(f"User session created: {session_id}")
        return {
            "session_id": session.session_id,
            "user_level": session.user_level.value,
            "project_path": session.project_path,
            "created_at": session.last_activity.isoformat()
        }

    def get_tool_recommendations(self, session_id: str, current_tool: Optional[str] = None,
                               project_type: Optional[str] = None,
                               max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get tool recommendations

        Args:
            session_id: Session ID
            current_tool: Currently used tool
            project_type: Project type
            max_recommendations: Maximum recommendation count

        Returns:
            Recommendation list
        """
        if not self._initialized:
            self.initialize()

        # Get user context
        session = self.guidance_manager.active_sessions.get(session_id)
        if not session:
            return []

        user_context = {
            "session_id": session_id,
            "user_level": session.user_level.value,
            "current_tool": current_tool,
            "project_type": project_type,
            "recent_tools": session.recent_tools,
            "error_history": session.error_history
        }

        recommendations = self.tool_recommender.get_recommendations(
            user_context, max_recommendations
        )

        # Convert to dictionary format
        result = []
        for rec in recommendations:
            result.append({
                "tool_name": rec.tool_name,
                "confidence_score": rec.confidence_score,
                "reason": rec.reason.value,
                "explanation": rec.explanation,
                "usage_hint": rec.usage_hint,
                "prerequisites": rec.prerequisites,
                "estimated_value": rec.estimated_value
            })

        self.logger.info(f"Tool recommendations generated: {session_id}, count: {len(result)}")
        return result

    def get_contextual_hints(self, session_id: str, current_tool: Optional[str] = None,
                           error_occurred: bool = False,
                           error_message: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get contextual usage hints

        Args:
            session_id: Session ID
            current_tool: Current tool
            error_occurred: Whether an error occurred
            error_message: Error message

        Returns:
            Hint list
        """
        if not self._initialized:
            self.initialize()

        session = self.guidance_manager.active_sessions.get(session_id)
        if not session:
            return []

        context = {
            "current_tool": current_tool,
            "user_level": session.user_level.value,
            "recent_tools": session.recent_tools,
            "error_occurred": error_occurred,
            "error_message": error_message
        }

        hints = self.usage_hint_provider.get_contextual_hints(session_id, context)

        # Convert to dictionary format
        result = []
        for hint in hints:
            result.append({
                "hint_id": hint.hint_id,
                "hint_type": hint.hint_type.value,
                "trigger": hint.trigger.value,
                "tool_name": hint.tool_name,
                "title": hint.title,
                "message": hint.message,
                "example": hint.example,
                "priority": hint.priority
            })

        self.logger.info(f"Usage hints generated: {session_id}, count: {len(result)}")
        return result

    def get_learning_path_recommendations(self, session_id: str) -> Dict[str, Any]:
        """
        Get learning path recommendations

        Args:
            session_id: Session ID

        Returns:
            Learning path recommendation information
        """
        if not self._initialized:
            self.initialize()

        session = self.guidance_manager.active_sessions.get(session_id)
        if not session:
            return {}

        # Get available paths
        available_paths = self.learning_path_manager.get_available_paths(session.user_level.value)

        # Get next recommendation
        next_recommendation = self.learning_path_manager.get_next_recommendation(session_id)

        # Get user progress
        user_progress = self.learning_path_manager.get_user_progress(session_id)

        result = {
            "available_paths": [
                {
                    "path_id": path.path_id,
                    "name": path.name,
                    "description": path.description,
                    "difficulty": path.difficulty.value,
                    "total_steps": path.total_steps,
                    "estimated_time_minutes": path.estimated_total_time_minutes,
                    "target_audience": path.target_audience
                }
                for path in available_paths
            ],
            "next_recommendation": next_recommendation,
            "current_progress": user_progress
        }

        self.logger.info(f"Learning path recommendations generated: {session_id}")
        return result

    def start_learning_path(self, session_id: str, path_id: str) -> bool:
        """
        Start learning path

        Args:
            session_id: Session ID
            path_id: Path ID

        Returns:
            Whether successfully started
        """
        if not self._initialized:
            self.initialize()

        success = self.learning_path_manager.start_learning_path(session_id, path_id)

        if success:
            self.logger.info(f"Learning path started: {session_id}, {path_id}")
        else:
            self.logger.warning(f"Learning path start failed: {session_id}, {path_id}")

        return success

    def complete_learning_step(self, session_id: str, path_id: str, step_id: str) -> bool:
        """
        Complete learning step

        Args:
            session_id: Session ID
            path_id: Path ID
            step_id: Step ID

        Returns:
            Whether successfully completed
        """
        if not self._initialized:
            self.initialize()

        success = self.learning_path_manager.complete_step(session_id, path_id, step_id)

        if success:
            self.logger.info(f"Learning step completed: {session_id}, {path_id}/{step_id}")
        else:
            self.logger.warning(f"Learning step completion failed: {session_id}, {path_id}/{step_id}")

        return success

    def update_user_activity(self, session_id: str, tool_used: str, success: bool = True,
                           params_used: Optional[Dict[str, Any]] = None):
        """
        Update user activity

        Args:
            session_id: Session ID
            tool_used: Tool used
            success: Whether successful
            params_used: Parameters used
        """
        if not self._initialized:
            self.initialize()

        # Update guidance manager
        self.guidance_manager.update_user_activity(session_id, tool_used, success)

        # Update tool recommender
        self.tool_recommender.update_usage_patterns(session_id, tool_used, success)

        # Update usage hint provider
        if params_used:
            self.usage_hint_provider.update_tool_usage_stats(
                session_id, tool_used, params_used, success
            )

        self.logger.debug(f"User activity updated: {session_id}, {tool_used}")

    def get_guidance_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get guidance system statistics

        Args:
            session_id: Session ID, if None returns global statistics

        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.initialize()

        result = {
            "guidance_manager": self.guidance_manager.get_usage_statistics(session_id),
            "tool_recommender": self.tool_recommender.get_recommendation_statistics(),
            "usage_hints": self.usage_hint_provider.get_usage_statistics(session_id),
            "learning_paths": self.learning_path_manager.get_learning_statistics()
        }

        self.logger.info(f"Guidance system statistics retrieved: {session_id or 'global'}")
        return result

    def cleanup_guidance_data(self, max_age_days: int = 7):
        """
        Clean up old data from guidance system

        Args:
            max_age_days: Maximum retention days
        """
        if not self._initialized:
            self.initialize()

        self.guidance_manager.cleanup_inactive_sessions(max_age_days * 24)  # Convert to hours
        self.usage_hint_provider.cleanup_old_data(max_age_days)
        self.learning_path_manager.cleanup_inactive_users(max_age_days)

        self.logger.info(f"Guidance system data cleanup completed, retained data within {max_age_days} days")

    def stop_security_monitoring(self):
        """
        Stop security monitoring service
        """
        if not self._initialized:
            return

        self.security_monitoring_manager.stop_coordination_service()
        self.logger.info("Security monitoring service stopped")

    def scan_and_remember_project(self, project_path: str, quick_scan: bool = True) -> Dict[str, Any]:
        """
        Scan project and store to memory system

        Args:
            project_path: Project path
            quick_scan: Quick scan mode

        Returns:
            Project snapshot and memory ID
        """
        if not self._initialized:
            self.initialize()

        self.logger.info(f"Starting project scan: {project_path}")

        # 1. Scan project
        snapshot = self.project_scanner.scan_project(project_path, quick_scan)

        # 2. Generate memory content
        summary = self.project_scanner.generate_summary(snapshot)

        # 3. Store to memory system (including detailed metadata)
        memory_id = self.store_memory(
            content=f"Project snapshot - {snapshot.project_name}\n\n{summary}",
            importance=0.8,
            memory_type="project",
            category="project_snapshot",
            project_path=project_path
        )

        # Supplement metadata (store_memory creates basic metadata, we need to update it)
        if memory_id:
            try:
                memory = self.memory_repo.get_memory(memory_id)
                if memory and memory.metadata:
                    memory.metadata.update({
                        'project_name': snapshot.project_name,
                        'project_type': snapshot.project_type,
                        'tech_stack': snapshot.tech_stack,
                        'frameworks': snapshot.frameworks,
                        'languages': snapshot.languages,
                        'statistics': snapshot.statistics,
                        'scan_time': snapshot.scan_time.isoformat()
                    })
                    self.memory_repo.update_memory(memory)
            except Exception as e:
                self.logger.warning(f"Failed to update snapshot metadata: {e}")

        # 4. Update project context
        self._update_project_context_from_snapshot(project_path, snapshot)

        self.logger.info(f"Project scan completed: {snapshot.project_name}, memory ID: {memory_id}")

        return {
            'snapshot': snapshot,
            'memory_id': memory_id,
            'summary': summary
        }

    def _update_project_context_from_snapshot(self, project_path: str, snapshot):
        """Update project context from project snapshot"""
        # Check if project already exists
        project = self.project_repo.get_project_by_path(project_path)
        if not project:
            # Create new project
            try:
                project = Project(
                    name=snapshot.project_name,
                    path=project_path,
                    tech_stack=', '.join(snapshot.tech_stack),
                    description=snapshot.project_type
                )
                self.project_repo.create_project(project)
                self.logger.info(f"Project record created: {snapshot.project_name}")
            except Exception as e:
                self.logger.warning(f"Failed to create project record: {e}")

    def get_project_progress(self, project_path: str) -> Dict[str, Any]:
        """
        Get project progress and status

        Args:
            project_path: Project path

        Returns:
            Project progress report
        """
        if not self._initialized:
            self.initialize()

        self.logger.info(f"Getting project progress: {project_path}")

        # 1. Check if project snapshot exists and is fresh (get all snapshots, sort by time and take latest)
        project_memories = self.memory_repo.search_memories(
            query="",
            project_path=project_path,
            memory_type="project",
            category="project_snapshot",
            limit=100  # Get enough, then manually sort to find latest
        )

        # Sort by creation time in descending order, get latest snapshot
        if project_memories:
            project_memories.sort(key=lambda m: m.created_at if m.created_at else datetime.min, reverse=True)

        should_rescan = False
        snapshot_data = None
        last_scan_time = None

        if not project_memories:
            should_rescan = True
            self.logger.info("Project snapshot does not exist, need to scan")
        else:
            # Use latest snapshot to check if expired (>7 days)
            latest_snapshot = project_memories[0]
            last_scan_str = latest_snapshot.created_at.isoformat() if latest_snapshot.created_at else None
            if last_scan_str:
                last_scan_time = datetime.fromisoformat(last_scan_str)
                days_old = (datetime.now() - last_scan_time).days
                if days_old > 7:
                    should_rescan = True
                    self.logger.info(f"Project snapshot expired ({days_old} days), need to rescan")
                else:
                    self.logger.info(f"Using existing project snapshot (created {days_old} days ago)")

        # 2. If needed, rescan
        if should_rescan:
            scan_result = self.scan_and_remember_project(project_path, quick_scan=True)
            snapshot = scan_result['snapshot']
        else:
            # Restore snapshot info from memory (using latest snapshot)
            latest_snapshot = project_memories[0]
            snapshot_data = {
                'project_name': latest_snapshot.content.split('\n')[0].replace('Project snapshot - ', ''),
                'scan_time': last_scan_str,
                'project_type': latest_snapshot.metadata.get('project_type') if latest_snapshot.metadata else None,
                'tech_stack': latest_snapshot.metadata.get('tech_stack', []) if latest_snapshot.metadata else [],
                'languages': latest_snapshot.metadata.get('languages', []) if latest_snapshot.metadata else [],
                'statistics': latest_snapshot.metadata.get('statistics', {}) if latest_snapshot.metadata else {}
            }

        # 3. Get project-related memories (exclude project snapshots, only get work records)
        recent_work = self.memory_repo.search_memories(
            query="",
            project_path=project_path,
            limit=10
        )
        # Filter out project snapshots, only keep other types of memories
        recent_work = [m for m in recent_work if not (m.memory_type == "project" and m.category == "project_snapshot")]

        # 4. Get active task
        active_task = None
        try:
            active_task = self.active_task_manager.get_active_task(project_path)
        except Exception as e:
            self.logger.warning(f"Failed to get active task: {e}")

        # 5. Generate progress report
        if should_rescan:
            # Just scanned, use snapshot object
            progress_report = {
                "success": True,
                "project_path": project_path,
                "project_name": snapshot.project_name,
                "project_type": snapshot.project_type,
                "tech_stack": snapshot.tech_stack,
                "languages": snapshot.languages,
                "statistics": snapshot.statistics,
                "active_task": active_task,
                "recent_work": [
                    {
                        "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                        "type": m.memory_type,
                        "category": m.category,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    }
                    for m in recent_work[:5]
                ],
                "last_scan": snapshot.scan_time.isoformat(),
                "auto_scanned": True
            }
        else:
            # Use cached snapshot data
            progress_report = {
                "success": True,
                "project_path": project_path,
                "project_name": snapshot_data.get('project_name', 'Unknown'),
                "project_type": snapshot_data.get('project_type'),
                "tech_stack": snapshot_data.get('tech_stack', []),
                "languages": snapshot_data.get('languages', []),
                "statistics": snapshot_data.get('statistics', {}),
                "active_task": active_task,
                "recent_work": [
                    {
                        "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                        "type": m.memory_type,
                        "category": m.category,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    }
                    for m in recent_work[:5]
                ],
                "last_scan": snapshot_data.get('scan_time'),
                "auto_scanned": False
            }

        self.logger.info(f"Project progress report generation completed: {project_path}")
        return progress_report

    def update_project_status(self, project_path: str, status: Optional[str] = None,
                              current_task: Optional[str] = None,
                              progress_percentage: Optional[int] = None,
                              notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Update project status and progress

        Args:
            project_path: Project path
            status: Project phase
            current_task: Current task description
            progress_percentage: Progress percentage
            notes: Notes information

        Returns:
            Update result
        """
        if not self._initialized:
            self.initialize()

        self.logger.info(f"Updating project status: {project_path}")

        # Store status update
        content_parts = ["Project status update"]

        if status:
            content_parts.append(f"Phase: {status}")
        if current_task:
            content_parts.append(f"Current task: {current_task}")
        if progress_percentage is not None:
            content_parts.append(f"Progress: {progress_percentage}%")
        if notes:
            content_parts.append(f"Notes: {notes}")

        content = "\n".join(content_parts)

        memory_id = self.store_memory(
            content=content,
            importance=0.7,
            memory_type="project",
            category="status_update",
            project_path=project_path
        )

        # Update active task
        if current_task:
            try:
                self.active_task_manager.set_active_task(project_path, current_task)
            except Exception as e:
                self.logger.warning(f"Failed to set active task: {e}")

        self.logger.info(f"Project status updated: {project_path}, memory ID: {memory_id}")

        return {
            "success": True,
            "memory_id": memory_id,
            "message": "Project status updated",
            "content": content
        }

    def shutdown(self):
        """Shutdown memory engine (cleanup resources)"""
        try:
            # Stop cleanup scheduler
            if hasattr(self, 'cleanup_scheduler') and self.cleanup_scheduler:
                self.cleanup_scheduler.stop()
                self.logger.info("Cleanup scheduler stopped")

            # Stop event bus
            if self._event_bus:
                self._event_bus.stop()
                self.logger.info("Event bus stopped")

            # Stop database watcher
            if self._db_watcher:
                self._db_watcher.stop()
                self.logger.info("Database watcher stopped")

            self.logger.info("Memory engine shutdown")
        except Exception as e:
            self.logger.error(f"Failed to shutdown memory engine: {e}")

    def __del__(self):
        """Destructor - ensure resource cleanup"""
        self.shutdown()