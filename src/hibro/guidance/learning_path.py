#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Path Manager
Provides progressive learning paths for users at different levels
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import Config


class LearningStage(Enum):
    """Learning stage"""
    NOT_STARTED = "not_started"      # Not started
    IN_PROGRESS = "in_progress"      # In progress
    COMPLETED = "completed"          # Completed
    SKIPPED = "skipped"             # Skipped


class PathDifficulty(Enum):
    """Path difficulty level"""
    BEGINNER = "beginner"           # Beginner
    INTERMEDIATE = "intermediate"   # Intermediate
    ADVANCED = "advanced"          # Advanced
    EXPERT = "expert"              # Expert


@dataclass
class LearningStep:
    """Learning step"""
    step_id: str
    title: str
    description: str
    tools_to_learn: List[str]
    prerequisites: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 15
    practice_tasks: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """Learning path"""
    path_id: str
    name: str
    description: str
    difficulty: PathDifficulty
    total_steps: int
    estimated_total_time_minutes: int
    steps: List[LearningStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    target_audience: str = ""


@dataclass
class UserProgress:
    """User learning progress"""
    user_id: str
    path_id: str
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    total_time_spent_minutes: int = 0
    step_completion_times: Dict[str, datetime] = field(default_factory=dict)


class LearningPathManager:
    """Learning path manager"""

    def __init__(self, config: Config):
        """
        Initialize learning path manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.learning_path_manager')

        # Learning path definitions
        self.learning_paths = self._define_learning_paths()

        # User progress tracking
        self.user_progress: Dict[str, Dict[str, UserProgress]] = {}  # user_id -> path_id -> progress

        # Learning statistics
        self.completion_stats: Dict[str, Dict[str, int]] = {}  # path_id -> step_id -> completion_count

        self.logger.info("Learning path manager initialized")

    def _define_learning_paths(self) -> Dict[str, LearningPath]:
        """Define learning paths"""
        paths = {}

        # Beginner path
        beginner_steps = [
            LearningStep(
                step_id="beginner_step_1",
                title="Get Personal Context",
                description="Learn how to get and manage personal preference settings",
                tools_to_learn=["get_quick_context", "get_preferences"],
                estimated_time_minutes=10,
                practice_tasks=[
                    "Call get_quick_context to view current context information",
                    "Use get_preferences to get programming preference settings",
                    "Understand the structure and meaning of returned information"
                ],
                success_criteria=[
                    "Able to successfully call get_quick_context",
                    "Understand returned preference information",
                    "Know when to call these tools"
                ],
                tips=[
                    "Recommend calling get_quick_context at the start of each new session",
                    "Call get_preferences before programming tasks to ensure consistent code style",
                    "These two tools are the foundation of all personalized features"
                ]
            ),
            LearningStep(
                step_id="beginner_step_2",
                title="Information Storage and Retrieval",
                description="Master basic information storage and search functions",
                tools_to_learn=["remember", "search_memories"],
                prerequisites=["beginner_step_1"],
                estimated_time_minutes=15,
                practice_tasks=[
                    "Use remember to store some personal preferences or important information",
                    "Use search_memories to search for the information just stored",
                    "Try different search keywords"
                ],
                success_criteria=[
                    "Able to store and retrieve information",
                    "Understand different types of memory categories",
                    "Master basic search techniques"
                ],
                tips=[
                    "Choose appropriate importance level when storing information",
                    "Use descriptive keywords for easier future searching",
                    "Regularly organize and update stored information"
                ]
            ),
            LearningStep(
                step_id="beginner_step_3",
                title="Introduction to Intelligent Q&A",
                description="Learn to use intelligent analysis tools to solve problems",
                tools_to_learn=["answer_specific_question", "get_smart_suggestions"],
                prerequisites=["beginner_step_2"],
                estimated_time_minutes=20,
                practice_tasks=[
                    "Ask a specific technical question and use answer_specific_question to get an answer",
                    "Use get_smart_suggestions to get project improvement suggestions",
                    "Compare the different uses of the two tools"
                ],
                success_criteria=[
                    "Able to ask clear questions",
                    "Understand the differences and applicable scenarios of the two tools",
                    "Get useful analysis results"
                ],
                tips=[
                    "More detailed problem descriptions lead to more accurate answers",
                    "Provide relevant context information",
                    "answer_specific_question is for specific questions, get_smart_suggestions is for getting suggestions"
                ]
            )
        ]

        paths["beginner_path"] = LearningPath(
            path_id="beginner_path",
            name="Beginner Path",
            description="Suitable for first-time hibro users, starting from basic features and gradually mastering core capabilities",
            difficulty=PathDifficulty.BEGINNER,
            total_steps=len(beginner_steps),
            estimated_total_time_minutes=45,
            steps=beginner_steps,
            target_audience="Developers using hibro for the first time"
        )

        # Intermediate path
        intermediate_steps = [
            LearningStep(
                step_id="intermediate_step_1",
                title="Semantic Search and Knowledge Graph",
                description="Master advanced search and knowledge association features",
                tools_to_learn=["search_semantic", "build_knowledge_graph"],
                prerequisites=["beginner_path"],
                estimated_time_minutes=25,
                practice_tasks=[
                    "Use search_semantic for conceptual searching",
                    "Build a project-related knowledge graph",
                    "Compare the differences between keyword search and semantic search"
                ],
                success_criteria=[
                    "Understand the advantages of semantic search",
                    "Able to build and analyze knowledge graphs",
                    "Master concept association analysis"
                ],
                tips=[
                    "Semantic search is suitable for finding related concepts and ideas",
                    "Knowledge graphs help understand relationships between concepts",
                    "Combined use provides a more comprehensive information view"
                ]
            ),
            LearningStep(
                step_id="intermediate_step_2",
                title="Deep Project Analysis",
                description="Learn to use advanced analysis tools for project insights",
                tools_to_learn=["analyze_project_deeply", "analyze_causal_relations"],
                prerequisites=["intermediate_step_1"],
                estimated_time_minutes=30,
                practice_tasks=[
                    "Perform deep analysis on the current project",
                    "Analyze causal relationships in the project",
                    "Generate project improvement suggestions"
                ],
                success_criteria=[
                    "Able to perform comprehensive project analysis",
                    "Understand the value of causal analysis",
                    "Get actionable improvement suggestions"
                ],
                tips=[
                    "Deep analysis is suitable for use before important decisions",
                    "Causal analysis helps understand the root causes of problems",
                    "Combine with project context for more accurate analysis"
                ]
            ),
            LearningStep(
                step_id="intermediate_step_3",
                title="Workflow Optimization",
                description="Identify and optimize repetitive workflows",
                tools_to_learn=["detect_workflow_patterns", "get_workflow_recommendations"],
                prerequisites=["intermediate_step_2"],
                estimated_time_minutes=25,
                practice_tasks=[
                    "Detect repetitive patterns in current work",
                    "Get workflow optimization recommendations",
                    "Try to implement an automation suggestion"
                ],
                success_criteria=[
                    "Able to identify repetitive work patterns",
                    "Understand the value and methods of automation",
                    "Implement at least one optimization suggestion"
                ],
                tips=[
                    "Regularly detect workflow patterns",
                    "Start automation from simple repetitive tasks",
                    "Gradually build efficient workflows"
                ]
            )
        ]

        paths["intermediate_path"] = LearningPath(
            path_id="intermediate_path",
            name="Intermediate Path",
            description="Suitable for users who have mastered basic features, learning advanced analysis and optimization capabilities",
            difficulty=PathDifficulty.INTERMEDIATE,
            total_steps=len(intermediate_steps),
            estimated_total_time_minutes=80,
            steps=intermediate_steps,
            prerequisites=["beginner_path"],
            target_audience="Developers with some experience"
        )

        # Advanced expert path
        advanced_steps = [
            LearningStep(
                step_id="advanced_step_1",
                title="Comprehensive Intelligent Assistant",
                description="Master the highest level comprehensive intelligent assistance features",
                tools_to_learn=["get_comprehensive_assistance", "process_intelligent_assistant"],
                prerequisites=["intermediate_path"],
                estimated_time_minutes=35,
                practice_tasks=[
                    "Use comprehensive assistant to solve complex project problems",
                    "Experience multi-module coordinated intelligent suggestions",
                    "Compare applicable scenarios of different assistant tools"
                ],
                success_criteria=[
                    "Able to effectively use comprehensive intelligent assistant",
                    "Understand the value of multi-module coordination",
                    "Master methods for solving complex problems"
                ],
                tips=[
                    "Comprehensive assistant is suitable for complex multi-dimensional problems",
                    "Provide detailed problem descriptions and context",
                    "Fully utilize coordination features for best recommendations"
                ]
            ),
            LearningStep(
                step_id="advanced_step_2",
                title="Adaptive Learning System",
                description="Use the system's adaptive learning capabilities to improve efficiency",
                tools_to_learn=["track_user_behavior", "get_personalized_recommendations", "adaptive_importance_scoring"],
                prerequisites=["advanced_step_1"],
                estimated_time_minutes=30,
                practice_tasks=[
                    "Enable user behavior tracking",
                    "Get personalized recommendations",
                    "Experience adaptive importance scoring"
                ],
                success_criteria=[
                    "Understand the principles of adaptive learning",
                    "Able to utilize personalized recommendations",
                    "Master system optimization techniques"
                ],
                tips=[
                    "Adaptive learning requires some usage history",
                    "Regularly check personalized recommendations",
                    "Feedback on usage experience helps system improve"
                ]
            ),
            LearningStep(
                step_id="advanced_step_3",
                title="System Security and Monitoring",
                description="Master enterprise-level security monitoring features",
                tools_to_learn=["check_security_status", "apply_security_policy", "create_backup"],
                prerequisites=["advanced_step_2"],
                estimated_time_minutes=25,
                practice_tasks=[
                    "Check system security status",
                    "Apply appropriate security policies",
                    "Create system backup"
                ],
                success_criteria=[
                    "Able to monitor system security status",
                    "Understand applicable scenarios for different security policies",
                    "Master data protection best practices"
                ],
                tips=[
                    "Regularly check security status",
                    "Create backup before important changes",
                    "Choose appropriate security policy based on environment"
                ]
            )
        ]

        paths["advanced_path"] = LearningPath(
            path_id="advanced_path",
            name="Advanced Expert Path",
            description="Suitable for advanced users, mastering all advanced features and enterprise-level capabilities",
            difficulty=PathDifficulty.ADVANCED,
            total_steps=len(advanced_steps),
            estimated_total_time_minutes=90,
            steps=advanced_steps,
            prerequisites=["intermediate_path"],
            target_audience="Advanced developers and system administrators"
        )

        return paths

    def get_available_paths(self, user_level: str = "beginner") -> List[LearningPath]:
        """
        Get available learning paths

        Args:
            user_level: User level

        Returns:
            List[LearningPath]: List of suitable learning paths
        """
        available_paths = []

        for path in self.learning_paths.values():
            if user_level == "beginner" and path.difficulty == PathDifficulty.BEGINNER:
                available_paths.append(path)
            elif user_level == "intermediate" and path.difficulty in [PathDifficulty.BEGINNER, PathDifficulty.INTERMEDIATE]:
                available_paths.append(path)
            elif user_level in ["advanced", "expert"]:
                available_paths.append(path)

        return sorted(available_paths, key=lambda x: x.difficulty.value)

    def start_learning_path(self, user_id: str, path_id: str) -> bool:
        """
        Start learning path

        Args:
            user_id: User ID
            path_id: Path ID

        Returns:
            bool: Whether successfully started
        """
        if path_id not in self.learning_paths:
            self.logger.error(f"Learning path does not exist: {path_id}")
            return False

        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        # Check prerequisites
        path = self.learning_paths[path_id]
        if path.prerequisites:
            for prereq in path.prerequisites:
                if prereq not in self.user_progress[user_id] or self.user_progress[user_id][prereq].current_step < len(self.learning_paths[prereq].steps):
                    self.logger.warning(f"User {user_id} does not meet prerequisites for path {path_id}: {prereq}")
                    return False

        # Create progress record
        self.user_progress[user_id][path_id] = UserProgress(
            user_id=user_id,
            path_id=path_id
        )

        self.logger.info(f"User {user_id} started learning path: {path_id}")
        return True

    def get_current_step(self, user_id: str, path_id: str) -> Optional[LearningStep]:
        """
        Get current learning step

        Args:
            user_id: User ID
            path_id: Path ID

        Returns:
            Optional[LearningStep]: Current step, or None if not available
        """
        if (user_id not in self.user_progress or
            path_id not in self.user_progress[user_id]):
            return None

        progress = self.user_progress[user_id][path_id]
        path = self.learning_paths[path_id]

        if progress.current_step >= len(path.steps):
            return None  # All steps completed

        return path.steps[progress.current_step]

    def complete_step(self, user_id: str, path_id: str, step_id: str) -> bool:
        """
        Complete learning step

        Args:
            user_id: User ID
            path_id: Path ID
            step_id: Step ID

        Returns:
            bool: Whether successfully completed
        """
        if (user_id not in self.user_progress or
            path_id not in self.user_progress[user_id]):
            return False

        progress = self.user_progress[user_id][path_id]
        path = self.learning_paths[path_id]

        # Verify step ID
        if progress.current_step >= len(path.steps):
            return False

        current_step = path.steps[progress.current_step]
        if current_step.step_id != step_id:
            return False

        # Mark as completed
        progress.completed_steps.append(step_id)
        progress.step_completion_times[step_id] = datetime.now()
        progress.current_step += 1
        progress.last_activity = datetime.now()

        # Update statistics
        if path_id not in self.completion_stats:
            self.completion_stats[path_id] = {}
        if step_id not in self.completion_stats[path_id]:
            self.completion_stats[path_id][step_id] = 0
        self.completion_stats[path_id][step_id] += 1

        self.logger.info(f"User {user_id} completed step: {path_id}/{step_id}")
        return True

    def skip_step(self, user_id: str, path_id: str, step_id: str, reason: str = "") -> bool:
        """
        Skip learning step

        Args:
            user_id: User ID
            path_id: Path ID
            step_id: Step ID
            reason: Reason for skipping

        Returns:
            bool: Whether successfully skipped
        """
        if (user_id not in self.user_progress or
            path_id not in self.user_progress[user_id]):
            return False

        progress = self.user_progress[user_id][path_id]
        path = self.learning_paths[path_id]

        if progress.current_step >= len(path.steps):
            return False

        current_step = path.steps[progress.current_step]
        if current_step.step_id != step_id:
            return False

        # Mark as skipped
        progress.skipped_steps.append(step_id)
        progress.current_step += 1
        progress.last_activity = datetime.now()

        self.logger.info(f"User {user_id} skipped step: {path_id}/{step_id}, reason: {reason}")
        return True

    def get_user_progress(self, user_id: str, path_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user learning progress

        Args:
            user_id: User ID
            path_id: Path ID, if None returns progress for all paths

        Returns:
            Dict[str, Any]: Progress information
        """
        if user_id not in self.user_progress:
            return {}

        if path_id:
            if path_id not in self.user_progress[user_id]:
                return {}

            progress = self.user_progress[user_id][path_id]
            path = self.learning_paths[path_id]

            completion_rate = len(progress.completed_steps) / len(path.steps) if path.steps else 0

            return {
                "path_id": path_id,
                "path_name": path.name,
                "current_step": progress.current_step,
                "total_steps": len(path.steps),
                "completed_steps": len(progress.completed_steps),
                "skipped_steps": len(progress.skipped_steps),
                "completion_rate": completion_rate,
                "start_time": progress.start_time.isoformat(),
                "last_activity": progress.last_activity.isoformat(),
                "estimated_remaining_time": self._calculate_remaining_time(progress, path)
            }
        else:
            # Return progress for all paths
            all_progress = {}
            for path_id, progress in self.user_progress[user_id].items():
                all_progress[path_id] = self.get_user_progress(user_id, path_id)

            return all_progress

    def _calculate_remaining_time(self, progress: UserProgress, path: LearningPath) -> int:
        """Calculate remaining learning time (in minutes)"""
        remaining_steps = len(path.steps) - progress.current_step
        if remaining_steps <= 0:
            return 0

        # Calculate estimated time for remaining steps
        remaining_time = 0
        for i in range(progress.current_step, len(path.steps)):
            remaining_time += path.steps[i].estimated_time_minutes

        return remaining_time

    def get_next_recommendation(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next learning recommendation

        Args:
            user_id: User ID

        Returns:
            Optional[Dict[str, Any]]: Learning recommendation
        """
        if user_id not in self.user_progress:
            # New user, recommend starting beginner path
            return {
                "type": "start_path",
                "path_id": "beginner_path",
                "path_name": "Beginner Path",
                "description": "Recommend starting from the beginner path to learn hibro's core features",
                "estimated_time": 45
            }

        # Find currently in-progress paths
        for path_id, progress in self.user_progress[user_id].items():
            path = self.learning_paths[path_id]

            if progress.current_step < len(path.steps):
                # Have uncompleted steps
                current_step = path.steps[progress.current_step]
                return {
                    "type": "continue_step",
                    "path_id": path_id,
                    "path_name": path.name,
                    "step_id": current_step.step_id,
                    "step_title": current_step.title,
                    "step_description": current_step.description,
                    "tools_to_learn": current_step.tools_to_learn,
                    "estimated_time": current_step.estimated_time_minutes
                }

        # All current paths completed, recommend next level path
        completed_paths = set(self.user_progress[user_id].keys())

        if "beginner_path" in completed_paths and "intermediate_path" not in completed_paths:
            return {
                "type": "start_path",
                "path_id": "intermediate_path",
                "path_name": "Intermediate Path",
                "description": "Congratulations on completing the beginner path! Recommend continuing to learn intermediate features",
                "estimated_time": 80
            }
        elif "intermediate_path" in completed_paths and "advanced_path" not in completed_paths:
            return {
                "type": "start_path",
                "path_id": "advanced_path",
                "path_name": "Advanced Expert Path",
                "description": "Congratulations on completing the intermediate path! Recommend learning advanced and enterprise-level features",
                "estimated_time": 90
            }

        # All paths completed
        return {
            "type": "completed",
            "message": "Congratulations! You have completed all learning paths and become a hibro expert user!",
            "suggestion": "Continue exploring advanced features or help other users learn"
        }

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        total_users = len(self.user_progress)
        total_completions = sum(
            len(progress.completed_steps)
            for user_progress in self.user_progress.values()
            for progress in user_progress.values()
        )

        # Path completion statistics
        path_completions = {}
        for path_id in self.learning_paths.keys():
            completed_count = 0
            for user_progress in self.user_progress.values():
                if path_id in user_progress:
                    progress = user_progress[path_id]
                    path = self.learning_paths[path_id]
                    if progress.current_step >= len(path.steps):
                        completed_count += 1
            path_completions[path_id] = completed_count

        # Most popular steps
        popular_steps = []
        for path_id, step_stats in self.completion_stats.items():
            for step_id, count in step_stats.items():
                popular_steps.append((f"{path_id}/{step_id}", count))

        popular_steps.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_users": total_users,
            "total_step_completions": total_completions,
            "path_completions": path_completions,
            "popular_steps": popular_steps[:10],
            "available_paths": len(self.learning_paths)
        }

    def cleanup_inactive_users(self, max_inactive_days: int = 30):
        """
        Clean up inactive user data

        Args:
            max_inactive_days: Maximum inactive days
        """
        cutoff_date = datetime.now() - timedelta(days=max_inactive_days)
        inactive_users = []

        for user_id, user_paths in self.user_progress.items():
            # Check user's last activity time
            last_activity = None
            for progress in user_paths.values():
                if not last_activity or progress.last_activity > last_activity:
                    last_activity = progress.last_activity

            if last_activity and last_activity < cutoff_date:
                inactive_users.append(user_id)

        # Clean up inactive users
        for user_id in inactive_users:
            del self.user_progress[user_id]

        if inactive_users:
            self.logger.info(f"Cleaned up learning data for {len(inactive_users)} inactive users")