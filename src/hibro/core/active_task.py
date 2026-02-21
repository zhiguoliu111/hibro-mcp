#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active Task Management Module
Implements project active task management logic, ensuring only one active task per project
"""

import logging
from typing import Optional, List
from datetime import datetime

from ..storage.models import Memory, MemoryRepository
from ..utils.constants import DEFAULT_IMPORTANCE_SCORES


class ActiveTaskManager:
    """Active Task Manager

    Responsible for managing project active tasks, ensuring:
    - Only one active task per project
    - Active tasks have highest priority
    - Correct task status transitions
    """

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize active task manager

        Args:
            memory_repo: Memory data repository
        """
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.active_task_manager')

    def set_active_task(self, project_path: str, task_content: str,
                       importance: Optional[float] = None) -> int:
        """
        Set project active task

        Only one active task per project, setting new task will automatically deactivate other tasks

        Args:
            project_path: Project path
            task_content: Task content
            importance: Importance score, uses default value if None

        Returns:
            Created task memory ID

        Raises:
            Exception: Raised when task creation fails
        """
        try:
            # 1. Deactivate other active tasks for this project
            self._deactivate_project_tasks(project_path)

            # 2. Create new active task
            if importance is None:
                importance = DEFAULT_IMPORTANCE_SCORES.get('active_task', 1.0)

            task_memory = Memory(
                content=task_content,
                memory_type='active_task',
                importance=importance,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0
            )

            # Set project context information
            task_memory.setup_as_project_memory(
                project_path=project_path,
                context_type='task',
                is_active=True
            )

            # Set high LFU counter to ensure priority
            task_memory.set_lfu_counter(100.0)

            # Save to database
            memory_id = self.memory_repo.create_memory(task_memory)

            self.logger.info(f"Active task set successfully: project={project_path}, ID={memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Set active task failed: {e}")
            raise

    def get_active_task(self, project_path: str) -> Optional[Memory]:
        """
        Get project's current active task

        Args:
            project_path: Project path

        Returns:
            Active task memory object, returns None if none exists
        """
        try:
            # Search for active tasks in this project
            memories = self.memory_repo.search_memories(
                memory_type='active_task',
                limit=100  # Get enough tasks for filtering
            )

            # Filter out active tasks for the specified project
            for memory in memories:
                if (memory.is_project_memory(project_path) and
                    memory.is_active_task()):
                    return memory

            return None

        except Exception as e:
            self.logger.error(f"Get active task failed: {e}")
            return None

    def complete_active_task(self, project_path: str) -> bool:
        """
        Complete project's current active task

        Mark active task as inactive status

        Args:
            project_path: Project path

        Returns:
            Whether task completion was successful
        """
        try:
            active_task = self.get_active_task(project_path)
            if not active_task:
                self.logger.warning(f"Project has no active task: {project_path}")
                return False

            # Set to inactive status
            active_task.set_active_status(False)

            # Update database
            success = self.memory_repo.update_memory(active_task)

            if success:
                self.logger.info(f"Active task completed: project={project_path}, ID={active_task.id}")
            else:
                self.logger.error(f"Update task status failed: project={project_path}, ID={active_task.id}")

            return success

        except Exception as e:
            self.logger.error(f"Complete active task failed: {e}")
            return False

    def list_project_tasks(self, project_path: str, include_inactive: bool = True) -> List[Memory]:
        """
        List all tasks for a project

        Args:
            project_path: Project path
            include_inactive: Whether to include inactive tasks

        Returns:
            List of task memories
        """
        try:
            # Search for all tasks in this project
            memories = self.memory_repo.search_memories(
                memory_type='active_task',
                limit=100
            )

            # Filter out tasks for the specified project
            project_tasks = []
            for memory in memories:
                if memory.is_project_memory(project_path):
                    if include_inactive or memory.is_active_task():
                        project_tasks.append(memory)

            # Sort by importance and creation time
            project_tasks.sort(
                key=lambda m: (m.importance, m.created_at),
                reverse=True
            )

            return project_tasks

        except Exception as e:
            self.logger.error(f"List project tasks failed: {e}")
            return []

    def update_task_content(self, project_path: str, new_content: str) -> bool:
        """
        Update project active task content

        Args:
            project_path: Project path
            new_content: New task content

        Returns:
            Whether update was successful
        """
        try:
            active_task = self.get_active_task(project_path)
            if not active_task:
                self.logger.warning(f"Project has no active task: {project_path}")
                return False

            # Update content
            active_task.content = new_content
            active_task.last_accessed = datetime.now()

            # Update database
            success = self.memory_repo.update_memory(active_task)

            if success:
                self.logger.info(f"Task content updated successfully: project={project_path}, ID={active_task.id}")
            else:
                self.logger.error(f"Task content update failed: project={project_path}, ID={active_task.id}")

            return success

        except Exception as e:
            self.logger.error(f"Update task content failed: {e}")
            return False

    def _deactivate_project_tasks(self, project_path: str):
        """
        Set all active tasks in project to inactive status

        Args:
            project_path: Project path
        """
        try:
            # Get all tasks for this project
            memories = self.memory_repo.search_memories(
                memory_type='active_task',
                limit=100
            )

            # Find active tasks for this project and set them to inactive
            for memory in memories:
                if (memory.is_project_memory(project_path) and
                    memory.is_active_task()):

                    memory.set_active_status(False)
                    self.memory_repo.update_memory(memory)

                    self.logger.debug(f"Task set to inactive: ID={memory.id}")

        except Exception as e:
            self.logger.error(f"Deactivate project tasks failed: {e}")
            raise

    def get_all_active_tasks(self) -> List[Memory]:
        """
        Get active tasks for all projects

        Returns:
            List of all active tasks
        """
        try:
            # Search for all active tasks
            memories = self.memory_repo.search_memories(
                memory_type='active_task',
                limit=1000
            )

            # Filter out active tasks
            active_tasks = []
            for memory in memories:
                if memory.is_active_task():
                    active_tasks.append(memory)

            # Sort by project path and importance
            active_tasks.sort(
                key=lambda m: (m.get_project_path() or '', m.importance),
                reverse=True
            )

            return active_tasks

        except Exception as e:
            self.logger.error(f"Get all active tasks failed: {e}")
            return []

    def get_task_statistics(self, project_path: Optional[str] = None) -> dict:
        """
        Get task statistics

        Args:
            project_path: Project path, if None then statistics for all projects

        Returns:
            Task statistics dictionary
        """
        try:
            if project_path:
                # Statistics for specified project
                tasks = self.list_project_tasks(project_path, include_inactive=True)
                active_count = sum(1 for task in tasks if task.is_active_task())

                return {
                    'project_path': project_path,
                    'total_tasks': len(tasks),
                    'active_tasks': active_count,
                    'inactive_tasks': len(tasks) - active_count
                }
            else:
                # Statistics for all projects
                all_tasks = self.memory_repo.search_memories(
                    memory_type='active_task',
                    limit=1000
                )

                active_count = sum(1 for task in all_tasks if task.is_active_task())
                projects = set()

                for task in all_tasks:
                    project_path = task.get_project_path()
                    if project_path:
                        projects.add(project_path)

                return {
                    'total_tasks': len(all_tasks),
                    'active_tasks': active_count,
                    'inactive_tasks': len(all_tasks) - active_count,
                    'projects_with_tasks': len(projects)
                }

        except Exception as e:
            self.logger.error(f"Get task statistics failed: {e}")
            return {'error': str(e)}


# Convenience functions
def create_active_task_manager(memory_repo: MemoryRepository) -> ActiveTaskManager:
    """
    Create active task manager (convenience function)

    Args:
        memory_repo: Memory data repository

    Returns:
        Active task manager instance
    """
    return ActiveTaskManager(memory_repo)