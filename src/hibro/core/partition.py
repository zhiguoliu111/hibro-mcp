#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Partition Management Module
Implements separate management for global and project memories, supporting independent hot data ranking
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from ..storage.models import Memory, MemoryRepository
from ..core.lfu import LFUCalculator, get_hot_memories
from ..utils.constants import (
    GLOBAL_HOT_TOP_N, PROJECT_HOT_TOP_N, PROJECT_MEMORY_TYPES,
    GLOBAL_MEMORY_TYPES, ALL_MEMORY_TYPES
)


class MemoryPartition:
    """Memory partition manager

    Responsible for managing separation of global and project memories:
    - Global memories: User preferences, technical decisions, and other cross-project information
    - Project memories: Project architecture, specific commands, active tasks, and other project-specific information
    - Hot data independently ranked by partition to avoid cross-project interference
    """

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize memory partition manager

        Args:
            memory_repo: Memory data repository
        """
        self.memory_repo = memory_repo
        self.lfu_calculator = LFUCalculator()
        self.logger = logging.getLogger('hibro.memory_partition')

    def get_global_memories(self, memory_types: Optional[List[str]] = None,
                           min_importance: float = 0.0) -> List[Memory]:
        """
        Get global memories (memories without project_path)

        Args:
            memory_types: Memory type filter, uses all global types if None
            min_importance: Minimum importance threshold

        Returns:
            Global memory list
        """
        try:
            # If no type specified, use all global memory types
            if memory_types is None:
                memory_types = list(GLOBAL_MEMORY_TYPES)

            # Get all memories meeting criteria
            all_memories = []
            seen_ids = set()  # For deduplication

            for memory_type in memory_types:
                memories = self.memory_repo.search_memories(
                    memory_type=memory_type,
                    min_importance=min_importance,
                    limit=1000  # Get enough memories for subsequent filtering
                )

                # Deduplicate and filter global memories
                for memory in memories:
                    if memory.id not in seen_ids and self._is_global_memory(memory):
                        all_memories.append(memory)
                        seen_ids.add(memory.id)

            self.logger.debug(f"Retrieved global memories: types={memory_types}, count={len(all_memories)}")
            return all_memories

        except Exception as e:
            self.logger.error(f"Failed to get global memories: {e}")
            return []

    def get_project_memories(self, project_path: str, memory_types: Optional[List[str]] = None,
                           min_importance: float = 0.0) -> List[Memory]:
        """
        Get project-specific memories

        Args:
            project_path: Project path
            memory_types: Memory type filter, uses all memory types if None
            min_importance: Minimum importance threshold

        Returns:
            Project memory list
        """
        try:
            # If no type specified, use all memory types (project memories may include global types)
            if memory_types is None:
                memory_types = list(ALL_MEMORY_TYPES)

            # Get all memories meeting criteria
            all_memories = []
            seen_ids = set()  # For deduplication

            for memory_type in memory_types:
                memories = self.memory_repo.search_memories(
                    memory_type=memory_type,
                    min_importance=min_importance,
                    limit=1000
                )

                # Deduplicate and filter memories for specified project
                for memory in memories:
                    if memory.id not in seen_ids and self._is_project_memory(memory, project_path):
                        all_memories.append(memory)
                        seen_ids.add(memory.id)

            self.logger.debug(f"Retrieved project memories: project={project_path}, types={memory_types}, count={len(all_memories)}")
            return all_memories

        except Exception as e:
            self.logger.error(f"Failed to get project memories: {e}")
            return []

    def get_context_for_project(self, project_path: Optional[str] = None,
                              current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get context: global hot data + project hot data

        Args:
            project_path: Project path, returns only global context if None
            current_time: Current time for LFU calculation

        Returns:
            Context dictionary containing global and project hot data
        """
        if current_time is None:
            current_time = datetime.now()

        try:
            # 1. Get global hot data
            global_memories = self.get_global_memories()
            global_hot = self.lfu_calculator.get_hot_memories(
                global_memories, current_time, GLOBAL_HOT_TOP_N
            )

            # 2. Get project hot data
            project_hot = []
            project_context = {}

            if project_path:
                project_memories = self.get_project_memories(project_path)
                project_hot = self.lfu_calculator.get_hot_memories(
                    project_memories, current_time, PROJECT_HOT_TOP_N
                )

                # 3. Ensure active tasks are always included in project hot data
                active_tasks = self._get_active_tasks(project_memories)
                for task in active_tasks:
                    if task not in project_hot:
                        project_hot.append(task)

                # 4. Build project context structure
                project_context = self._build_project_context(project_memories)

            context = {
                'global_hot': global_hot,
                'project_hot': project_hot,
                'project_context': project_context,
                'stats': {
                    'global_hot_count': len(global_hot),
                    'project_hot_count': len(project_hot),
                    'project_path': project_path,
                    'timestamp': current_time.isoformat()
                }
            }

            self.logger.info(f"Context retrieval complete: global_hot={len(global_hot)}, project_hot={len(project_hot)}")
            return context

        except Exception as e:
            self.logger.error(f"Failed to get project context: {e}")
            return {
                'global_hot': [],
                'project_hot': [],
                'project_context': {},
                'stats': {'error': str(e)}
            }

    def get_memory_scope(self, memory: Memory) -> str:
        """
        Determine memory ownership: global or project

        Args:
            memory: Memory object

        Returns:
            Memory ownership scope: 'global' or 'project'
        """
        if self._is_global_memory(memory):
            return 'global'
        else:
            return 'project'

    def get_project_path_from_memory(self, memory: Memory) -> Optional[str]:
        """
        Get project path from memory

        Args:
            memory: Memory object

        Returns:
            Project path, returns None if global memory
        """
        if memory.metadata and 'project_path' in memory.metadata:
            return memory.metadata['project_path']
        return None

    def _is_global_memory(self, memory: Memory) -> bool:
        """
        Determine if memory is global

        Args:
            memory: Memory object

        Returns:
            Whether memory is global
        """
        # Memories without project_path are considered global
        if not memory.metadata or 'project_path' not in memory.metadata:
            return True

        # Empty or None project_path is also considered global
        project_path = memory.metadata.get('project_path')
        return not project_path

    def _is_project_memory(self, memory: Memory, project_path: str) -> bool:
        """
        Determine if memory belongs to specified project

        Args:
            memory: Memory object
            project_path: Project path

        Returns:
            Whether memory belongs to specified project
        """
        if not memory.metadata or 'project_path' not in memory.metadata:
            return False

        memory_project_path = memory.metadata.get('project_path')
        return memory_project_path == project_path

    def _get_active_tasks(self, project_memories: List[Memory]) -> List[Memory]:
        """
        Get active tasks

        Args:
            project_memories: Project memory list

        Returns:
            Active task list
        """
        active_tasks = []
        for memory in project_memories:
            if (memory.memory_type == 'active_task' and
                memory.metadata and
                memory.metadata.get('is_active', False)):
                active_tasks.append(memory)

        return active_tasks

    def _build_project_context(self, project_memories: List[Memory]) -> Dict[str, Any]:
        """
        Build project context structure

        Args:
            project_memories: Project memory list

        Returns:
            Project context dictionary
        """
        context = {
            'project_name': None,
            'project_type': None,
            'tech_stack': [],
            'languages': [],
            'statistics': {},
            'architecture': None,
            'commands': [],
            'active_task': None,
            'snapshot': None
        }

        for memory in project_memories:
            # Handle project snapshot (from scan_project)
            if memory.memory_type == 'project' and memory.category == 'project_snapshot':
                context['snapshot'] = memory.content
                # Extract metadata if available
                if memory.metadata:
                    context['project_name'] = memory.metadata.get('project_name')
                    context['project_type'] = memory.metadata.get('project_type')
                    context['tech_stack'] = memory.metadata.get('tech_stack', [])
                    context['languages'] = memory.metadata.get('languages', [])
                    context['statistics'] = memory.metadata.get('statistics', {})
            elif memory.memory_type == 'project_architecture':
                context['architecture'] = memory.content
            elif memory.memory_type == 'project_command':
                context['commands'].append(memory.content)
            elif (memory.memory_type == 'active_task' and
                  memory.metadata and
                  memory.metadata.get('is_active', False)):
                context['active_task'] = memory.content

        return context

    def update_memory_partition_info(self, memory: Memory, project_path: Optional[str] = None,
                                   context_type: Optional[str] = None, is_active: bool = False) -> Memory:
        """
        Update memory partition information

        Args:
            memory: Memory object
            project_path: Project path
            context_type: Context type (architecture/command/task)
            is_active: Whether active (only valid for active_task)

        Returns:
            Updated memory object
        """
        if memory.metadata is None:
            memory.metadata = {}

        # Update project path
        if project_path:
            memory.metadata['project_path'] = project_path

        # Update context type
        if context_type:
            memory.metadata['context_type'] = context_type

        # Update active status
        if memory.memory_type == 'active_task':
            memory.metadata['is_active'] = is_active

        return memory

    def get_partition_statistics(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get partition statistics

        Args:
            project_path: Project path, statistics for all partitions if None

        Returns:
            Partition statistics information
        """
        try:
            stats = {
                'global_memories': 0,
                'project_memories': 0,
                'total_memories': 0,
                'memory_types': {},
                'project_path': project_path
            }

            # Get global memory statistics
            try:
                global_memories = self.get_global_memories()
                stats['global_memories'] = len(global_memories)

                # Count global memory type distribution
                for memory in global_memories:
                    memory_type = memory.memory_type
                    if memory_type not in stats['memory_types']:
                        stats['memory_types'][memory_type] = {'global': 0, 'project': 0}
                    stats['memory_types'][memory_type]['global'] += 1
            except Exception as e:
                self.logger.error(f"Failed to get global memory statistics: {e}")
                # If getting global memories fails, return error information
                return {'error': f"Failed to get global memory statistics: {e}"}

            # If project path specified, get project memory statistics
            if project_path:
                try:
                    project_memories = self.get_project_memories(project_path)
                    stats['project_memories'] = len(project_memories)

                    # Count project memory type distribution
                    for memory in project_memories:
                        memory_type = memory.memory_type
                        if memory_type not in stats['memory_types']:
                            stats['memory_types'][memory_type] = {'global': 0, 'project': 0}
                        stats['memory_types'][memory_type]['project'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to get project memory statistics: {e}")
                    # If getting project memories fails, return error information
                    return {'error': f"Failed to get project memory statistics: {e}"}

            stats['total_memories'] = stats['global_memories'] + stats['project_memories']

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get partition statistics: {e}")
            return {'error': str(e)}


# Convenience functions
def create_memory_partition(memory_repo: MemoryRepository) -> MemoryPartition:
    """
    Create memory partition manager (convenience function)

    Args:
        memory_repo: Memory data repository

    Returns:
        Memory partition manager instance
    """
    return MemoryPartition(memory_repo)