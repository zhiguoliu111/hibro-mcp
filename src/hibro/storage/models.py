#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data model definitions
Defines core data models and ORM operations for the memory system
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from .database import DatabaseManager
from ..utils.config import Config
from ..utils.constants import (
    ALL_MEMORY_TYPES, PROJECT_MEMORY_TYPES, GLOBAL_MEMORY_TYPES,
    DEFAULT_IMPORTANCE_SCORES, LFU_COUNTER_INIT
)


@dataclass
class Memory:
    """Memory data model"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "conversation"
    importance: float = 0.5
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def update_access(self):
        """Update access information"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def increment_access(self):
        """Increment access count (alias for update_access)"""
        self.update_access()

    # ==================== LFU Related Methods ====================

    def get_lfu_counter(self) -> float:
        """
        Get LFU counter value

        Returns:
            LFU counter value, returns initial value based on access count if not exists
        """
        if self.metadata and 'lfu_counter' in self.metadata:
            return float(self.metadata['lfu_counter'])

        # If no LFU counter, use access count as initial value
        return float(self.access_count) if self.access_count > 0 else LFU_COUNTER_INIT

    def set_lfu_counter(self, value: float):
        """
        Set LFU counter value

        Args:
            value: LFU counter value
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata['lfu_counter'] = value

    def update_lfu_access(self, lfu_counter: float):
        """
        Update LFU access information (including counter and access time)

        Args:
            lfu_counter: New LFU counter value
        """
        self.set_lfu_counter(lfu_counter)
        self.update_access()

    # ==================== Project Context Related Methods ====================

    def get_project_path(self) -> Optional[str]:
        """
        Get project path

        Returns:
            Project path, returns None if it's global memory
        """
        if self.metadata and 'project_path' in self.metadata:
            return self.metadata['project_path']
        return None

    def set_project_path(self, project_path: Optional[str]):
        """
        Set project path

        Args:
            project_path: Project path, set to None for global memory
        """
        if self.metadata is None:
            self.metadata = {}

        if project_path:
            self.metadata['project_path'] = project_path
        elif 'project_path' in self.metadata:
            del self.metadata['project_path']

    def get_context_type(self) -> Optional[str]:
        """
        Get context type

        Returns:
            Context type (architecture/command/task), returns None if not set
        """
        if self.metadata and 'context_type' in self.metadata:
            return self.metadata['context_type']
        return None

    def set_context_type(self, context_type: Optional[str]):
        """
        Set context type

        Args:
            context_type: Context type (architecture/command/task)
        """
        if self.metadata is None:
            self.metadata = {}

        if context_type:
            self.metadata['context_type'] = context_type
        elif 'context_type' in self.metadata:
            del self.metadata['context_type']

    def is_active_task(self) -> bool:
        """
        Check if it's an active task

        Returns:
            Whether it's an active task
        """
        return (self.memory_type == 'active_task' and
                self.metadata and
                self.metadata.get('is_active', False))

    def set_active_status(self, is_active: bool):
        """
        Set active status (only valid for active_task type)

        Args:
            is_active: Whether it's active
        """
        if self.memory_type == 'active_task':
            if self.metadata is None:
                self.metadata = {}
            self.metadata['is_active'] = is_active

    def is_global_memory(self) -> bool:
        """
        Check if it's global memory

        Returns:
            Whether it's global memory (memory without project path)
        """
        project_path = self.get_project_path()
        return not project_path

    def is_project_memory(self, project_path: str) -> bool:
        """
        Check if it's memory for specified project

        Args:
            project_path: Project path

        Returns:
            Whether it's memory for specified project
        """
        memory_project_path = self.get_project_path()
        return memory_project_path == project_path

    # ==================== Memory Type Validation ====================

    def validate_memory_type(self) -> bool:
        """
        Validate if memory type is valid

        Returns:
            Whether memory type is valid
        """
        return self.memory_type in ALL_MEMORY_TYPES

    def is_project_specific_type(self) -> bool:
        """
        Check if it's project-specific memory type

        Returns:
            Whether it's project-specific memory type
        """
        return self.memory_type in PROJECT_MEMORY_TYPES

    def get_default_importance(self) -> float:
        """
        Get default importance score for this memory type

        Returns:
            Default importance score
        """
        return DEFAULT_IMPORTANCE_SCORES.get(self.memory_type, 0.5)

    # ==================== Convenience Methods ====================

    def setup_as_project_memory(self, project_path: str, context_type: str,
                               is_active: bool = False) -> 'Memory':
        """
        Set up as project memory

        Args:
            project_path: Project path
            context_type: Context type
            is_active: Whether it's active (only valid for active_task)

        Returns:
            Self reference, supports method chaining
        """
        self.set_project_path(project_path)
        self.set_context_type(context_type)

        if self.memory_type == 'active_task':
            self.set_active_status(is_active)

        return self

    def setup_as_global_memory(self) -> 'Memory':
        """
        Set up as global memory

        Returns:
            Self reference, supports method chaining
        """
        self.set_project_path(None)
        self.set_context_type(None)

        if self.memory_type == 'active_task':
            self.set_active_status(False)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Handle datetime objects
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory object from dictionary"""
        # Handle datetime fields
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_accessed' in data and isinstance(data['last_accessed'], str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])

        # Handle metadata field
        if 'metadata' in data and isinstance(data['metadata'], str):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except json.JSONDecodeError:
                data['metadata'] = {}

        return cls(**data)


@dataclass
class Project:
    """Project data model"""
    id: Optional[int] = None
    name: str = ""
    path: str = ""
    tech_stack: Optional[List[str]] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self):
        """Post-initialization processing"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.tech_stack is None:
            self.tech_stack = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create Project object from dictionary"""
        # Handle datetime fields
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_accessed' in data and isinstance(data['last_accessed'], str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])

        # Handle tech_stack field
        if 'tech_stack' in data and isinstance(data['tech_stack'], str):
            try:
                data['tech_stack'] = json.loads(data['tech_stack'])
            except json.JSONDecodeError:
                data['tech_stack'] = []

        return cls(**data)


@dataclass
class Preference:
    """Preference settings data model"""
    id: Optional[int] = None
    category: str = ""
    key: str = ""
    value: str = ""
    description: Optional[str] = None
    confidence_score: float = 0.5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preference':
        """Create Preference object from dictionary"""
        # Handle datetime fields
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)


class MemoryRepository:
    """Memory data repository - handles CRUD operations for memories"""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize memory repository

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.memory_repository')

    def create_memory(self, memory: Memory) -> int:
        """
        Create new memory

        Args:
            memory: Memory object

        Returns:
            Created memory ID
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO memories (content, memory_type, importance, category,
                                        created_at, last_accessed, access_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.category,
                    memory.created_at,
                    memory.last_accessed,
                    memory.access_count,
                    json.dumps(memory.metadata) if memory.metadata else None
                ))

                memory_id = cursor.lastrowid
                conn.commit()

                self.logger.info(f"Memory created successfully: ID={memory_id}")
                return memory_id

        except Exception as e:
            self.logger.error(f"Failed to create memory: {e}")
            raise

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Get memory by ID

        Args:
            memory_id: Memory ID

        Returns:
            Memory object, returns None if not exists
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories WHERE id = ?
                """, (memory_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_memory(row)
                return None

        except Exception as e:
            self.logger.error(f"Failed to get memory: {e}")
            return None

    def update_memory(self, memory: Memory) -> bool:
        """
        Update memory

        Args:
            memory: Memory object

        Returns:
            Whether update was successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE memories
                    SET content = ?, memory_type = ?, importance = ?, category = ?,
                        last_accessed = ?, access_count = ?, metadata = ?
                    WHERE id = ?
                """, (
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.category,
                    memory.last_accessed,
                    memory.access_count,
                    json.dumps(memory.metadata) if memory.metadata else None,
                    memory.id
                ))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Memory updated successfully: ID={memory.id}")
                else:
                    self.logger.warning(f"Memory update failed, record not found: ID={memory.id}")

                return success

        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
            return False

    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete memory

        Args:
            memory_id: Memory ID

        Returns:
            Whether deletion was successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Memory deleted successfully: ID={memory_id}")
                else:
                    self.logger.warning(f"Memory deletion failed, record not found: ID={memory_id}")

                return success

        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False

    def search_memories(self, query: str = "", memory_type: Optional[str] = None,
                       category: Optional[str] = None, min_importance: float = 0.0,
                       project_path: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Memory]:
        """
        Search memories

        Args:
            query: Search keywords
            memory_type: Memory type filter
            category: Category filter
            min_importance: Minimum importance
            project_path: Project path filter, if None then no project filtering
            limit: Return count limit
            offset: Offset

        Returns:
            Memory list
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build query conditions
                conditions = ["importance >= ?"]
                params = [min_importance]

                if query:
                    # Improvement: support word segmentation search (multiple keywords separated by spaces)
                    # All keywords must match (AND logic)
                    keywords = query.split()
                    keyword_conditions = []
                    for keyword in keywords:
                        keyword_conditions.append("content LIKE ?")
                        params.append(f"%{keyword}%")
                    conditions.append(f"({' AND '.join(keyword_conditions)})")

                if memory_type:
                    conditions.append("memory_type = ?")
                    params.append(memory_type)

                if category:
                    conditions.append("category = ?")
                    params.append(category)

                # Add project path filter
                if project_path is not None:
                    if project_path == "":
                        # Empty string means query global memories (memories without project_path)
                        conditions.append("(metadata IS NULL OR json_extract(metadata, '$.project_path') IS NULL OR json_extract(metadata, '$.project_path') = '')")
                    else:
                        # Query memories for specified project
                        conditions.append("json_extract(metadata, '$.project_path') = ?")
                        params.append(project_path)

                where_clause = " AND ".join(conditions)
                params.extend([limit, offset])

                cursor = conn.execute(f"""
                    SELECT * FROM memories
                    WHERE {where_clause}
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ? OFFSET ?
                """, params)

                memories = []
                for row in cursor.fetchall():
                    memories.append(self._row_to_memory(row))

                return memories

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []

    def get_memories_by_type(self, memory_type: str, limit: int = 50) -> List[Memory]:
        """
        Get memories by type

        Args:
            memory_type: Memory type
            limit: Return count limit

        Returns:
            Memory list
        """
        return self.search_memories(memory_type=memory_type, limit=limit)

    def _row_to_memory(self, row) -> Memory:
        """
        Convert database row to Memory object

        Args:
            row: Database row

        Returns:
            Memory object
        """
        metadata = None
        if row['metadata']:
            try:
                metadata = json.loads(row['metadata'])
            except json.JSONDecodeError:
                metadata = {}

        return Memory(
            id=row['id'],
            content=row['content'],
            memory_type=row['memory_type'],
            importance=row['importance'],
            category=row['category'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            access_count=row['access_count'],
            metadata=metadata
        )


class ProjectRepository:
    """Project data repository - handles CRUD operations for projects"""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize project repository

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.project_repository')

    def create_project(self, project: Project) -> int:
        """
        Create new project

        Args:
            project: Project object

        Returns:
            Created project ID
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO projects (name, path, tech_stack, description,
                                        created_at, last_accessed, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    project.name,
                    project.path,
                    json.dumps(project.tech_stack) if project.tech_stack else None,
                    project.description,
                    project.created_at,
                    project.last_accessed,
                    project.is_active
                ))

                project_id = cursor.lastrowid
                conn.commit()

                self.logger.info(f"Project created successfully: ID={project_id}")
                return project_id

        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            raise

    def get_project_by_path(self, path: str) -> Optional[Project]:
        """
        Get project by path

        Args:
            path: Project path

        Returns:
            Project object, returns None if not exists
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM projects WHERE path = ?", (path,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_project(row)
                return None

        except Exception as e:
            self.logger.error(f"Failed to get project: {e}")
            return None

    def _row_to_project(self, row) -> Project:
        """
        Convert database row to Project object

        Args:
            row: Database row

        Returns:
            Project object
        """
        tech_stack = []
        if row['tech_stack']:
            try:
                tech_stack = json.loads(row['tech_stack'])
            except json.JSONDecodeError:
                tech_stack = []

        return Project(
            id=row['id'],
            name=row['name'],
            path=row['path'],
            tech_stack=tech_stack,
            description=row['description'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            is_active=bool(row['is_active'])
        )