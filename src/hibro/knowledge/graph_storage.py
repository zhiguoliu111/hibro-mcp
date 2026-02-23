#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge graph storage layer

Provides storage, retrieval and management of knowledge graph nodes and relations
"""

import json
import logging
import hashlib
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..storage.database import DatabaseManager
from ..storage.models import Memory


class GraphNodeType(Enum):
    """Graph node type"""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    API_ENDPOINT = "api_endpoint"


class RelationType(Enum):
    """Relation type"""
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES = "uses"
    CONTAINS = "contains"


@dataclass
class GraphNode:
    """
    Knowledge graph node
    """
    node_id: Optional[int] = None
    node_type: GraphNodeType = GraphNodeType.FILE
    name: str = ""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    importance: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    file_hash: Optional[str] = None
    project_path: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_memory(self) -> Memory:
        """
        Convert to Memory object

        Returns:
            Memory object
        """
        # Build content string
        if self.node_type == GraphNodeType.FILE:
            content = f"File: {self.file_path}"
        elif self.node_type == GraphNodeType.CLASS:
            content = f"Class: {self.name} in {self.file_path}:{self.line_number}"
        elif self.node_type == GraphNodeType.FUNCTION:
            content = f"Function: {self.name} in {self.file_path}:{self.line_number}"
        elif self.node_type == GraphNodeType.MODULE:
            content = f"Module: {self.name}"
        elif self.node_type == GraphNodeType.API_ENDPOINT:
            content = f"API: {self.name}"
        else:
            content = f"{self.node_type.value}: {self.name}"

        # Build graph_metadata
        graph_metadata = {
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            **self.metadata
        }

        # Build complete metadata (including project_path)
        full_metadata = {}
        if self.project_path:
            full_metadata["project_path"] = self.project_path

        return Memory(
            id=self.node_id,
            content=content,
            memory_type="knowledge_graph",
            category="knowledge_graph",
            importance=self.importance,
            metadata=full_metadata,
            created_at=self.created_at
        )

    @classmethod
    def from_memory(cls, memory: Memory) -> 'GraphNode':
        """
        Create from Memory object

        Args:
            memory: Memory object

        Returns:
            GraphNode object
        """
        # Extract information from memory
        graph_metadata = {}
        if hasattr(memory, 'graph_metadata') and memory.graph_metadata:
            try:
                if isinstance(memory.graph_metadata, str):
                    graph_metadata = json.loads(memory.graph_metadata)
                else:
                    graph_metadata = memory.graph_metadata
            except (json.JSONDecodeError, TypeError):
                graph_metadata = {}

        # Extract project_path
        project_path = None
        if memory.metadata and 'project_path' in memory.metadata:
            project_path = memory.metadata['project_path']

        # Extract node type
        node_type_str = getattr(memory, 'graph_node_type', None) or GraphNodeType.FILE.value
        try:
            node_type = GraphNodeType(node_type_str)
        except ValueError:
            node_type = GraphNodeType.FILE

        return cls(
            node_id=memory.id,
            node_type=node_type,
            name=graph_metadata.get('name', ''),
            file_path=graph_metadata.get('file_path'),
            line_number=graph_metadata.get('line_number'),
            importance=memory.importance,
            metadata={k: v for k, v in graph_metadata.items()
                     if k not in ['name', 'file_path', 'line_number']},
            file_hash=getattr(memory, 'file_hash', None),
            project_path=project_path,
            created_at=memory.created_at
        )


@dataclass
class GraphRelation:
    """
    Knowledge graph relation
    """
    relation_id: Optional[int] = None
    source_node_id: int = 0
    target_node_id: int = 0
    relation_type: RelationType = RelationType.USES
    weight: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class GraphStorage:
    """
    Knowledge graph storage layer
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize storage layer

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger('hibro.graph_storage')

    # ==================== Node Operations ====================

    def create_node(self, node: GraphNode) -> int:
        """
        Create node

        Args:
            node: Graph node

        Returns:
            Node ID
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build graph_metadata JSON
                graph_metadata = {
                    "name": node.name,
                    "file_path": node.file_path,
                    "line_number": node.line_number,
                    **node.metadata
                }

                # Build complete metadata (including project_path)
                full_metadata = {}
                if node.project_path:
                    full_metadata["project_path"] = node.project_path

                cursor = conn.execute("""
                    INSERT INTO memories (
                        content, memory_type, category, importance,
                        graph_node_type, graph_metadata, file_hash,
                        metadata, created_at, last_accessed, access_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{node.node_type.value}: {node.name}",
                    "knowledge_graph",
                    "knowledge_graph",
                    node.importance,
                    node.node_type.value,
                    json.dumps(graph_metadata, ensure_ascii=False),
                    node.file_hash,
                    json.dumps(full_metadata, ensure_ascii=False) if full_metadata else None,
                    node.created_at,
                    node.created_at,
                    0
                ))

                node_id = cursor.lastrowid
                conn.commit()

                self.logger.info(f"Node created: ID={node_id}, type={node.node_type.value}, name={node.name}")
                return node_id

        except Exception as e:
            self.logger.error(f"Node creation failed: {e}")
            raise

    def get_node(self, node_id: int) -> Optional[GraphNode]:
        """
        Get node

        Args:
            node_id: Node ID

        Returns:
            Graph node, None if not exists
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories
                    WHERE id = ? AND category = 'knowledge_graph'
                """, (node_id,))

                row = cursor.fetchone()
                if row:
                    memory = self._row_to_memory(row)
                    return GraphNode.from_memory(memory)
                return None

        except Exception as e:
            self.logger.error(f"Get node failed: {e}")
            return None

    def update_node(self, node: GraphNode) -> bool:
        """
        Update node

        Args:
            node: Graph node

        Returns:
            Whether successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build graph_metadata JSON
                graph_metadata = {
                    "name": node.name,
                    "file_path": node.file_path,
                    "line_number": node.line_number,
                    **node.metadata
                }

                # Build complete metadata
                full_metadata = {}
                if node.project_path:
                    full_metadata["project_path"] = node.project_path

                cursor = conn.execute("""
                    UPDATE memories
                    SET content = ?,
                        importance = ?,
                        graph_node_type = ?,
                        graph_metadata = ?,
                        file_hash = ?,
                        metadata = ?
                    WHERE id = ? AND category = 'knowledge_graph'
                """, (
                    f"{node.node_type.value}: {node.name}",
                    node.importance,
                    node.node_type.value,
                    json.dumps(graph_metadata, ensure_ascii=False),
                    node.file_hash,
                    json.dumps(full_metadata, ensure_ascii=False) if full_metadata else None,
                    node.node_id
                ))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Node updated: ID={node.node_id}")
                else:
                    self.logger.warning(f"Node update failed, record not found: ID={node.node_id}")

                return success

        except Exception as e:
            self.logger.error(f"Node update failed: {e}")
            return False

    def delete_node(self, node_id: int) -> bool:
        """
        Delete node

        Args:
            node_id: Node ID

        Returns:
            Whether successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM memories
                    WHERE id = ? AND category = 'knowledge_graph'
                """, (node_id,))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Node deleted: ID={node_id}")
                else:
                    self.logger.warning(f"Node deletion failed, record not found: ID={node_id}")

                return success

        except Exception as e:
            self.logger.error(f"Node deletion failed: {e}")
            return False

    def search_nodes(
        self,
        node_type: Optional[GraphNodeType] = None,
        project_path: Optional[str] = None,
        file_path: Optional[str] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """
        Search nodes

        Args:
            node_type: Node type filter
            project_path: Project path filter
            file_path: File path filter
            name_pattern: Name pattern matching
            limit: Return count limit

        Returns:
            Node list
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Build query conditions
                conditions = ["category = 'knowledge_graph'"]
                params = []

                if node_type:
                    conditions.append("graph_node_type = ?")
                    params.append(node_type.value)

                if project_path is not None:
                    conditions.append("json_extract(metadata, '$.project_path') = ?")
                    params.append(project_path)

                if file_path:
                    conditions.append("json_extract(graph_metadata, '$.file_path') = ?")
                    params.append(file_path)

                if name_pattern:
                    conditions.append("json_extract(graph_metadata, '$.name') LIKE ?")
                    params.append(f"%{name_pattern}%")

                where_clause = " AND ".join(conditions)
                params.append(limit)

                cursor = conn.execute(f"""
                    SELECT * FROM memories
                    WHERE {where_clause}
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """, params)

                nodes = []
                for row in cursor.fetchall():
                    memory = self._row_to_memory(row)
                    nodes.append(GraphNode.from_memory(memory))

                return nodes

        except Exception as e:
            self.logger.error(f"Search nodes failed: {e}")
            return []

    def get_nodes_by_file(self, file_path: str, project_path: Optional[str] = None) -> List[GraphNode]:
        """
        Get all nodes in a file

        Args:
            file_path: File path
            project_path: Project path

        Returns:
            Node list
        """
        return self.search_nodes(file_path=file_path, project_path=project_path, limit=1000)

    # ==================== Relation Operations ====================

    def create_relation(self, relation: GraphRelation) -> int:
        """
        Create relation

        Args:
            relation: Graph relation

        Returns:
            Relation ID
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO knowledge_relations (
                        source_memory_id, target_memory_id,
                        relation_type, weight, metadata, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    relation.source_node_id,
                    relation.target_node_id,
                    relation.relation_type.value,
                    relation.weight,
                    json.dumps(relation.metadata, ensure_ascii=False) if relation.metadata else None,
                    relation.created_at
                ))

                relation_id = cursor.lastrowid
                conn.commit()

                self.logger.info(f"Relation created: ID={relation_id}, type={relation.relation_type.value}")
                return relation_id

        except Exception as e:
            self.logger.error(f"Relation creation failed: {e}")
            raise

    def get_relation(self, relation_id: int) -> Optional[GraphRelation]:
        """
        Get relation

        Args:
            relation_id: Relation ID

        Returns:
            Graph relation, None if not exists
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_relations
                    WHERE id = ?
                """, (relation_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_relation(row)
                return None

        except Exception as e:
            self.logger.error(f"Get relation failed: {e}")
            return None

    def delete_relation(self, relation_id: int) -> bool:
        """
        Delete relation

        Args:
            relation_id: Relation ID

        Returns:
            Whether successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM knowledge_relations
                    WHERE id = ?
                """, (relation_id,))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Relation deleted: ID={relation_id}")

                return success

        except Exception as e:
            self.logger.error(f"Relation deletion failed: {e}")
            return False

    def get_node_relations(
        self,
        node_id: int,
        relation_type: Optional[RelationType] = None,
        direction: str = 'both'
    ) -> List[Tuple[GraphRelation, GraphNode]]:
        """
        Get node relations

        Args:
            node_id: Node ID
            relation_type: Relation type filter
            direction: Direction ('outgoing', 'incoming', 'both')

        Returns:
            List of (relation, related node)
        """
        try:
            with self.db_manager.get_connection() as conn:
                conditions = []
                params = []

                if direction in ['outgoing', 'both']:
                    conditions.append("source_memory_id = ?")
                    params.append(node_id)

                if direction in ['incoming', 'both']:
                    if conditions:
                        conditions.append("target_memory_id = ?")
                    else:
                        conditions.append("target_memory_id = ?")
                    params.append(node_id)

                if relation_type:
                    conditions.append("relation_type = ?")
                    params.append(relation_type.value)

                where_clause = " OR ".join(conditions) if direction == 'both' and len(conditions) > 1 else " AND ".join(conditions)

                cursor = conn.execute(f"""
                    SELECT * FROM knowledge_relations
                    WHERE {where_clause}
                    ORDER BY weight DESC
                """, params)

                results = []
                for row in cursor.fetchall():
                    relation = self._row_to_relation(row)

                    # Get related node
                    related_node_id = (
                        relation.target_node_id if relation.source_node_id == node_id
                        else relation.source_node_id
                    )
                    related_node = self.get_node(related_node_id)

                    if related_node:
                        results.append((relation, related_node))

                return results

        except Exception as e:
            self.logger.error(f"Get node relations failed: {e}")
            return []

    # ==================== Helper Methods ====================

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

        graph_metadata = None
        if row['graph_metadata']:
            try:
                graph_metadata = json.loads(row['graph_metadata'])
            except json.JSONDecodeError:
                graph_metadata = {}

        memory = Memory(
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

        # Add knowledge graph specific fields
        memory.graph_node_type = row['graph_node_type']
        memory.graph_metadata = graph_metadata
        memory.file_hash = row['file_hash']

        return memory

    def _row_to_relation(self, row) -> GraphRelation:
        """
        Convert database row to GraphRelation object

        Args:
            row: Database row

        Returns:
            GraphRelation object
        """
        metadata = None
        if row['metadata']:
            try:
                metadata = json.loads(row['metadata'])
            except json.JSONDecodeError:
                metadata = {}

        try:
            relation_type = RelationType(row['relation_type'])
        except ValueError:
            relation_type = RelationType.USES

        return GraphRelation(
            relation_id=row['id'],
            source_node_id=row['source_memory_id'],
            target_node_id=row['target_memory_id'],
            relation_type=relation_type,
            weight=row['weight'],
            metadata=metadata,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        )

    @staticmethod
    def calculate_file_hash(content: str) -> str:
        """
        Calculate file content hash

        Args:
            content: File content

        Returns:
            SHA256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
