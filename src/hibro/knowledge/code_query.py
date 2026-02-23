#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge graph query module

Provides optimized query interfaces for:
- Quick project context loading
- Natural language code queries
- Progressive context loading
- Token-efficient summaries
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .graph_storage import GraphStorage, GraphNode, GraphNodeType, RelationType
from .summary_generator import SummaryGenerator
from .session_update import SessionUpdateManager


@dataclass
class QueryResult:
    """Result of a knowledge graph query"""
    success: bool
    data: Any
    query_time_ms: float
    token_estimate: int
    error: Optional[str] = None


@dataclass
class ProjectContext:
    """Lightweight project context for quick loading"""
    project_path: str
    last_updated: Optional[datetime] = None

    # Statistics
    file_count: int = 0
    class_count: int = 0
    function_count: int = 0
    api_endpoint_count: int = 0

    # Core info
    core_modules: List[Dict[str, Any]] = None
    key_classes: List[str] = None
    recent_changes: List[Dict[str, Any]] = None

    # Token estimate
    token_estimate: int = 0

    def __post_init__(self):
        if self.core_modules is None:
            self.core_modules = []
        if self.key_classes is None:
            self.key_classes = []
        if self.recent_changes is None:
            self.recent_changes = []


class KnowledgeGraphQuery:
    """
    Optimized query interface for knowledge graph

    Provides fast, token-efficient queries for IDE integration.
    """

    # Token estimation constants (approximate)
    TOKENS_PER_CHAR = 0.25  # ~4 chars per token
    MAX_TOKENS_LIGHTWEIGHT = 500
    MAX_TOKENS_MEDIUM = 2000
    MAX_TOKENS_DETAILED = 5000

    def __init__(self, storage: GraphStorage):
        """
        Initialize query interface

        Args:
            storage: GraphStorage instance
        """
        self.storage = storage
        self.summary_generator = SummaryGenerator(storage)
        self.logger = logging.getLogger('hibro.kg_query')

        # Query cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    def get_quick_project_context(
        self,
        project_path: str,
        max_tokens: int = MAX_TOKENS_LIGHTWEIGHT
    ) -> QueryResult:
        """
        Get quick project context for session start

        This is the main entry point for loading project context
        at the beginning of a conversation.

        Args:
            project_path: Project path
            max_tokens: Maximum tokens to use

        Returns:
            QueryResult with ProjectContext
        """
        start_time = time.time()

        try:
            # Check cache
            cache_key = f"quick_context:{project_path}:{max_tokens}"
            cached = self._get_cached(cache_key)
            if cached:
                return QueryResult(
                    success=True,
                    data=cached,
                    query_time_ms=(time.time() - start_time) * 1000,
                    token_estimate=self._estimate_tokens(cached)
                )

            # Generate lightweight summary
            summary = self.summary_generator.generate_lightweight_summary(
                project_path,
                max_tokens=max_tokens
            )

            if not summary:
                return QueryResult(
                    success=False,
                    data=None,
                    query_time_ms=(time.time() - start_time) * 1000,
                    token_estimate=0,
                    error="Failed to generate summary"
                )

            # Build context
            stats = summary.get('statistics', {})
            context = ProjectContext(
                project_path=project_path,
                last_updated=datetime.now(),
                file_count=stats.get('total_files', 0),
                class_count=stats.get('total_classes', 0),
                function_count=stats.get('total_functions', 0),
                api_endpoint_count=stats.get('total_api_endpoints', 0),
                core_modules=summary.get('core_modules', []),
                key_classes=summary.get('key_classes', []),
                recent_changes=summary.get('recent_changes', [])
            )

            # Estimate tokens
            context.token_estimate = self._estimate_tokens(summary)

            # Cache result
            self._set_cached(cache_key, context)

            return QueryResult(
                success=True,
                data=context,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=context.token_estimate
            )

        except Exception as e:
            self.logger.error(f"Quick context query failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def get_detailed_context(
        self,
        project_path: str,
        max_tokens: int = MAX_TOKENS_MEDIUM
    ) -> QueryResult:
        """
        Get detailed project context with class/function info

        Args:
            project_path: Project path
            max_tokens: Maximum tokens to use

        Returns:
            QueryResult with detailed summary dict
        """
        start_time = time.time()

        try:
            # Check cache
            cache_key = f"detailed_context:{project_path}:{max_tokens}"
            cached = self._get_cached(cache_key)
            if cached:
                return QueryResult(
                    success=True,
                    data=cached,
                    query_time_ms=(time.time() - start_time) * 1000,
                    token_estimate=self._estimate_tokens(cached)
                )

            # Generate medium summary
            summary = self.summary_generator.generate_medium_summary(
                project_path,
                max_tokens=max_tokens
            )

            if not summary:
                return QueryResult(
                    success=False,
                    data=None,
                    query_time_ms=(time.time() - start_time) * 1000,
                    token_estimate=0,
                    error="Failed to generate summary"
                )

            token_estimate = self._estimate_tokens(summary)

            # Cache result
            self._set_cached(cache_key, summary)

            return QueryResult(
                success=True,
                data=summary,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=token_estimate
            )

        except Exception as e:
            self.logger.error(f"Detailed context query failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def search_code(
        self,
        project_path: str,
        query: str,
        search_type: str = 'keyword'
    ) -> QueryResult:
        """
        Search for code entities by name or description

        Args:
            project_path: Project path
            query: Search query
            search_type: 'keyword' or 'semantic'

        Returns:
            QueryResult with list of matching nodes
        """
        start_time = time.time()

        try:
            results = []

            # Search by name pattern
            nodes = self.storage.search_nodes(
                project_path=project_path,
                name_pattern=query,
                limit=20
            )

            for node in nodes:
                result = {
                    'name': node.name,
                    'type': node.node_type.value,
                    'file': node.file_path,
                    'line': node.line_number,
                    'importance': node.importance
                }

                # Add type-specific info
                if node.node_type == GraphNodeType.CLASS:
                    result['methods'] = node.metadata.get('methods', [])
                elif node.node_type == GraphNodeType.FUNCTION:
                    result['return_type'] = node.metadata.get('return_type')
                elif node.node_type == GraphNodeType.API_ENDPOINT:
                    result['method'] = node.metadata.get('method')
                    result['path'] = node.metadata.get('path')

                results.append(result)

            token_estimate = self._estimate_tokens(results)

            return QueryResult(
                success=True,
                data=results,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=token_estimate
            )

        except Exception as e:
            self.logger.error(f"Code search failed: {e}")
            return QueryResult(
                success=False,
                data=[],
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def get_class_hierarchy(
        self,
        project_path: str,
        class_name: str
    ) -> QueryResult:
        """
        Get inheritance hierarchy for a class

        Args:
            project_path: Project path
            class_name: Class name

        Returns:
            QueryResult with hierarchy info
        """
        start_time = time.time()

        try:
            # Find the class node
            nodes = self.storage.search_nodes(
                project_path=project_path,
                node_type=GraphNodeType.CLASS,
                name_pattern=class_name,
                limit=1
            )

            if not nodes:
                return QueryResult(
                    success=False,
                    data=None,
                    query_time_ms=(time.time() - start_time) * 1000,
                    token_estimate=0,
                    error=f"Class not found: {class_name}"
                )

            class_node = nodes[0]
            hierarchy = {
                'class_name': class_node.name,
                'file': class_node.file_path,
                'line': class_node.line_number,
                'parent': class_node.metadata.get('bases', []),
                'methods': class_node.metadata.get('methods', []),
                'children': [],
                'siblings': []
            }

            # Find children (classes that inherit from this class)
            # This requires scanning all class nodes
            all_classes = self.storage.search_nodes(
                project_path=project_path,
                node_type=GraphNodeType.CLASS,
                limit=1000
            )

            for other_class in all_classes:
                bases = other_class.metadata.get('bases', [])
                if class_name in bases:
                    hierarchy['children'].append({
                        'name': other_class.name,
                        'file': other_class.file_path
                    })

            token_estimate = self._estimate_tokens(hierarchy)

            return QueryResult(
                success=True,
                data=hierarchy,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=token_estimate
            )

        except Exception as e:
            self.logger.error(f"Class hierarchy query failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def get_file_structure(
        self,
        project_path: str,
        file_path: str
    ) -> QueryResult:
        """
        Get structure of a specific file

        Args:
            project_path: Project path
            file_path: File path within project

        Returns:
            QueryResult with file structure
        """
        start_time = time.time()

        try:
            # Get all nodes in file
            nodes = self.storage.get_nodes_by_file(file_path, project_path)

            structure = {
                'file_path': file_path,
                'classes': [],
                'functions': [],
                'api_endpoints': []
            }

            for node in nodes:
                if node.node_type == GraphNodeType.CLASS:
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.line_number,
                        'methods': node.metadata.get('methods', []),
                        'bases': node.metadata.get('bases', [])
                    })
                elif node.node_type == GraphNodeType.FUNCTION:
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.line_number,
                        'is_async': node.metadata.get('is_async', False)
                    })
                elif node.node_type == GraphNodeType.API_ENDPOINT:
                    structure['api_endpoints'].append({
                        'name': node.name,
                        'line': node.line_number,
                        'method': node.metadata.get('method'),
                        'path': node.metadata.get('path')
                    })

            # Sort by line number
            structure['classes'].sort(key=lambda x: x.get('line', 0))
            structure['functions'].sort(key=lambda x: x.get('line', 0))
            structure['api_endpoints'].sort(key=lambda x: x.get('line', 0))

            token_estimate = self._estimate_tokens(structure)

            return QueryResult(
                success=True,
                data=structure,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=token_estimate
            )

        except Exception as e:
            self.logger.error(f"File structure query failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def get_api_endpoints(
        self,
        project_path: str,
        method: Optional[str] = None
    ) -> QueryResult:
        """
        Get all API endpoints in project

        Args:
            project_path: Project path
            method: Filter by HTTP method (optional)

        Returns:
            QueryResult with list of endpoints
        """
        start_time = time.time()

        try:
            nodes = self.storage.search_nodes(
                project_path=project_path,
                node_type=GraphNodeType.API_ENDPOINT,
                limit=500
            )

            endpoints = []
            for node in nodes:
                endpoint = {
                    'name': node.name,
                    'file': node.file_path,
                    'line': node.line_number,
                    'method': node.metadata.get('method'),
                    'path': node.metadata.get('path')
                }

                if method is None or endpoint.get('method') == method.upper():
                    endpoints.append(endpoint)

            token_estimate = self._estimate_tokens(endpoints)

            return QueryResult(
                success=True,
                data=endpoints,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=token_estimate
            )

        except Exception as e:
            self.logger.error(f"API endpoints query failed: {e}")
            return QueryResult(
                success=False,
                data=[],
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def get_progressive_context(
        self,
        project_path: str,
        current_tokens: int,
        max_tokens: int
    ) -> QueryResult:
        """
        Get progressively more context based on token budget

        Args:
            project_path: Project path
            current_tokens: Tokens already used
            max_tokens: Maximum tokens available

        Returns:
            QueryResult with appropriate level of context
        """
        start_time = time.time()

        try:
            available_tokens = max_tokens - current_tokens

            if available_tokens < 100:
                # Very limited budget - just return stats
                data = self._get_minimal_context(project_path)
            elif available_tokens < self.MAX_TOKENS_LIGHTWEIGHT:
                # Limited budget - lightweight summary
                result = self.get_quick_project_context(
                    project_path,
                    max_tokens=min(available_tokens, 300)
                )
                data = result.data
            elif available_tokens < self.MAX_TOKENS_MEDIUM:
                # Medium budget - medium summary
                result = self.get_detailed_context(
                    project_path,
                    max_tokens=min(available_tokens, 1500)
                )
                data = result.data
            else:
                # Full budget - detailed context
                result = self.get_detailed_context(
                    project_path,
                    max_tokens=available_tokens
                )
                data = result.data

            return QueryResult(
                success=True,
                data=data,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=self._estimate_tokens(data)
            )

        except Exception as e:
            self.logger.error(f"Progressive context query failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                query_time_ms=(time.time() - start_time) * 1000,
                token_estimate=0,
                error=str(e)
            )

    def _get_minimal_context(self, project_path: str) -> Dict[str, Any]:
        """Get minimal context (< 100 tokens)"""
        # Count nodes by type
        counts = {}
        for node_type in GraphNodeType:
            nodes = self.storage.search_nodes(
                project_path=project_path,
                node_type=node_type,
                limit=1000
            )
            counts[node_type.value] = len(nodes)

        return {
            'project_path': project_path,
            'counts': counts
        }

    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count for data"""
        import json
        try:
            text = json.dumps(data, ensure_ascii=False)
            return int(len(text) * self.TOKENS_PER_CHAR)
        except:
            return 0

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value"""
        self._cache[key] = (value, time.time())

    def clear_cache(self):
        """Clear query cache"""
        self._cache.clear()
        self.logger.info("Query cache cleared")


def create_optimized_get_quick_context(
    db_manager,
    project_path: str
) -> Tuple[Dict[str, Any], KnowledgeGraphQuery]:
    """
    Factory function to create optimized get_quick_context

    This function integrates with the existing MCP tool system.

    Args:
        db_manager: Database manager
        project_path: Project path

    Returns:
        Tuple of (context_dict, query_instance)
    """
    from ..storage.migration_manager import MigrationManager
    from .graph_storage import GraphStorage

    # Ensure migration is done
    migration_manager = MigrationManager(db_manager)
    migration_manager.migrate()

    # Create storage and query
    storage = GraphStorage(db_manager)
    query = KnowledgeGraphQuery(storage)

    # Get quick context
    result = query.get_quick_project_context(project_path)

    if result.success and result.data:
        context = {
            'project_path': project_path,
            'statistics': {
                'files': result.data.file_count,
                'classes': result.data.class_count,
                'functions': result.data.function_count,
                'api_endpoints': result.data.api_endpoint_count
            },
            'core_modules': result.data.core_modules,
            'key_classes': result.data.key_classes,
            'recent_changes': result.data.recent_changes,
            'query_time_ms': result.query_time_ms,
            'token_estimate': result.token_estimate,
            'source': 'knowledge_graph'
        }
    else:
        context = {
            'project_path': project_path,
            'error': result.error,
            'source': 'knowledge_graph'
        }

    return context, query
