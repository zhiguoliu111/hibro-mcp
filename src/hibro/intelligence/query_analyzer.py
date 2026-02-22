#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Analyzer Module
Analyzes user queries to detect project-related content and suggest appropriate tools
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta

from ..utils.constants import (
    QUERY_KEYWORDS_PROJECT_META,
    QUERY_KEYWORDS_PROJECT_SCAN,
    QUERY_KEYWORDS_MEMORY_STORE,
    QUERY_KEYWORDS_MEMORY_QUERY,
    QUERY_KEYWORDS_TECH_STACK,
    SEMANTIC_SIMILARITY_THRESHOLD
)


class QueryKeywords:
    """Project-related keyword definitions for smart trigger"""

    # Project meta keywords -> trigger get_project_progress
    PROJECT_META = QUERY_KEYWORDS_PROJECT_META

    # Project scan keywords -> trigger scan_project
    PROJECT_SCAN = QUERY_KEYWORDS_PROJECT_SCAN

    # Memory store keywords -> trigger remember
    MEMORY_STORE = QUERY_KEYWORDS_MEMORY_STORE

    # Memory query keywords -> trigger search_memories / search_semantic
    MEMORY_QUERY = QUERY_KEYWORDS_MEMORY_QUERY

    # Tech stack keywords -> may be project-related
    TECH_STACK = QUERY_KEYWORDS_TECH_STACK

    # Project path patterns -> high relevance
    PROJECT_PATH_PATTERNS = [
        r"[A-Z]:\\[\w\\]+",      # Windows path
        r"/[\w/]+",               # Unix path
        r"~/[\w/]+",              # User home directory
    ]


class QueryAnalyzer:
    """
    User query intelligent analyzer

    Analyzes user messages to detect project-related content
    using keyword matching and semantic similarity.
    """

    def __init__(self, memory_engine=None, project_scanner=None):
        """
        Initialize query analyzer

        Args:
            memory_engine: Memory engine for semantic matching
            project_scanner: Project scanner for context
        """
        self.memory_engine = memory_engine
        self.project_scanner = project_scanner
        self.logger = logging.getLogger(__name__)

        # Cache for project context (used in semantic matching)
        self._project_context_cache: Dict[str, Dict] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)

    def analyze(self, query: str, project_path: str) -> Dict[str, Any]:
        """
        Analyze user query and return relevance and suggested actions

        Args:
            query: User's query text
            project_path: Current project path

        Returns:
            {
                "is_project_related": bool,    # Whether related to project
                "confidence": float,           # Confidence score 0-1
                "matched_keywords": list,      # Matched keyword categories
                "suggested_tools": list,       # Suggested tools to call
                "semantic_score": float,       # Semantic similarity score
            }
        """
        result = {
            "is_project_related": False,
            "confidence": 0.0,
            "matched_keywords": [],
            "suggested_tools": [],
            "semantic_score": 0.0
        }

        if not query or not query.strip():
            return result

        # Step 1: Keyword quick filtering
        keyword_result = self._match_keywords(query)
        result["matched_keywords"] = keyword_result["matched"]

        if not keyword_result["has_match"]:
            # No keyword match, try semantic matching
            semantic_score = self._semantic_match(query, project_path)
            result["semantic_score"] = semantic_score
            result["is_project_related"] = semantic_score > SEMANTIC_SIMILARITY_THRESHOLD
            result["confidence"] = semantic_score
        else:
            # Has keyword match
            result["is_project_related"] = True
            result["confidence"] = keyword_result["confidence"]
            result["suggested_tools"] = keyword_result["suggested_tools"]

        return result

    def _match_keywords(self, query: str) -> Dict[str, Any]:
        """
        Keyword matching

        Args:
            query: User query

        Returns:
            {
                "has_match": bool,
                "matched": list,           # Matched categories
                "suggested_tools": list,   # Suggested tools
                "confidence": float        # Confidence score
            }
        """
        query_lower = query.lower()
        matched = []
        suggested_tools = []
        confidence = 0.0

        # Check project meta keywords
        if any(kw in query_lower for kw in QueryKeywords.PROJECT_META):
            matched.append("project_meta")
            suggested_tools.append("get_project_progress")
            confidence = max(confidence, 0.8)

        # Check project scan keywords
        if any(kw in query_lower for kw in QueryKeywords.PROJECT_SCAN):
            matched.append("project_scan")
            suggested_tools.append("scan_project")
            confidence = max(confidence, 0.9)

        # Check memory store keywords
        if any(kw in query_lower for kw in QueryKeywords.MEMORY_STORE):
            matched.append("memory_store")
            suggested_tools.append("remember")
            confidence = max(confidence, 0.7)

        # Check memory query keywords
        if any(kw in query_lower for kw in QueryKeywords.MEMORY_QUERY):
            matched.append("memory_query")
            suggested_tools.append("search_semantic")
            confidence = max(confidence, 0.7)

        # Check tech stack keywords (lower confidence)
        if any(kw in query_lower for kw in QueryKeywords.TECH_STACK):
            matched.append("tech_stack")
            confidence = max(confidence, 0.5)

        # Check project path patterns (high confidence)
        for pattern in QueryKeywords.PROJECT_PATH_PATTERNS:
            if re.search(pattern, query):
                matched.append("project_path")
                confidence = max(confidence, 0.85)
                break

        return {
            "has_match": len(matched) > 0,
            "matched": matched,
            "suggested_tools": suggested_tools,
            "confidence": confidence
        }

    def _semantic_match(self, query: str, project_path: str) -> float:
        """
        Semantic similarity matching

        Uses embedding models to calculate similarity between
        query and project context.

        Args:
            query: User query
            project_path: Current project path

        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Get project context
        context = self._get_project_context(project_path)
        if not context:
            return 0.0

        # Build context text for comparison
        context_text = self._build_context_text(context)
        if not context_text:
            return 0.0

        # Calculate semantic similarity
        try:
            if self.memory_engine and hasattr(self.memory_engine, 'embedding_manager'):
                embedding_manager = self.memory_engine.embedding_manager
                if embedding_manager:
                    query_embedding = embedding_manager.encode(query)
                    context_embedding = embedding_manager.encode(context_text)

                    # Cosine similarity
                    similarity = self._cosine_similarity(query_embedding, context_embedding)
                    return float(similarity)
        except Exception as e:
            self.logger.warning(f"Semantic matching failed: {e}")

        return 0.0

    def _get_project_context(self, project_path: str) -> Optional[Dict]:
        """
        Get project context with caching

        Args:
            project_path: Project path

        Returns:
            Project context dict or None
        """
        now = datetime.now()

        # Check cache
        if project_path in self._project_context_cache:
            if project_path in self._cache_expiry:
                if now < self._cache_expiry[project_path]:
                    return self._project_context_cache[project_path]

        # Fetch fresh context
        try:
            if self.memory_engine:
                # Get project memories
                project_memories = self.memory_engine.memory_repo.search_memories(
                    query="",
                    project_path=project_path,
                    limit=20
                )

                context = {
                    "project_path": project_path,
                    "memories": project_memories[:20] if project_memories else []
                }

                # Update cache
                self._project_context_cache[project_path] = context
                self._cache_expiry[project_path] = now + self._cache_ttl

                return context
        except Exception as e:
            self.logger.warning(f"Failed to get project context: {e}")

        return None

    def _build_context_text(self, context: Dict) -> str:
        """
        Build text representation of project context for semantic matching

        Args:
            context: Project context dict

        Returns:
            Context text string
        """
        parts = []

        # Add project path
        if context.get("project_path"):
            parts.append(f"Project path: {context['project_path']}")

        # Add memory content
        memories = context.get("memories", [])
        for memory in memories[:10]:
            if hasattr(memory, 'content'):
                parts.append(memory.content)
            elif isinstance(memory, dict) and 'content' in memory:
                parts.append(memory['content'])

        return " ".join(parts)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0.0 - 1.0)
        """
        import numpy as np

        try:
            vec1 = np.array(vec1).flatten()
            vec2 = np.array(vec2).flatten()

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            # Clamp to [0, 1]
            return max(0.0, min(1.0, float(similarity)))
        except Exception:
            return 0.0

    def clear_cache(self):
        """Clear project context cache"""
        self._project_context_cache.clear()
        self._cache_expiry.clear()
