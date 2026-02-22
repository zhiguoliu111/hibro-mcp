#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants definition module
Defines various constants used in the hibro system
"""

from typing import Set

# ==================== LFU Algorithm Configuration ====================

# LFU decay configuration
LFU_DECAY_TIME = 1      # Decay period (minutes)
LFU_COUNTER_INIT = 5    # Initial counter value
LFU_PROB_BASE = 1.0     # Probability growth base
LFU_DECAY_RATE = 0.1    # Decay rate

# Hot data configuration
GLOBAL_HOT_TOP_N = 20   # Global hot data count
PROJECT_HOT_TOP_N = 10  # Hot data count per project
COLD_THRESHOLD = 0.5    # LFU counter below this value is considered cold data

# ==================== Memory Type Definitions ====================

# Global memory types
GLOBAL_MEMORY_TYPES: Set[str] = {
    'preference',      # User preferences
    'decision',        # Technical decisions
    'important',       # Important facts
    'learning',        # Learning content
    'conversation',    # Conversation records
}

# Project-level memory types
PROJECT_MEMORY_TYPES: Set[str] = {
    'project',               # Project snapshot (from scan_project)
    'project_architecture',  # Project architecture description
    'project_command',       # Project-specific commands
    'active_task'            # Current iteration goals
}

# All memory types
ALL_MEMORY_TYPES: Set[str] = GLOBAL_MEMORY_TYPES | PROJECT_MEMORY_TYPES

# ==================== Project Context Types ====================

# Project context types
PROJECT_CONTEXT_TYPES: Set[str] = {
    'architecture',  # Architecture description
    'command',       # Specific commands
    'task'          # Task related
}

# ==================== Importance Score Configuration ====================

# Default importance scores
DEFAULT_IMPORTANCE_SCORES = {
    'preference': 0.7,
    'decision': 0.8,
    'project': 0.6,
    'important': 0.9,
    'learning': 0.5,
    'conversation': 0.3,
    'project_architecture': 0.8,
    'project_command': 0.6,
    'active_task': 1.0,  # Active tasks have highest importance
}

# ==================== Database Configuration ====================

# Memory table fields
MEMORY_TABLE_FIELDS = [
    'id', 'content', 'memory_type', 'importance', 'created_at',
    'last_accessed', 'access_count', 'metadata', 'project_path'
]

# Metadata fields
METADATA_FIELDS = {
    'project_path',      # Project path
    'context_type',      # Context type
    'is_active',         # Whether active (only used by active_task)
    'lfu_counter',       # LFU counter
    'source',            # Source
    'tags',              # Tags
    'category'           # Category
}

# ==================== Performance Configuration ====================

# Batch processing size
BATCH_SIZE = 1000

# Cache configuration
CACHE_SIZE_MB = 100
CACHE_TTL_SECONDS = 300  # 5 minutes

# Query limits
MAX_SEARCH_RESULTS = 100
DEFAULT_SEARCH_LIMIT = 10

# ==================== Time Configuration ====================

# Time formats
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'

# Cleanup configuration
CLEANUP_INTERVAL_DAYS = 7
MAX_MEMORY_AGE_DAYS = 365

# ==================== Error Messages ====================

ERROR_MESSAGES = {
    'invalid_memory_type': 'Invalid memory type: {}',
    'invalid_importance': 'Importance score must be between 0.0 and 1.0',
    'invalid_project_path': 'Invalid project path: {}',
    'memory_not_found': 'Memory does not exist: ID {}',
    'database_error': 'Database operation failed: {}',
    'lfu_calculation_error': 'LFU calculation failed: {}',
}

# ==================== Memory Cleanup Configuration ====================

# Scheduled cleanup
CLEANUP_INTERVAL_HOURS = 24        # Cleanup interval (hours)
CLEANUP_TIME_OF_DAY = "03:00"      # Daily cleanup time

# Threshold triggers
MEMORY_THRESHOLD_WARNING = 0.7     # 70% - warning level
MEMORY_THRESHOLD_CLEANUP = 0.85    # 85% - trigger cleanup
MEMORY_THRESHOLD_CRITICAL = 0.95   # 95% - force cleanup (block storage)

# Cleanup target
CLEANUP_TARGET_RATIO = 0.7         # After cleanup, reduce to 70% capacity

# LFU eviction
LFU_CLEANUP_BOTTOM_PERCENT = 0.2   # Clean bottom 20% LFU-ranked memories

# Time-based expiry
TIME_EXPIRY_DAYS_DEFAULT = 365     # Default retention: 365 days
TIME_EXPIRY_DAYS_LOW_IMPORTANCE = 90  # Low importance retention: 90 days

# Importance-based eviction
IMPORTANCE_THRESHOLD_CLEANUP = 0.2 # Clean memories with importance < 0.2

# New memory protection period (days)
NEW_MEMORY_PROTECTION_DAYS = 30    # Don't cleanup memories created within 30 days

# ==================== Smart Trigger Configuration ====================

# Confidence thresholds for trigger execution
TRIGGER_HIGH_CONFIDENCE = 0.8      # Silent execution
TRIGGER_MEDIUM_CONFIDENCE = 0.5    # Return suggestions
TRIGGER_LOW_CONFIDENCE = 0.3       # No trigger

# Semantic similarity threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Minimum similarity to consider project-related

# ==================== Query Keywords for Smart Trigger ====================

# Project meta keywords -> trigger get_project_progress
QUERY_KEYWORDS_PROJECT_META: Set[str] = {
    "project", "progress", "status", "stats", "overview", "snapshot",
    "项目", "进度", "状态", "统计", "概况", "快照"
}

# Project scan keywords -> trigger scan_project
QUERY_KEYWORDS_PROJECT_SCAN: Set[str] = {
    "scan", "analyze", "structure", "tech stack", "technology",
    "扫描", "分析项目", "重新扫描", "项目结构", "技术栈"
}

# Memory store keywords -> trigger remember
QUERY_KEYWORDS_MEMORY_STORE: Set[str] = {
    "remember", "save", "store", "note",
    "记住", "记住这个", "记录", "保存", "存储"
}

# Memory query keywords -> trigger search_memories / search_semantic
QUERY_KEYWORDS_MEMORY_QUERY: Set[str] = {
    "previous", "last time", "history", "search", "find",
    "之前", "上次", "历史", "找", "搜索", "查询"
}

# Tech stack keywords -> may be project-related
QUERY_KEYWORDS_TECH_STACK: Set[str] = {
    "react", "vue", "angular", "fastapi", "django", "flask",
    "python", "typescript", "javascript", "java", "go", "rust",
    "docker", "kubernetes", "postgresql", "mysql", "redis",
    "api", "database", "frontend", "backend", "architecture",
    "数据库", "前端", "后端", "架构"
}

# Success Messages ====================

SUCCESS_MESSAGES = {
    'memory_created': 'Memory created successfully: ID {}',
    'memory_updated': 'Memory updated successfully: ID {}',
    'memory_deleted': 'Memory deleted successfully: ID {}',
    'lfu_updated': 'LFU counter updated successfully',
    'hot_memories_calculated': 'Hot data calculation completed: {} memories',
    'cleanup_completed': 'Cleanup completed: {} memories deleted',
    'trigger_executed': 'Smart trigger executed: {}',
}