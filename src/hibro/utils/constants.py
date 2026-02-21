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

# ==================== Success Messages ====================

SUCCESS_MESSAGES = {
    'memory_created': 'Memory created successfully: ID {}',
    'memory_updated': 'Memory updated successfully: ID {}',
    'memory_deleted': 'Memory deleted successfully: ID {}',
    'lfu_updated': 'LFU counter updated successfully',
    'hot_memories_calculated': 'Hot data calculation completed: {} memories',
}