#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions
Provides logging, validation, formatting and other common functionality
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import re


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Setup logging system

    Args:
        verbose: Whether to enable verbose logging mode

    Returns:
        Configured logger object
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    logger = logging.getLogger('hibro')
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File output (optional)
        log_dir = Path.home() / '.hibro' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f'hibro_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_importance_score(score: float) -> bool:
    """
    Validate if importance score is valid

    Args:
        score: Importance score

    Returns:
        Whether it's valid
    """
    return 0.0 <= score <= 1.0


def sanitize_content(content: str) -> str:
    """
    Clean and filter content, remove sensitive information

    Args:
        content: Original content

    Returns:
        Cleaned content
    """
    if not content:
        return ""

    # Sensitive information patterns
    sensitive_patterns = [
        r'password\s*[:=]\s*["\']?[^"\'\s]+["\']?',  # Password
        r'api[_-]?key\s*[:=]\s*["\']?[^"\'\s]+["\']?',  # API key
        r'token\s*[:=]\s*["\']?[^"\'\s]+["\']?',  # Token
        r'secret\s*[:=]\s*["\']?[^"\'\s]+["\']?',  # Secret
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card format
    ]

    cleaned_content = content
    for pattern in sensitive_patterns:
        cleaned_content = re.sub(pattern, '[REDACTED]', cleaned_content, flags=re.IGNORECASE)

    return cleaned_content


def format_memory_content(content: str, max_length: int = 200) -> str:
    """
    Format memory content for display

    Args:
        content: Original content
        max_length: Maximum display length

    Returns:
        Formatted content
    """
    if not content:
        return ""

    # Clean content
    cleaned = sanitize_content(content.strip())

    # Truncate overly long content
    if len(cleaned) > max_length:
        return cleaned[:max_length-3] + "..."

    return cleaned


def calculate_time_decay(days_since_access: int, decay_rate: float = 0.1) -> float:
    """
    Calculate time decay factor

    Args:
        days_since_access: Days since last access
        decay_rate: Decay rate

    Returns:
        Decay factor (0.0-1.0)
    """
    import math
    return math.exp(-decay_rate * days_since_access)


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in MB

    Args:
        file_path: File path

    Returns:
        File size in MB
    """
    try:
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0


def ensure_directory(dir_path: Path) -> bool:
    """
    Ensure directory exists

    Args:
        dir_path: Directory path

    Returns:
        Whether successfully created or already exists
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger = logging.getLogger('hibro')
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp

    Args:
        timestamp: Timestamp, if None use current time

    Returns:
        Formatted time string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def parse_memory_type(content: str) -> str:
    """
    Automatically identify memory type based on content

    Args:
        content: Memory content

    Returns:
        Memory type
    """
    content_lower = content.lower()

    # Preference type keywords
    preference_keywords = ['i like', 'i prefer', 'my style', 'i usually', 'my preference']
    if any(keyword in content_lower for keyword in preference_keywords):
        return 'preference'

    # Decision type keywords
    decision_keywords = ['decided to use', 'choose', 'adopt', 'architecture', 'tech selection']
    if any(keyword in content_lower for keyword in decision_keywords):
        return 'decision'

    # Project type keywords
    project_keywords = ['project', 'requirement', 'feature', 'module', 'development']
    if any(keyword in content_lower for keyword in project_keywords):
        return 'project'

    # Important information keywords
    important_keywords = ['important', 'key', 'core', 'note']
    if any(keyword in content_lower for keyword in important_keywords):
        return 'important'

    # Default to general conversation
    return 'conversation'


class ProgressTracker:
    """Progress tracker"""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.current_step = 0

    def add_step(self, name: str, description: str = ""):
        """Add step"""
        self.steps.append({
            'name': name,
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None
        })

    def start_step(self, step_index: int):
        """Start executing step"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'in_progress'
            self.steps[step_index]['start_time'] = datetime.now()
            self.current_step = step_index

    def complete_step(self, step_index: int):
        """Complete step"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['end_time'] = datetime.now()

    def get_progress_summary(self) -> str:
        """Get progress summary"""
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        total = len(self.steps)
        return f"Progress: {completed}/{total} ({completed/total*100:.1f}%)"