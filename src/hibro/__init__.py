#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hibro - Intelligent Memory System
Intelligent memory assistant that solves AI assistant context loss after restart

Version: 2.0.0
Author: hibro Development Team
"""

__version__ = "2.0.0"
__author__ = "hibro Development Team"
__email__ = "dev@hibro.com"
__description__ = "Intelligent Memory System - Solves AI assistant context loss issue"

# Export main classes and functions
from .core.memory_engine import MemoryEngine

__all__ = [
    "MemoryEngine",
]