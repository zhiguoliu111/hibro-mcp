#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hibro MCP Server Module
Provides Model Context Protocol support, enabling Claude Code to automatically invoke hibro
"""

from .server import MCPServer, run_server

__all__ = ['MCPServer', 'run_server']
