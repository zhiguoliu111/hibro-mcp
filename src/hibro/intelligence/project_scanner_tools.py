#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Tools: Project Scanning and Snapshots
Allows users to manually or automatically scan projects, generate project snapshots and store them in the memory system
"""

# The content of this file will be integrated into server.py
# This is just a design document

"""
New MCP Tool Design:

1. scan_project - Scan current project
   Purpose: Manually trigger project scanning
   Parameters:
   - project_path: Project path (optional, defaults to current directory)
   - quick_scan: Quick scan mode (default true)

2. get_project_progress - Get project progress
   Purpose: View current project status and progress
   Parameters:
   - project_path: Project path (optional)

   Automatic behavior:
   - If project snapshot is outdated (>7 days), automatically rescan
   - Combined with memory system, provides project history and current tasks

3. update_project_status - Update project status
   Purpose: Manually update project status
   Parameters:
   - project_path: Project path
   - status: Project status (planning, development, testing, production)
   - current_task: Current task description
   - progress_percentage: Progress percentage
"""

# Tool implementation will be added to server.py's _get_tools() and _handle_tool_call() methods
