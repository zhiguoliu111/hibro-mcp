#!/usr/bin/env python3
"""hibro Claude Code MCP Configuration Script"""

import json
import sys
from pathlib import Path


def main():
    # Use sys.executable to get the actual Python path currently running
    python_path = sys.executable

    # Configure mcpServers in BOTH ~/.claude.json AND ~/.claude/settings.json
    # Claude Code may read from either location depending on version

    # 1. Configure ~/.claude.json (legacy location)
    config_path = Path.home() / ".claude.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["hibro"] = {
        "command": python_path,
        "args": ["-m", "hibro.mcp.server"],
        "env": {}
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 2. Also configure ~/.claude/settings.json (primary location for Claude Code)
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = {}

    # Add mcpServers configuration
    if "mcpServers" not in settings:
        settings["mcpServers"] = {}

    settings["mcpServers"]["hibro"] = {
        "command": python_path,
        "args": ["-m", "hibro.mcp.server"],
        "env": {}
    }

    # Complete list of hibro MCP tools (updated with knowledge graph tools)
    hibro_tools = [
        # Core Tools
        "mcp__hibro__get_preferences",
        "mcp__hibro__get_quick_context",
        "mcp__hibro__search_memories",
        "mcp__hibro__remember",
        "mcp__hibro__forget",
        "mcp__hibro__update_memory",
        "mcp__hibro__get_status",

        # Intelligent Analysis Tools
        "mcp__hibro__analyze_conversation",
        "mcp__hibro__search_semantic",

        # Reasoning Engine Tools
        "mcp__hibro__analyze_causal_relations",
        "mcp__hibro__predict_next_needs",
        "mcp__hibro__build_knowledge_graph",
        "mcp__hibro__analyze_project_deeply",
        "mcp__hibro__answer_specific_question",

        # Adaptive Learning Tools
        "mcp__hibro__track_user_behavior",
        "mcp__hibro__get_personalized_recommendations",
        "mcp__hibro__analyze_user_patterns",
        "mcp__hibro__adaptive_importance_scoring",
        "mcp__hibro__get_learning_insights",

        # Context Tools
        "mcp__hibro__get_project_context",
        "mcp__hibro__get_recent_decisions",
        "mcp__hibro__get_important_facts",

        # Project Context Tools
        "mcp__hibro__set_project_context",
        "mcp__hibro__set_active_task",
        "mcp__hibro__complete_active_task",

        # Intelligent Assistant Tools
        "mcp__hibro__get_smart_suggestions",
        "mcp__hibro__detect_workflow_patterns",
        "mcp__hibro__get_workflow_recommendations",
        "mcp__hibro__execute_workflow",
        "mcp__hibro__get_intelligent_reminders",
        "mcp__hibro__get_comprehensive_assistance",
        "mcp__hibro__get_assistant_statistics",

        # Security Monitoring Tools
        "mcp__hibro__check_security_status",
        "mcp__hibro__apply_security_policy",
        "mcp__hibro__get_security_events",
        "mcp__hibro__resolve_security_event",

        # Backup & Sync Tools
        "mcp__hibro__create_backup",
        "mcp__hibro__restore_backup",
        "mcp__hibro__get_backup_statistics",
        "mcp__hibro__register_sync_device",
        "mcp__hibro__start_device_migration",

        # System Health Tools
        "mcp__hibro__get_system_health",
        "mcp__hibro__perform_security_scan",

        # Intelligent Guidance Tools
        "mcp__hibro__create_user_session",
        "mcp__hibro__get_tool_recommendations",
        "mcp__hibro__get_usage_hints",
        "mcp__hibro__get_learning_paths",
        "mcp__hibro__start_learning_path",
        "mcp__hibro__complete_learning_step",
        "mcp__hibro__get_guidance_statistics",

        # Project Scanning Tools
        "mcp__hibro__scan_project",
        "mcp__hibro__get_project_progress",
        "mcp__hibro__update_project_status",

        # Knowledge Graph Tools
        "mcp__hibro__init_knowledge_graph",
        "mcp__hibro__update_knowledge_graph",
        "mcp__hibro__query_knowledge_graph",
        "mcp__hibro__get_project_summary",

        # Code Knowledge Graph Tools (New)
        "mcp__hibro__init_code_knowledge_graph",
        "mcp__hibro__get_code_context",

        # Memory Refresh Tool (New)
        "mcp__hibro__refresh_memory",

        # Event & Sync Tools
        "mcp__hibro__get_sync_status",
        "mcp__hibro__get_event_bus_status",
        "mcp__hibro__list_event_subscribers",
    ]

    # Add permissions for hibro tools
    if "permissions" not in settings:
        settings["permissions"] = {}
    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []

    existing = set(settings["permissions"]["allow"])
    for tool in hibro_tools:
        if tool not in existing:
            settings["permissions"]["allow"].append(tool)

    # Update SessionStart hook with new hibro reminder
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Simple hook command for SessionStart
    session_hook = {
        "matcher": "startup|resume",
        "hooks": [
            {
                "type": "command",
                "command": "echo '[MANDATORY] must call mcp__hibro__get_quick_context first. If project_init missing, must call mcp__hibro__scan_project(quick_scan=false)'"
            }
        ]
    }

    # Remove old hibro hooks first
    if "SessionStart" in settings["hooks"]:
        settings["hooks"]["SessionStart"] = [
            hc for hc in settings["hooks"]["SessionStart"]
            if not any("hibro" in h.get("command", "") for h in hc.get("hooks", []))
        ]

    if "SessionStart" not in settings["hooks"]:
        settings["hooks"]["SessionStart"] = []

    # Add new hibro hook
    settings["hooks"]["SessionStart"].append(session_hook)

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    print("  [OK] Claude Code MCP configured")
    print(f"  [OK] Added {len(hibro_tools)} hibro tool permissions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
