#!/usr/bin/env python3
"""Test hibro initialization with black-spider project"""

import asyncio
import sys
import json
from pathlib import Path

# Add hibro to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test():
    from hibro.mcp.server import MCPServer
    from hibro.utils.config import Config
    from hibro.core.memory_engine import MemoryEngine
    from hibro.core.partition import MemoryPartition

    config = Config()
    server = MCPServer(config)

    # Initialize server components
    server.memory_engine = MemoryEngine(config)
    server.memory_engine.initialize()
    server.memory_partition = MemoryPartition(server.memory_engine.memory_repo)

    # Test get_quick_context (should trigger auto-init)
    project_path = 'D:/projects/black-spider'
    print(f"\n{'='*60}")
    print(f"Testing hibro initialization")
    print(f"Project: {project_path}")
    print(f"{'='*60}\n")

    result = await server._tool_get_quick_context({
        'project_path': project_path,
        'context_depth': 'detailed'
    })

    # Check result
    print(f"Success: {result.get('success')}")

    project_init = result.get('project_init', {})
    print(f"\nproject_init:")
    print(f"  initialized: {project_init.get('initialized')}")
    print(f"  is_first_time: {project_init.get('is_first_time')}")

    memory_status = result.get('memory_status', {})
    print(f"\nmemory_status:")
    print(f"  initialized_at: {memory_status.get('initialized_at')}")
    print(f"  cached_at: {memory_status.get('cached_at')}")
    print(f"  freshness_stars: {memory_status.get('freshness_stars')}")
    print(f"  age_human: {memory_status.get('age_human')}")

    # Check display hint
    display_hint = result.get('_display_hint', '')
    if display_hint:
        print(f"\n{'='*60}")
        print("Display Output:")
        print(f"{'='*60}")
        print(display_hint)

    # Check knowledge graph stats
    kg_stats = project_init.get('knowledge_graph', {}).get('actual_statistics', {})
    if kg_stats:
        print(f"\n{'='*60}")
        print("Knowledge Graph Statistics:")
        print(f"{'='*60}")
        print(f"  Total nodes: {kg_stats.get('total_nodes', 0)}")
        print(f"  Files: {kg_stats.get('files_count', 0)}")
        print(f"  Classes: {kg_stats.get('classes_count', 0)}")
        print(f"  Functions: {kg_stats.get('functions_count', 0)}")
        print(f"  Modules: {kg_stats.get('modules_count', 0)}")
        print(f"  API endpoints: {kg_stats.get('api_endpoints_count', 0)}")

    print(f"\n{'='*60}")
    print("Test Result: ", "✅ PASS" if result.get('success') else "❌ FAIL")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(test())
