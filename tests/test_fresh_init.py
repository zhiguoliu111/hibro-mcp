#!/usr/bin/env python3
"""Test get_quick_context with fresh database"""

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
    print(f"Testing get_quick_context for: {project_path}")
    print(f"{'='*60}\n")

    result = await server._tool_get_quick_context({
        'project_path': project_path,
        'context_depth': 'detailed'
    })

    # Check memory_status
    memory_status = result.get('memory_status', {})
    project_init = result.get('project_init', {})

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"\nmemory_status:")
    print(json.dumps(memory_status, indent=2, ensure_ascii=False))

    print(f"\nproject_init:")
    print(json.dumps(project_init, indent=2, ensure_ascii=False))

    # Verify
    print(f"\n{'='*60}")
    print(f"Verification:")
    print(f"{'='*60}")

    if memory_status.get('initialized_at') and memory_status['initialized_at'] != 'unknown':
        print(f"✅ initialized_at: {memory_status['initialized_at']}")
    else:
        print(f"❌ initialized_at: {memory_status.get('initialized_at', 'MISSING')}")

    if memory_status.get('cached_at') and memory_status['cached_at'] != 'unknown':
        print(f"✅ cached_at: {memory_status['cached_at']}")
    else:
        print(f"⚠️  cached_at: {memory_status.get('cached_at', 'MISSING')}")

    if project_init.get('is_first_time'):
        print(f"✅ is_first_time: {project_init['is_first_time']}")
    else:
        print(f"ℹ️  is_first_time: {project_init.get('is_first_time', False)}")

    # Expected display format
    print(f"\n{'='*60}")
    print(f"Expected Display Format:")
    print(f"{'='*60}")

    if project_init.get('is_first_time'):
        init_time = memory_status.get('initialized_at', memory_status.get('cached_at', 'unknown'))
        stars = memory_status.get('freshness_stars', '⭐⭐⭐⭐⭐')
        print(f"Memory Status: [{stars}] (just now) | Initialized: {init_time}")
    else:
        cache_time = memory_status.get('cached_at', 'unknown')
        stars = memory_status.get('freshness_stars', '❓')
        age = memory_status.get('age_human', 'unknown')
        print(f"Memory Status: [{stars}] ({age} ago) | Cached: {cache_time}")

if __name__ == "__main__":
    asyncio.run(test())
