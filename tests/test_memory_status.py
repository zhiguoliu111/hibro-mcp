#!/usr/bin/env python3
"""Test script to verify get_quick_context returns initialized_at"""

import asyncio
import sys
import json
from pathlib import Path

# Add hibro to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test():
    from hibro.mcp.server import MCPServer
    from hibro.utils.config import Config

    config = Config()
    server = MCPServer(config)

    # Initialize server components
    from hibro.core.memory_engine import MemoryEngine
    from hibro.core.partition import MemoryPartition

    server.memory_engine = MemoryEngine(config)
    server.memory_engine.initialize()

    server.memory_partition = MemoryPartition(server.memory_engine.memory_repo)

    # Test get_quick_context
    result = await server._tool_get_quick_context({
        'project_path': 'D:/projects/black-spider',
        'context_depth': 'detailed'
    })

    print("=" * 60)
    print("memory_status:")
    print("=" * 60)
    print(json.dumps(result.get('memory_status', {}), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("project_init:")
    print("=" * 60)
    print(json.dumps(result.get('project_init', {}), indent=2, ensure_ascii=False))

    # Check if initialized_at is present
    memory_status = result.get('memory_status', {})
    if 'initialized_at' in memory_status:
        print(f"\n✅ initialized_at found: {memory_status['initialized_at']}")
    else:
        print("\n❌ initialized_at NOT found in memory_status")

    if 'cached_at' in memory_status:
        print(f"✅ cached_at found: {memory_status['cached_at']}")
    else:
        print("❌ cached_at NOT found in memory_status")

if __name__ == "__main__":
    asyncio.run(test())
