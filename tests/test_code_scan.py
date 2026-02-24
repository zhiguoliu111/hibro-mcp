#!/usr/bin/env python3
"""Test code scanning functionality"""

import asyncio
import sys
from pathlib import Path

# Add hibro to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_scan():
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

    project_path = 'D:/projects/black-spider'

    print(f"\n{'='*60}")
    print(f"Testing project scan")
    print(f"Project: {project_path}")
    print(f"{'='*60}\n")

    # Test scan_project
    print("1. Testing scan_project...")
    scan_result = await server._tool_scan_project({
        "project_path": project_path,
        "quick_scan": False
    })
    print(f"   Success: {scan_result.get('success')}")
    if scan_result.get('success'):
        print(f"   Project name: {scan_result.get('project_name')}")
        print(f"   Project type: {scan_result.get('project_type')}")
        print(f"   Summary: {scan_result.get('summary')[:100]}...")
        print(f"   Statistics: {scan_result.get('statistics')}")
    else:
        print(f"   Error: {scan_result.get('error')}")

    # Test init_code_knowledge_graph
    print("\n2. Testing init_code_knowledge_graph...")
    kg_result = await server._tool_init_code_knowledge_graph({
        "project_path": project_path
    })
    print(f"   Success: {kg_result.get('success')}")
    if kg_result.get('success'):
        stats = kg_result.get('statistics', {})
        print(f"   Files processed: {stats.get('files_processed', 0)}")
        print(f"   Classes added: {stats.get('classes_added', 0)}")
        print(f"   Functions added: {stats.get('functions_added', 0)}")
        print(f"   API endpoints added: {stats.get('api_endpoints_added', 0)}")
        print(f"   Errors: {stats.get('errors', 0)}")
    else:
        print(f"   Error: {kg_result.get('error')}")
        import traceback
        if kg_result.get('error'):
            print(f"   Error details: {kg_result.get('error')}")

    # Check knowledge graph storage
    print("\n3. Checking knowledge graph storage...")
    try:
        from hibro.knowledge.graph_storage import GraphStorage
        storage = GraphStorage(server.memory_engine.db_manager)
        nodes = storage.search_nodes(project_path=project_path, limit=10000)
        print(f"   Total nodes found: {len(nodes)}")

        if nodes:
            by_type = {}
            for node in nodes:
                node_type = node.node_type.value
                by_type[node_type] = by_type.get(node_type, 0) + 1

            print("   Nodes by type:")
            for node_type, count in by_type.items():
                print(f"     {node_type}: {count}")
    except Exception as e:
        print(f"   Error checking storage: {e}")

    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(test_scan())
