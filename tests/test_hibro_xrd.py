#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test hibro with xrd-tool - Simulate MCP calls
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add hibro to path
hibro_path = Path(__file__).parent.parent
sys.path.insert(0, str(hibro_path / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

XRD_TOOL_PATH = r"D:\projects\xrd-tool"


async def test_hibro_with_xrd():
    """Test hibro workflow with xrd-tool"""

    logger.info("=" * 60)
    logger.info("Testing hibro with xrd-tool")
    logger.info("=" * 60)

    from hibro.mcp.server import MCPServer

    # Initialize server
    logger.info("\n1. Initializing hibro MCP Server...")
    server = MCPServer()
    logger.info("   Server initialized")

    # Clean up any existing xrd-tool memories
    logger.info("\n2. Cleaning up existing xrd-tool memories...")
    memories = server.memory_engine.memory_repo.search_memories(
        project_path=XRD_TOOL_PATH,
        limit=100
    )
    for m in memories:
        server.memory_engine.memory_repo.delete_memory(m.id)
    logger.info(f"   Deleted {len(memories)} memories")

    # Test get_quick_context (first time - should trigger initialization)
    logger.info("\n3. Calling get_quick_context (first time)...")
    result = await server._tool_get_quick_context({"project_path": XRD_TOOL_PATH})

    logger.info(f"   Success: {result.get('success')}")
    logger.info(f"   Project init: {result.get('project_init', {})}")

    workflow = result.get('workflow_overview', '')
    if workflow:
        logger.info("\n--- WORKFLOW OVERVIEW (first 2000 chars) ---")
        logger.info(workflow[:2000])
        logger.info("..." if len(workflow) > 2000 else "")

    # Test get_quick_context again (second time - should load from cache)
    logger.info("\n4. Calling get_quick_context (second time - should load from cache)...")
    result2 = await server._tool_get_quick_context({"project_path": XRD_TOOL_PATH})

    logger.info(f"   Success: {result2.get('success')}")
    project_init = result2.get('project_init', {})
    logger.info(f"   From cache: {project_init.get('from_cache', False)}")

    workflow2 = result2.get('workflow_overview', '')
    if workflow2:
        logger.info("   Workflow loaded from memory: YES")
    else:
        logger.info("   Workflow loaded from memory: NO (missing)")

    # Test get_code_context
    logger.info("\n5. Testing get_code_context...")
    code_result = await server._tool_get_code_context({
        "project_path": XRD_TOOL_PATH,
        "detail_level": "lightweight"
    })
    logger.info(f"   Success: {code_result.get('success')}")
    if code_result.get('success'):
        data = code_result.get('data', {})
        if hasattr(data, 'file_count'):
            logger.info(f"   Files: {data.file_count}")
            logger.info(f"   Classes: {data.class_count}")
            logger.info(f"   Functions: {data.function_count}")
            logger.info(f"   Token estimate: {code_result.get('token_estimate')}")

    # Summary
    logger.info("\n" + "=" * 60)
    if result.get('success') and result2.get('success') and code_result.get('success'):
        logger.info("ALL TESTS PASSED!")
    else:
        logger.info("SOME TESTS FAILED")
    logger.info("=" * 60)

    return result.get('success', False)


if __name__ == "__main__":
    success = asyncio.run(test_hibro_with_xrd())
    sys.exit(0 if success else 1)
