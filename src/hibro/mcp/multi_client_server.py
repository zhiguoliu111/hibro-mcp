"""
Multi-Client MCP Server
MCP server implementation supporting multiple IDE clients connecting simultaneously
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, Resource, TextContent, Prompt, GetPromptResult, PromptMessage

from ..core.memory_engine import MemoryEngine
from ..utils.config import Config
from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class MultiClientMCPServer:
    """Multi-client MCP Server"""

    def __init__(self, config: Config):
        self.config = config
        self.memory_engine = MemoryEngine(config)
        self.connection_manager = ConnectionManager()

        # Create independent MCP server instance for each connection
        self.client_servers: Dict[str, Server] = {}

        # Shared tool and resource definitions
        self._tools = self._create_tools()
        self._resources = self._create_resources()
        self._prompts = self._create_prompts()

    async def handle_connection(self, connection_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection"""
        try:
            logger.info(f"Starting to handle client connection: {connection_id}")

            # Create MCP server instance for this connection
            server = Server(f"hibro-{connection_id[:8]}")
            self.client_servers[connection_id] = server

            # Register handlers
            self._register_handlers(server, connection_id)

            # Handle MCP protocol
            await server.run(reader, writer, server.create_initialization_options())

        except Exception as e:
            logger.error(f"Failed to handle client connection {connection_id}: {e}")
        finally:
            # Cleanup
            if connection_id in self.client_servers:
                del self.client_servers[connection_id]

    def _register_handlers(self, server: Server, connection_id: str):
        """Register handlers for MCP server"""

        @server.list_tools()
        async def list_tools() -> List[Tool]:
            return self._tools

        @server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            return await self._handle_tool_call(connection_id, name, arguments)

        @server.list_resources()
        async def list_resources() -> List[Resource]:
            return self._resources

        @server.read_resource()
        async def read_resource(uri: str) -> str:
            return await self._handle_resource_read(connection_id, uri)

        @server.list_prompts()
        async def list_prompts():
            return self._prompts

        @server.get_prompt()
        async def get_prompt(name: str, arguments: dict):
            return await self._handle_get_prompt(connection_id, name, arguments)

    def _create_tools(self) -> List[Tool]:
        """Create tool definitions (copied from original server.py)"""
        # Contains all original tool definitions
        # For simplicity, creating a few core tools first
        return [
            Tool(
                name="get_quick_context",
                description="Get user context information - must be called first in each session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_depth": {
                            "type": "string",
                            "enum": ["basic", "detailed", "comprehensive"],
                            "default": "detailed"
                        },
                        "include_project_context": {
                            "type": "boolean",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="remember",
                description="Store new memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["preference", "decision", "project", "important", "learning", "conversation"],
                            "default": "conversation"
                        },
                        "importance": {"type": "number", "default": 0.5},
                        "category": {"type": "string"}
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="search_memories",
                description="Search memories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "type": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_preferences",
                description="Get user preference settings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["code", "tool", "workflow", "project"]
                        },
                        "include_examples": {"type": "boolean", "default": False}
                    }
                }
            )
        ]

    def _create_resources(self) -> List[Resource]:
        """Create resource definitions"""
        return [
            Resource(
                uri="hibro://system-prompt",
                name="hibro Behavior Guide",
                description="READ FIRST: Smart memory behavior guidelines",
                mimeType="text/markdown"
            ),
            Resource(
                uri="hibro://quick-context",
                name="Quick Context",
                description="Essential context: preferences, decisions, important facts",
                mimeType="application/json"
            ),
            Resource(
                uri="hibro://status",
                name="System Status",
                description="System statistics",
                mimeType="application/json"
            )
        ]

    def _create_prompts(self) -> List[Prompt]:
        """Create prompt definitions"""
        return [
            Prompt(
                name="system_instructions",
                description="hibro Smart Memory System Behavior Guide"
            )
        ]

    async def _handle_tool_call(self, connection_id: str, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls"""
        try:
            logger.info(f"Client {connection_id} calling tool: {name}")

            # Get client information
            connection = self.connection_manager.get_connection(connection_id)
            client_context = {
                'connection_id': connection_id,
                'client_name': connection.get_client_name() if connection else 'Unknown',
                'client_type': connection.get_client_type() if connection else 'unknown'
            }

            # Dispatch handling based on tool name
            if name == "get_quick_context":
                result = await self._handle_get_quick_context(client_context, arguments)
            elif name == "remember":
                result = await self._handle_remember(client_context, arguments)
            elif name == "search_memories":
                result = await self._handle_search_memories(client_context, arguments)
            elif name == "get_preferences":
                result = await self._handle_get_preferences(client_context, arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}

            # Broadcast tool call event
            await self.connection_manager.broadcast_event(
                'tool_called',
                {
                    'tool_name': name,
                    'arguments': arguments,
                    'client_context': client_context,
                    'result_summary': str(result)[:200]
                },
                exclude_connection=connection_id
            )

            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]

        except Exception as e:
            logger.error(f"Tool call failed {name}: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )]

    async def _handle_get_quick_context(self, client_context: Dict, arguments: Dict) -> Dict:
        """Handle get quick context"""
        try:
            context_depth = arguments.get('context_depth', 'detailed')
            include_project = arguments.get('include_project_context', True)

            # Get user preferences
            preferences = await self.memory_engine.get_preferences()

            # Get recent decisions
            recent_decisions = await self.memory_engine.get_recent_decisions(limit=5)

            # Get important facts
            important_facts = await self.memory_engine.get_important_facts(limit=5)

            result = {
                'success': True,
                'preferences': [
                    {
                        'content': pref.content,
                        'category': pref.category,
                        'lfu_score': getattr(pref, 'lfu_score', 0.0)
                    }
                    for pref in preferences
                ],
                'recent_decisions': [
                    {
                        'content': dec.content,
                        'importance': dec.importance,
                        'lfu_score': getattr(dec, 'lfu_score', 0.0)
                    }
                    for dec in recent_decisions
                ],
                'important_facts': [
                    {
                        'content': fact.content,
                        'importance': fact.importance,
                        'lfu_score': getattr(fact, 'lfu_score', 0.0)
                    }
                    for fact in important_facts
                ],
                'client_context': client_context,
                'cache_version': 0,
                'stats': {
                    'total_memories': await self.memory_engine.get_memory_count(),
                    'connection_count': len(self.connection_manager),
                    'active_clients': [
                        conn.get_client_name()
                        for conn in self.connection_manager.get_all_connections()
                    ]
                }
            }

            return result

        except Exception as e:
            logger.error(f"Failed to get quick context: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_remember(self, client_context: Dict, arguments: Dict) -> Dict:
        """Handle memory storage"""
        try:
            content = arguments['content']
            memory_type = arguments.get('type', 'conversation')
            importance = arguments.get('importance', 0.5)
            category = arguments.get('category', 'general')

            # Store memory
            memory_id = await self.memory_engine.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                category=category,
                metadata={'client': client_context['client_name']}
            )

            # Broadcast memory storage event
            await self.connection_manager.broadcast_event(
                'memory_stored',
                {
                    'memory_id': memory_id,
                    'content': content[:100] + '...' if len(content) > 100 else content,
                    'type': memory_type,
                    'client': client_context['client_name']
                }
            )

            return {
                'success': True,
                'memory_id': memory_id,
                'message': f'Memory stored (ID: {memory_id})'
            }

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_search_memories(self, client_context: Dict, arguments: Dict) -> Dict:
        """Handle memory search"""
        try:
            query = arguments['query']
            memory_type = arguments.get('type')
            limit = arguments.get('limit', 10)

            # Search memories
            memories = await self.memory_engine.search_memories(
                query=query,
                memory_type=memory_type,
                limit=limit
            )

            return {
                'success': True,
                'memories': [
                    {
                        'id': memory.id,
                        'content': memory.content,
                        'type': memory.type,
                        'importance': memory.importance,
                        'created_at': memory.created_at.isoformat() if memory.created_at else None
                    }
                    for memory in memories
                ],
                'count': len(memories)
            }

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_get_preferences(self, client_context: Dict, arguments: Dict) -> Dict:
        """Handle get preferences"""
        try:
            category = arguments.get('category')
            include_examples = arguments.get('include_examples', False)

            preferences = await self.memory_engine.get_preferences(category=category)

            return {
                'success': True,
                'preferences': [
                    {
                        'content': pref.content,
                        'category': pref.category,
                        'importance': pref.importance,
                        'lfu_score': getattr(pref, 'lfu_score', 0.0)
                    }
                    for pref in preferences
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_resource_read(self, connection_id: str, uri: str) -> str:
        """Handle resource read"""
        try:
            if uri == "hibro://system-prompt":
                return self._get_system_prompt()
            elif uri == "hibro://quick-context":
                context = await self._handle_get_quick_context(
                    {'connection_id': connection_id},
                    {'context_depth': 'detailed'}
                )
                return json.dumps(context, ensure_ascii=False, indent=2)
            elif uri == "hibro://status":
                stats = self.connection_manager.get_connection_stats()
                return json.dumps(stats, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})

        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            return json.dumps({"error": str(e)})

    async def _handle_get_prompt(self, connection_id: str, name: str, arguments: dict):
        """Handle get prompt"""
        if name == "system_instructions":
            return GetPromptResult(
                description="hibro Smart Memory System Behavior Guide",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=self._get_system_prompt()
                        )
                    )
                ]
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        return """# hibro Smart Memory System Behavior Guide

You are an assistant for the hibro smart memory system. Your responsibility is to help the AI assistant intelligently store and query user memories.

## Proactive Behavior Rules

### 1. When to Proactively Store Memories (remember)

**Automatically** call the remember tool when detecting the following patterns:

**User Preference Expressions:**
- "I like..." / "I'm used to..." / "My style is..."
- "I usually..." / "I prefer..." / "I want..."
- "Please from now on..." / "Remember..." / "Don't forget..."

**Technical Decisions:**
- "Decided to use..." / "Choose..." / "Adopt..."
- "The architecture is..." / "The tech stack is..."
- "The approach is set to..." / "Finally chose..."

**Important Information:**
- "Important..." / "Key..." / "Core..."
- "Must..." / "Definitely..." / "Forbidden..."

### 2. When to Proactively Query Memories (get_preferences / search_memories)

**Automatically** query relevant memories in the following scenarios:

**New Conversation Starts:**
- Call get_quick_context to get user context

**Code-Related Tasks:**
- Before writing code, query coding style preferences
- When making technical selections, query historical decisions
- During code review, query coding standard preferences

**Project-Related Tasks:**
- When switching projects, call get_project_context
- When discussing architecture, query project-related decisions

## Multi-Client Environment

The current system supports multiple IDE clients connecting simultaneously:
- Memories are shared across all clients
- Tool calls are broadcast to other clients
- Each client can see the complete user context

Please use hibro tools intelligently according to these guidelines."""

    async def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'active_connections': len(self.client_servers),
            'connection_stats': self.connection_manager.get_connection_stats(),
            'memory_stats': {
                'total_memories': await self.memory_engine.get_memory_count()
            },
            'uptime': datetime.now().isoformat()
        }