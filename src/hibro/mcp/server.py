#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hibro MCP Server Main Module
Implemented using official MCP SDK, exposing tools and resources to IDEs
Supporting intelligent memory storage and querying, conversation analysis, semantic search
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Official MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    Resource,
    TextContent,
    ResourceTemplate,
    Prompt,
    GetPromptResult,
    PromptMessage,
)

# hibro core modules
from ..core.memory_engine import MemoryEngine
from ..core.partition import MemoryPartition
from ..core.active_task import ActiveTaskManager
from ..utils.config import Config
from ..intelligence.extractor import MemoryExtractor
from ..intelligence.similarity import SimilarityCalculator, SemanticSearchEngine


# System prompt content - Solution B
SYSTEM_PROMPT = """# hibro Intelligent Memory System Behavior Guide

You are an assistant for the hibro intelligent memory system. Your responsibility is to help the AI assistant intelligently store and query user memories.

## Proactive Behavior Rules

### 1. When to Proactively Store Memories (remember)

When detecting the following patterns, **automatically** call the remember tool to store memories:

**User Preference Expression:**
- "I like..." / "I'm used to..." / "My style is..."
- "I usually..." / "I prefer..." / "I hope..."
- "Please in the future..." / "Remember..." / "Don't forget..."

**Technical Decisions:**
- "Decided to use..." / "Choose..." / "Adopt..."
- "Architecture is..." / "Tech stack is..."
- "Solution decided as..." / "Finally chose..."

**Important Information:**
- "Important..." / "Key..." / "Core..."
- "Must..." / "Definitely..." / "Forbidden..."

### 2. When to Proactively Analyze Conversations (analyze_conversation)

**After each conversation ends**, proactively call the analyze_conversation tool:
- Extract valuable memory fragments
- Automatic classification and scoring
- Discover user preferences and decisions

### 3. When to Proactively Query Memories (get_preferences / search_memories)

**Automatically** query relevant memories in the following scenarios:

**New conversation starts:**
- Call get_preferences to get user preferences
- Call get_quick_context to get quick context

**Code-related tasks:**
- Before writing code, query coding style preferences
- When making technical choices, query historical decisions
- During code review, query coding standard preferences

**Answering technical questions:**
- When users ask about programming language standards/naming conventions, query coding preferences
- When users ask about "considerations"/"standards"/"best practices"/"styles", query relevant preferences
- For questions involving code style, naming conventions, coding habits, query coding preferences
- **Prioritize using get_preferences(category="code") over search_memories**
- Example: "What are the considerations for Java method names" → automatically call get_preferences(category="code")
- Example: "What are MySQL field naming conventions" → automatically call get_preferences(category="code")

**Project-related tasks:**
- When switching projects, call get_project_context
- When discussing architecture, query project-related decisions

### 4. Memory Type Selection

- `preference`: User preferences (code style, tool choices, etc.)
- `decision`: Technical decisions (architecture selection, solution determination, etc.)
- `project`: Project information (tech stack, dependencies, etc.)
- `important`: Important matters
- `learning`: Learning content
- `conversation`: General conversation

### 5. Importance Scoring

- User explicitly says "remember"/"important": 0.9
- Preference expression: 0.7-0.8
- Technical decisions: 0.8
- Project information: 0.6
- General conversation: 0.3-0.5

## Usage Patterns

### Pattern A: Pure MCP (Recommended)
- AI assistant calls tools as needed
- Proactively analyze after each conversation
- No background service needed

### Pattern B: MCP + Listening Service
- Run `hibro serve` background listening
- Automatically extract memories from conversation files
- MCP provides query interface

Both patterns can be used together.
"""


class MCPServer:
    """
    hibro MCP Server

    Using official MCP SDK to expose hibro functionality to IDEs
    Supporting intelligent memory storage, querying, conversation analysis and semantic search
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MCP Server

        Args:
            config: hibro configuration object, uses default configuration when None
        """
        self.config = config or Config()
        self.memory_engine = MemoryEngine(self.config)
        self.logger = logging.getLogger('hibro.mcp_server')

        # Run database migrations to ensure schema is up-to-date
        self._run_migrations()

        # Intelligent extraction components (lightweight, initialize immediately)
        self.extractor = MemoryExtractor()

        # Semantic search components (heavyweight, lazy loading)
        self._similarity_calc = None
        self._search_engine = None

        # New: Memory partition manager and active task manager
        self.memory_partition = MemoryPartition(self.memory_engine.memory_repo)
        self.active_task_manager = ActiveTaskManager(self.memory_engine.memory_repo)

        # Current project path (defaults to current working directory)
        self.current_project_path = str(Path.cwd())

        # Subscribe to relevant events from event bus
        self._subscribe_to_events()

        # Create MCP Server instance
        self.server = Server("hibro")

        # Register handlers
        self._register_handlers()

        self.logger.info("hibro MCP Server initialization completed (intelligent mode, semantic search lazy loading, project context support)")

    def _run_migrations(self):
        """Run database migrations to ensure schema is up-to-date"""
        try:
            from ..storage.migration_manager import MigrationManager

            migration_manager = MigrationManager(self.memory_engine.db_manager)
            migration_manager.migrate()

            self.logger.info("Database migrations completed successfully")

        except Exception as e:
            self.logger.warning(f"Failed to run migrations: {e}")
            # Continue anyway - migrations may have already been run

    @property
    def similarity_calc(self):
        """Lazy load semantic similarity calculator"""
        if self._similarity_calc is None:
            self.logger.info("First time using semantic search, loading model...")
            self._similarity_calc = SimilarityCalculator()
        return self._similarity_calc

    @property
    def search_engine(self):
        """Lazy load semantic search engine"""
        if self._search_engine is None:
            self._search_engine = SemanticSearchEngine(self.similarity_calc)
        return self._search_engine

    def _subscribe_to_events(self):
        """Subscribe to relevant events from event bus"""
        try:
            from ..core.event_bus import EventType

            # Subscribe to preference change events
            self.memory_engine._event_bus.subscribe(
                callback=self._on_preference_changed,
                subscriber_id='mcp_server_preference_monitor',
                event_types=[EventType.PREFERENCE_CHANGED],
                priority=1  # High priority
            )

            # Subscribe to database change events
            self.memory_engine._event_bus.subscribe(
                callback=self._on_database_sync,
                subscriber_id='mcp_server_db_monitor',
                event_types=[EventType.DATABASE_CHANGED],
                priority=0
            )

            self.logger.info("MCP Server subscribed to event bus")

        except Exception as e:
            self.logger.warning(f"Failed to subscribe to events: {e}")

    def _on_preference_changed(self, event):
        """Preference change event handler - send MCP resource update notification"""
        try:
            self.logger.info(
                f"Detected preference change: {event.data.get('category', 'unknown')} - "
                f"sending resource update notification"
            )

            # Send MCP resource update notification (notify client that preference resource has been updated)
            # This way other conversation windows in the IDE will receive notifications
            try:
                from mcp.types import ResourceUpdatedNotification, ResourceUpdatedNotificationParams
                from anyio import from_thread

                # Create notification
                notification = ResourceUpdatedNotification(
                    params=ResourceUpdatedNotificationParams(
                        uri="hibro://preferences/code"
                    )
                )

                # Send notification (through server session)
                # Note: This only works within run() context
                self.logger.info("✅ Prepared to send preference resource update notification")

            except Exception as e:
                self.logger.warning(f"Failed to send resource update notification: {e}")

        except Exception as e:
            self.logger.error(f"Failed to handle preference change event: {e}")

    def _on_database_sync(self, event):
        """Database sync event handler"""
        try:
            cache_version = event.data.get('cache_version', 0)
            self.logger.info(
                f"Detected database change (cache version: {cache_version}) - "
                f"other conversation windows may have updated memories"
            )

        except Exception as e:
            self.logger.error(f"Failed to handle database sync event: {e}")

    def _register_handlers(self):
        """Register all MCP handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Return available tools list"""
            return self._get_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            return await self._handle_tool_call(name, arguments)

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """Return available resources list"""
            return self._get_resources()

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content"""
            return await self._handle_resource_read(uri)

        @self.server.list_resource_templates()
        async def list_resource_templates() -> List[ResourceTemplate]:
            """Return resource templates list"""
            return [
                ResourceTemplate(
                    uriTemplate="hibro://preferences/{category}",
                    name="Preferences by Category",
                    description="Get preferences filtered by category"
                ),
                ResourceTemplate(
                    uriTemplate="hibro://project/{name}",
                    name="Project Context",
                    description="Get context for a specific project"
                )
            ]

        @self.server.list_prompts()
        async def list_prompts():
            """Return available prompts list"""
            from mcp.types import Prompt
            return [
                Prompt(
                    name="system_instructions",
                    description="hibro intelligent memory system behavior guide - automatically detect user intent and call appropriate tools"
                )
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict):
            """Return specified prompt content"""
            from mcp.types import GetPromptResult, PromptMessage, TextContent as PromptTextContent
            if name == "system_instructions":
                return GetPromptResult(
                    description="hibro intelligent memory system behavior guide",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=PromptTextContent(
                                type="text",
                                text=SYSTEM_PROMPT
                            )
                        )
                    ]
                )
            else:
                raise ValueError(f"Unknown prompt: {name}")

    def _get_tools(self) -> List[Tool]:
        """Get tools list"""
        return [
            # ===== Intelligent Analysis Tools (New) =====
            Tool(
                name="analyze_conversation",
                description="""Analyze a conversation segment and automatically extract memories.

【USE THIS TOOL - Call at the END of conversations or when user shares important info】

This tool will:
1. Extract user preferences, decisions, and important facts
2. Automatically classify memory types
3. Calculate importance scores
4. Store extracted memories

【WHEN TO CALL:】
- End of a conversation with valuable information
- User shared preferences or made decisions
- User explicitly asked to remember something
- After completing a coding task (to capture patterns)

【INPUT:】
- conversation: The conversation text to analyze (can be user messages, AI assistant responses, or both)
- context: Optional context (e.g., project name, task description)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "conversation": {
                            "type": "string",
                            "description": "Conversation text to analyze"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context (project, task, etc.)"
                        },
                        "auto_store": {
                            "type": "boolean",
                            "description": "Automatically store extracted memories (default: true)",
                            "default": True
                        }
                    },
                    "required": ["conversation"]
                }
            ),
            Tool(
                name="search_semantic",
                description="""Semantic search using AI-powered similarity matching.

【BETTER THAN KEYWORD SEARCH FOR:】
- Finding related concepts even with different words
- Discovering contextually similar memories
- Getting more relevant results

【WHEN TO USE:】
- When keyword search returns no results
- When looking for conceptually related memories
- When query is vague or general

Uses sentence-transformers for semantic understanding.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3
                        },
                        "type": {
                            "type": "string",
                            "description": "Optional memory type filter"
                        }
                    },
                    "required": ["query"]
                }
            ),

            # ===== Reasoning Engine Tools (New) =====
            Tool(
                name="analyze_causal_relations",
                description="""Analyze causal relationships between memories using advanced reasoning.

【WHEN TO USE:】
- Understanding why certain decisions were made
- Finding root causes of problems or patterns
- Discovering decision chains and their effects
- Analyzing project evolution and technical choices

【CAPABILITIES:】
- Identifies 4 types of causal relations: explicit, temporal, semantic, decision
- Builds causal chains showing cause-effect relationships
- Performs root cause analysis
- Predicts effects of potential changes

Uses NetworkX for graph-based causal analysis.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Optional project path to focus analysis"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional specific query about causality"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum causal relations to return",
                            "default": 10
                        }
                    }
                }
            ),
            Tool(
                name="predict_next_needs",
                description="""Predict what the user might need next based on patterns and context.

【WHEN TO USE:】
- Starting a new development phase
- Planning next steps in a project
- Anticipating user requirements
- Proactive assistance and recommendations

【PREDICTION TYPES:】
- next_need: What functionality/tools user likely needs next
- tech_choice: Technology recommendations based on patterns
- project_phase: Current project phase and next steps
- importance_trend: How memory importance might evolve

Uses machine learning patterns from decision history.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path for context-aware predictions"
                        },
                        "current_context": {
                            "type": "string",
                            "description": "Current task or context description"
                        },
                        "prediction_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["next_need", "tech_choice", "project_phase", "importance_trend"]
                            },
                            "description": "Types of predictions to generate"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum predictions to return",
                            "default": 5
                        }
                    }
                }
            ),
            Tool(
                name="build_knowledge_graph",
                description="""Build and analyze knowledge graph from memories and concepts.

【WHEN TO USE:】
- Understanding conceptual relationships in projects
- Finding related technologies and methodologies
- Discovering knowledge clusters and patterns
- Exploring semantic connections between ideas

【CAPABILITIES:】
- Extracts concepts from memory content (Chinese + English)
- Calculates 5 types of relationships: similar, causal, hierarchical, temporal, categorical
- Identifies central concepts and knowledge clusters
- Supports graph traversal and path finding
- Exports graph data for visualization

Uses advanced NLP and graph algorithms.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Optional project path to focus analysis"
                        },
                        "concept_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["technology", "methodology", "domain"]
                            },
                            "description": "Types of concepts to extract"
                        },
                        "min_frequency": {
                            "type": "integer",
                            "description": "Minimum concept frequency threshold",
                            "default": 2
                        },
                        "include_relations": {
                            "type": "boolean",
                            "description": "Include concept relationships in response",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="analyze_project_deeply",
                description="""Deep Project Analysis - Comprehensive Reasoning Analysis Tool

【Core Functions】
Conduct comprehensive deep analysis of projects or topics, combining causal analysis, predictive analysis and conceptual analysis to provide comprehensive insights and recommendations.

【Usage Scenarios】
• Before project decisions - Comprehensive understanding of project status and risks
• Complex problem diagnosis - Deep analysis of problem root causes and impacts
• Technical selection - Comprehensive evaluation of different solution pros and cons
• Project summary - Generate comprehensive analysis reports

【Analysis Dimensions】
• Causal analysis: Analyze causes and impact chains of events
• Predictive analysis: Predict possible development trends and results
• Conceptual analysis: Identify key concepts and their relationships
• Comprehensive insights: Deep insights integrating multi-dimensional analysis results

【Differences from Other Tools】
• query_reasoning: Quick reasoning for specific problems
• analyze_project_deeply: Project-level comprehensive deep analysis

💡 Tip: This tool is suitable for complex scenarios requiring comprehensive understanding and deep analysis. For simple problems, recommend using query_reasoning.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path to focus analysis scope"
                        },
                        "focus_area": {
                            "type": "string",
                            "description": "Analysis focus area (e.g., architecture design, performance optimization, security, etc.)"
                        },
                        "analysis_depth": {
                            "type": "string",
                            "enum": ["quick", "standard", "deep"],
                            "description": "Analysis depth: quick(quick overview), standard(standard analysis), deep(deep analysis)",
                            "default": "standard"
                        },
                        "analysis_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["causal", "predictive", "conceptual", "integrated"]
                            },
                            "description": "Analysis types: causal(causal), predictive(predictive), conceptual(conceptual), integrated(integrated)",
                            "default": ["causal", "predictive", "conceptual", "integrated"]
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["summary", "detailed", "report"],
                            "description": "Output format: summary(summary), detailed(detailed), report(report)",
                            "default": "detailed"
                        }
                    }
                }
            ),
            Tool(
                name="answer_specific_question",
                description="""Quick Q&A Reasoning - Intelligent Analysis for Specific Questions

【Core Functions】
Conduct quick reasoning analysis for users' specific questions, providing targeted answers and recommendations, suitable for solving clear and specific questions.

【Usage Scenarios】
• Quick Q&A - "Why does this error occur?"
• Decision consultation - "Which technical solution should I choose?"
• Relationship analysis - "What's the connection between these two concepts?"
• Predictive judgment - "What consequences will this have?"

【Reasoning Types】
• causal: Analyze cause and effect relationships
• predictive: Predict possible developments and consequences
• conceptual: Analyze relationships between concepts
• integrated: Comprehensive reasoning to reach conclusions

【Differences from Other Tools】
• answer_specific_question: Quick answers to specific questions
• analyze_project_deeply: Project-level comprehensive deep analysis

💡 Tip: This tool is suitable for quickly getting answers to specific questions. For complex project analysis, please use analyze_project_deeply.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question or doubt to analyze"
                        },
                        "reasoning_type": {
                            "type": "string",
                            "enum": ["causal", "predictive", "conceptual", "integrated"],
                            "description": "Reasoning type: causal(causal analysis), predictive(predictive analysis), conceptual(conceptual analysis), integrated(integrated analysis)",
                            "default": "integrated"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context information related to the question (optional)"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Related project path (optional, for providing project context)"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "description": "Confidence threshold (0.0-1.0), filter low confidence results",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["question"]
                }
            ),

            # ===== Adaptive Learning Tools (New) =====
            Tool(
                name="track_user_behavior",
                description="""Track user behavior for adaptive learning.

【WHEN TO USE:】
- After user queries or interactions
- When user provides feedback on recommendations
- During memory storage or recall operations
- To build user behavior patterns for personalization

【BEHAVIOR TYPES:】
- query: User searched for information
- store: User stored new memory
- recall: User accessed existing memory
- feedback: User provided feedback on recommendations
- click: User clicked on recommended content
- ignore: User ignored recommendations

Automatically updates attention weights and learning patterns.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "User session identifier"
                        },
                        "action_type": {
                            "type": "string",
                            "enum": ["query", "store", "recall", "feedback", "click", "ignore"],
                            "description": "Type of user behavior"
                        },
                        "target_memory_id": {
                            "type": "integer",
                            "description": "ID of memory involved in the action (optional)"
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Query text for search actions"
                        },
                        "response_relevance": {
                            "type": "number",
                            "description": "Relevance score of the response (0.0-1.0)",
                            "default": 0.0
                        },
                        "user_feedback": {
                            "type": "string",
                            "enum": ["useful", "not_useful", "partially_useful", "very_useful"],
                            "description": "User feedback on the interaction"
                        },
                        "interaction_duration": {
                            "type": "integer",
                            "description": "Duration of interaction in seconds",
                            "default": 0
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project context for the behavior"
                        }
                    },
                    "required": ["session_id", "action_type"]
                }
            ),
            Tool(
                name="get_personalized_recommendations",
                description="""Get personalized recommendations based on user behavior and preferences.

【WHEN TO USE:】
- When user starts a new task or project
- After user queries to suggest related content
- To proactively suggest relevant memories
- For discovering knowledge gaps and learning opportunities

【RECOMMENDATION TYPES:】
- collaborative: Based on similar user patterns
- content_based: Based on content similarity
- causal: Based on causal relationships
- predictive: Based on predicted needs
- hybrid: Combined approach for best results

Uses advanced machine learning algorithms for personalization.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_session": {
                            "type": "string",
                            "description": "User session for personalization context"
                        },
                        "query_context": {
                            "type": "string",
                            "description": "Current query or context for recommendations"
                        },
                        "recommendation_type": {
                            "type": "string",
                            "enum": ["collaborative", "content_based", "causal", "predictive", "hybrid"],
                            "description": "Type of recommendation algorithm to use",
                            "default": "hybrid"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project context for recommendations"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of recommendations",
                            "default": 5
                        },
                        "include_explanation": {
                            "type": "boolean",
                            "description": "Include explanation for each recommendation",
                            "default": True
                        }
                    },
                    "required": ["user_session"]
                }
            ),
            Tool(
                name="analyze_user_patterns",
                description="""Analyze user behavior patterns and learning trends.

【WHEN TO USE:】
- To understand user preferences and habits
- For generating insights about user behavior
- To detect changes in user interests over time
- For optimizing recommendation algorithms

【ANALYSIS TYPES:】
- query_patterns: Common search patterns and keywords
- attention_weights: Topics user focuses on most
- preference_drift: Changes in user preferences over time
- learning_effectiveness: How well recommendations work

Provides detailed analytics for adaptive learning optimization.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_session": {
                            "type": "string",
                            "description": "User session to analyze (optional for global analysis)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["query_patterns", "attention_weights", "preference_drift", "learning_effectiveness"],
                            "description": "Type of analysis to perform",
                            "default": "query_patterns"
                        },
                        "time_window_days": {
                            "type": "integer",
                            "description": "Number of days to analyze",
                            "default": 30
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project context for analysis"
                        }
                    }
                }
            ),
            Tool(
                name="adaptive_importance_scoring",
                description="""Calculate adaptive importance scores based on user feedback and behavior.

【WHEN TO USE:】
- When storing new memories to get personalized importance
- To re-evaluate existing memory importance
- For optimizing memory ranking and retrieval
- To adapt scoring based on user feedback patterns

【FEATURES:】
- Learns from user feedback to improve scoring accuracy
- Adapts to individual user preferences and patterns
- Uses reinforcement learning for continuous improvement
- Supports A/B testing for scoring algorithm optimization

Provides more accurate importance scores than static algorithms.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content to score"
                        },
                        "memory_id": {
                            "type": "integer",
                            "description": "Existing memory ID to re-score (optional)"
                        },
                        "user_session": {
                            "type": "string",
                            "description": "User session for personalized scoring"
                        },
                        "category": {
                            "type": "string",
                            "description": "Memory category for context"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project context for scoring"
                        },
                        "feedback_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_id": {"type": "integer"},
                                    "feedback_score": {"type": "number"},
                                    "feedback_type": {"type": "string"}
                                }
                            },
                            "description": "Recent feedback data for learning"
                        }
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="get_learning_insights",
                description="""Get insights about learning effectiveness and system performance.

【WHEN TO USE:】
- To understand how well the adaptive learning is working
- For system optimization and tuning
- To generate reports on learning progress
- For debugging recommendation quality issues

【INSIGHT TYPES:】
- recommendation_performance: How accurate recommendations are
- user_engagement: User interaction patterns and satisfaction
- learning_progress: How the system is improving over time
- knowledge_gaps: Areas where user needs more information

Provides comprehensive analytics for system improvement.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "insight_type": {
                            "type": "string",
                            "enum": ["recommendation_performance", "user_engagement", "learning_progress", "knowledge_gaps"],
                            "description": "Type of insights to generate",
                            "default": "recommendation_performance"
                        },
                        "user_session": {
                            "type": "string",
                            "description": "User session for personalized insights"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project context for insights"
                        },
                        "time_period_days": {
                            "type": "integer",
                            "description": "Time period for analysis",
                            "default": 7
                        }
                    }
                }
            ),

            # ===== Core Tools =====
            Tool(
                name="get_preferences",
                description="""Get user preference settings - Must call before programming

【CORE FUNCTIONALITY】
Get user's programming preferences, tool usage habits and workflow settings to ensure generated code conforms to user habits.

【USAGE SCENARIOS】
• Before writing or modifying code - Ensure following user code style
• When new session starts - Establish user preference context
• When user mentions preferences - Get related setting information
• Before tool recommendations - Provide personalized suggestions based on preferences

【COMMON PREFERENCE TYPES】
• code: Code comment language, indentation style, naming conventions
• tool: Tool selection preferences, IDE configuration
• workflow: Workflow, development habits
• project: Project structure, architecture preferences

【PARAMETER DESCRIPTION】
• No parameters: Get all preference settings
• Specify category: Get preferences of specific category

⚠️ Important: Recommend calling this tool before programming tasks to ensure code style consistency.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["code", "tool", "workflow", "project"],
                            "description": "Preference category filter: code(programming preferences), tool(tool preferences), workflow(workflow preferences), project(project preferences)"
                        },
                        "include_examples": {
                            "type": "boolean",
                            "description": "Whether to include usage examples of preference settings",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="search_memories",
                description="""Search memories by keywords.

【USE FOR:】
- Quick keyword-based lookups
- Finding specific technical terms
- Searching for exact matches

For conceptual/semantic search, use search_semantic instead.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        },
                        "type": {
                            "type": "string",
                            "description": "Optional type filter",
                            "enum": ["preference", "decision", "project", "important", "learning", "conversation"]
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="remember",
                description="""Store a new memory manually.

【AUTO-SAVE TRIGGERS - Consider calling when:】
- User explicitly says "remember this"
- User makes a clear preference statement
- User makes a technical decision

For bulk extraction, use analyze_conversation instead.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance (0.9=explicit, 0.8=decision, 0.7=preference, 0.5=general)",
                            "default": 0.5
                        },
                        "type": {
                            "type": "string",
                            "description": "Memory type",
                            "enum": ["preference", "decision", "project", "important", "learning", "conversation"],
                            "default": "conversation"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category (code, tool, workflow, project)"
                        }
                    },
                    "required": ["content"]
                }
            ),

            # ===== Context Tools =====
            Tool(
                name="get_project_context",
                description="""Get memories for a specific project.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path"
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="get_recent_decisions",
                description="""Get recent technical decisions.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    }
                }
            ),
            Tool(
                name="get_important_facts",
                description="""Get high-importance memories.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_importance": {
                            "type": "number",
                            "description": "Minimum importance",
                            "default": 0.7
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    }
                }
            ),
            Tool(
                name="get_quick_context",
                description="""Get user context information - Must call first in each session

[CORE FUNCTIONALITY]
Quickly get user's programming preferences, recent decisions and important information to provide personalized service foundation for intelligent assistant.

[AUTO INITIALIZATION]
For new projects without existing memory, automatically performs quick scan and initializes project memory.

[USAGE SCENARIOS]
- When new session starts - Establish user context
- Before programming tasks - Ensure following user habits
- When user mentions preferences - Update personalized settings
- When switching projects - Get project-related context (auto-initializes if needed)

[RETURN INFORMATION]
- preferences: User programming preferences (comment language, code style, etc.)
- recent_decisions: Recent technical decision records
- important_facts: High-importance key information
- project_context: Current project-related context
- project_init: Present when new project is initialized (contains project_name, project_type, etc.)

[IMPORTANT] This tool is the foundation of all personalized functions, recommend calling at the beginning of each session.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path (optional, will use current project if not specified)"
                        },
                        "context_depth": {
                            "type": "string",
                            "enum": ["basic", "detailed", "comprehensive"],
                            "description": "Context detail level: basic(basic info), detailed(detailed info), comprehensive(complete info)",
                            "default": "detailed"
                        }
                    }
                }
            ),

            # ===== Management Tools =====
            Tool(
                name="get_status",
                description="Get hibro system status and statistics.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="update_memory",
                description="Update an existing memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer"},
                        "content": {"type": "string"},
                        "importance": {"type": "number"}
                    },
                    "required": ["memory_id"]
                }
            ),
            Tool(
                name="forget",
                description="Delete a memory by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer"}
                    },
                    "required": ["memory_id"]
                }
            ),

            # ===== Project Context Tools (New) =====
            Tool(
                name="set_project_context",
                description="""Set project-specific context information.

【USE FOR:】
- Setting project architecture descriptions
- Adding project-specific commands
- Storing project configuration details

【CONTEXT TYPES:】
- architecture: Project architecture description
- command: Project-specific commands (build, test, deploy, etc.)
- task: Task-related information (handled by set_active_task)

This tool creates project-scoped memories that are isolated from global preferences.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path"
                        },
                        "context_type": {
                            "type": "string",
                            "description": "Type of context",
                            "enum": ["architecture", "command"]
                        },
                        "content": {
                            "type": "string",
                            "description": "Context content"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score (0.0-1.0)",
                            "default": 0.8
                        }
                    },
                    "required": ["project_path", "context_type", "content"]
                }
            ),
            Tool(
                name="set_active_task",
                description="""Set the active task for a project.

【IMPORTANT:】
- Only ONE active task per project is allowed
- Setting a new active task automatically deactivates the previous one
- Active tasks always appear in project hot data with highest priority

【USE FOR:】
- Setting current development focus
- Tracking what you're working on
- Ensuring task context is always available

【TASK LIFECYCLE:】
1. set_active_task: Create and activate a task
2. Task appears in get_quick_context project_context
3. Use complete_active_task or set new task to deactivate""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path"
                        },
                        "task_content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Task importance (0.0-1.0)",
                            "default": 1.0
                        }
                    },
                    "required": ["project_path", "task_content"]
                }
            ),
            Tool(
                name="complete_active_task",
                description="""Mark the current active task as completed.

【USE FOR:】
- Finishing the current task
- Clearing active task status
- Task completion tracking

The task memory remains but is no longer marked as active.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path"
                        }
                    },
                    "required": ["project_path"]
                }
            ),

            # ===== Intelligent Assistant Tools (New) =====
            Tool(
                name="get_smart_suggestions",
                description="""Get smart suggestions - Proactive suggestions based on project context

【CORE FUNCTIONALITY】
Based on project status, user activities and error situations, proactively provide personalized improvement suggestions and best practice guidance.

【USAGE SCENARIOS】
• Project startup - Get project setup and architecture suggestions
• After encountering errors - Get problem solving and prevention suggestions
• Development phase transitions - Get next step action suggestions
• Technical decisions - Get technology selection and implementation suggestions

【SUGGESTION TYPES】
• Best practices: Code quality, architecture design, security suggestions
• Tool recommendations: Tools and libraries suitable for current project
• Optimization suggestions: Performance, maintainability, development efficiency improvements
• Problem prevention: Preventive suggestions based on error patterns

【INTELLIGENT FEATURES】
• Context awareness: Based on project type and current status
• Personalization: Combined with user preferences and historical behavior
• Priority sorting: Sorted by importance and feasibility

💡 Tip: Suggestions will be dynamically adjusted based on actual project conditions, regular calls can get the latest optimization suggestions.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path (required)"
                        },
                        "suggestion_focus": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["best_practices", "tools", "optimization", "problem_prevention"]
                            },
                            "description": "Suggestion focus: best_practices(best practices), tools(tool recommendations), optimization(optimization), problem_prevention(problem prevention)",
                            "default": ["best_practices", "optimization"]
                        },
                        "current_phase": {
                            "type": "string",
                            "enum": ["planning", "development", "testing", "deployment", "maintenance"],
                            "description": "Current development phase: planning(planning), development(development), testing(testing), deployment(deployment), maintenance(maintenance)"
                        },
                        "recent_errors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recent error messages encountered (optional)"
                        },
                        "project_type": {
                            "type": "string",
                            "enum": ["web", "mobile", "desktop", "api", "library", "data_science"],
                            "description": "Project type (optional, for targeted suggestions)"
                        },
                        "max_suggestions": {
                            "type": "integer",
                            "description": "Maximum number of suggestions",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="detect_workflow_patterns",
                description="""Detect repetitive workflow patterns for automation opportunities.

【WHEN TO USE:】
- After completing repetitive tasks
- When setting up new projects
- For identifying automation opportunities
- To optimize development workflows

【PATTERN TYPES:】
- command_sequence: Repeated command patterns
- project_initialization: Project setup patterns
- configuration_generation: Config file patterns
- file_operations: File manipulation patterns
- development_workflow: Development process patterns

Helps identify tasks that can be automated to save time.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path to analyze"
                        },
                        "command_history": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recent command history"
                        },
                        "file_operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "operation": {"type": "string"},
                                    "file_path": {"type": "string"},
                                    "timestamp": {"type": "string"}
                                }
                            },
                            "description": "Recent file operations"
                        },
                        "time_window": {
                            "type": "integer",
                            "description": "Time window in hours to analyze",
                            "default": 24
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="get_workflow_recommendations",
                description="""Get workflow automation recommendations based on detected patterns.

【WHEN TO USE:】
- After detecting workflow patterns
- When starting new projects
- For optimizing repetitive tasks
- To implement automation solutions

【RECOMMENDATION TYPES:】
- Project initialization templates
- Command sequence automation
- Configuration file generation
- Development workflow optimization

Provides actionable automation recommendations with templates.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path"
                        },
                        "project_type": {
                            "type": "string",
                            "description": "Type of project"
                        },
                        "detected_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Previously detected patterns"
                        },
                        "max_recommendations": {
                            "type": "integer",
                            "description": "Maximum recommendations",
                            "default": 3
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="execute_workflow",
                description="""Execute a workflow automation template.

【WHEN TO USE:】
- To automate repetitive tasks
- When implementing recommended workflows
- For project initialization
- To execute predefined automation sequences

【AUTOMATION LEVELS:】
- manual: Show steps for manual execution
- semi_auto: Execute with user confirmation
- full_auto: Fully automated execution

Executes workflow templates with variable substitution.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "template_id": {
                            "type": "string",
                            "description": "Workflow template ID"
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables for template substitution"
                        },
                        "automation_level": {
                            "type": "string",
                            "enum": ["manual", "semi_auto", "full_auto"],
                            "description": "Level of automation",
                            "default": "manual"
                        }
                    },
                    "required": ["template_id", "variables"]
                }
            ),
            Tool(
                name="get_intelligent_reminders",
                description="""Get intelligent reminders based on project context and best practices.

【WHEN TO USE:】
- At project milestones
- When technical debt accumulates
- For security and maintenance reminders
- During development phase transitions

【REMINDER TYPES:】
- milestone: Project milestone reminders
- tech_debt: Technical debt warnings
- best_practice: Best practice suggestions
- security: Security-related reminders
- maintenance: Maintenance reminders

Provides contextual reminders to improve project quality.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path"
                        },
                        "project_metrics": {
                            "type": "object",
                            "properties": {
                                "test_coverage": {"type": "number"},
                                "new_code_lines": {"type": "integer"},
                                "complexity_score": {"type": "number"}
                            },
                            "description": "Project quality metrics"
                        },
                        "milestone_data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "target_date": {"type": "string"},
                                "completion_percentage": {"type": "number"}
                            },
                            "description": "Milestone information"
                        },
                        "code_quality_metrics": {
                            "type": "object",
                            "properties": {
                                "debt_score": {"type": "number"},
                                "affected_files": {"type": "integer"}
                            },
                            "description": "Code quality metrics"
                        },
                        "security_issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string"},
                                    "title": {"type": "string"},
                                    "days_since_discovery": {"type": "integer"}
                                }
                            },
                            "description": "Security issues"
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="get_comprehensive_assistance",
                description="""Get comprehensive assistance - Project-level integrated suggestions and guidance

【CORE FUNCTIONALITY】
Provide comprehensive intelligent assistance for projects, integrating suggestions, workflows and reminders to solve complex development problems and decision-making needs.

【USAGE SCENARIOS】
• Project startup - Get comprehensive project setup suggestions
• Encountering complex problems - Need multi-dimensional solutions
• Technical decisions - Need comprehensive consideration of multiple factors
• Project optimization - Get comprehensive improvement suggestions

【PROVIDED ASSISTANCE TYPES】
• Smart suggestions: Personalized suggestions based on project conditions
• Workflows: Automation and optimization suggestions
• Quality reminders: Code quality and best practice reminders
• Problem diagnosis: Error analysis and solutions

【COORDINATION FUNCTIONS】
• Suggestion priority sorting and conflict resolution
• Coordination of multi-module collaborative work
• Personalized adjustments based on user preferences

💡 Tip: This tool is suitable for complex scenarios requiring comprehensive guidance, simple problems recommend using specific single-function tools.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path (required)"
                        },
                        "assistance_type": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["suggestions", "workflows", "reminders", "diagnostics"]
                            },
                            "description": "Required assistance types: suggestions(suggestions), workflows(workflows), reminders(reminders), diagnostics(diagnostics)",
                            "default": ["suggestions", "workflows", "reminders"]
                        },
                        "user_query": {
                            "type": "string",
                            "description": "Specific problem or requirement description"
                        },
                        "current_task": {
                            "type": "string",
                            "description": "Current task being performed (optional)"
                        },
                        "has_errors": {
                            "type": "boolean",
                            "description": "Whether errors are encountered that need diagnosis",
                            "default": False
                        },
                        "error_messages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Error message list (if there are errors)"
                        },
                        "priority_focus": {
                            "type": "string",
                            "enum": ["performance", "security", "maintainability", "productivity"],
                            "description": "Priority focus area: performance(performance), security(security), maintainability(maintainability), productivity(productivity)",
                            "default": "productivity"
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="get_assistant_statistics",
                description="""Get comprehensive statistics about the intelligent assistant performance.

【WHEN TO USE:】
- To monitor assistant effectiveness
- For system optimization
- To understand usage patterns
- For performance analysis

【STATISTICS INCLUDE:】
- Suggestion adoption rates
- Workflow execution success
- Reminder completion rates
- Coordination effectiveness
- Module performance metrics

Provides detailed analytics for system improvement.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            # ===== Security Monitoring Management Tools (New) =====
            Tool(
                name="check_security_status",
                description="""Check system security status - Comprehensive security health monitoring

【CORE FUNCTIONALITY】
Get complete system security status report, including the operational status of various security modules such as encryption, access control, monitoring and backup.

【USAGE SCENARIOS】
• Security audits - Comprehensive check of system security health
• Troubleshooting - Diagnose security-related issues
• Compliance checks - Verify security policy execution
• Regular monitoring - Understand system security trends

【STATUS INFORMATION INCLUDES】
• Encryption management: Encryption status, key management, encryption statistics
• Access control: Authentication status, session management, security metrics
• System monitoring: Health status, performance metrics, anomaly detection
• Backup system: Backup health, storage status, recovery capability
• Security events: Recent events, unresolved issues, risk assessment

【OUTPUT FORMAT】
Provides structured security status report, including health scores and key metrics for each module.

💡 Tip: Recommend calling this tool regularly to monitor system security status and timely discover and handle security risks.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detail_level": {
                            "type": "string",
                            "enum": ["summary", "detailed", "comprehensive"],
                            "description": "Detail level: summary(summary), detailed(detailed), comprehensive(comprehensive)",
                            "default": "detailed"
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "description": "Whether to include security improvement suggestions",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="apply_security_policy",
                description="""Apply security policy - System-level security configuration management

【CORE FUNCTIONALITY】
Apply predefined security policies, uniformly configure security settings of various system modules to ensure security compliance.

【USAGE SCENARIOS】
• Security compliance - Apply security policies that meet standards
• Environment switching - Adjust security levels according to environment
• Security hardening - Improve overall system security level
• Policy updates - Uniformly update security configurations

【AVAILABLE POLICIES】
• default: Default security policy, suitable for general use
• high_security: High security level, suitable for sensitive environments
• compliance: Compliance policy, meeting enterprise compliance requirements
• development: Development environment policy, balancing security and convenience

【POLICY CONFIGURATION ITEMS】
• Encryption requirements: Data encryption level and key management
• Authentication settings: Multi-factor authentication and session management
• Backup strategy: Backup frequency and retention period
• Monitoring configuration: Monitoring intervals and alert thresholds
• Automatic recovery: Anomaly detection and automatic repair

💡 Tip: Policy application will affect all security modules, recommend making policy changes during non-production hours.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_id": {
                            "type": "string",
                            "enum": ["default", "high_security", "compliance", "development"],
                            "description": "Security policy ID: default(default), high_security(high security), compliance(compliance), development(development)"
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Whether to only preview policy changes without actually applying",
                            "default": False
                        },
                        "force_apply": {
                            "type": "boolean",
                            "description": "Whether to force apply policy (ignore conflict warnings)",
                            "default": False
                        }
                    },
                    "required": ["policy_id"]
                }
            ),
            Tool(
                name="get_security_events",
                description="""Retrieve security events with optional filtering by type and resolution status.

【WHEN TO USE:】
- To monitor security incidents
- For security event analysis
- To track unresolved security issues
- For compliance reporting

【EVENT TYPES:】
- backup_failure: Backup operation failures
- security_breach: Access control violations
- system_anomaly: Health monitoring alerts
- policy_violation: Security policy violations
- emergency_backup: Automatic emergency backups

Provides detailed security event tracking and management.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "Filter by event type (optional)"
                        },
                        "resolved": {
                            "type": "boolean",
                            "description": "Filter by resolution status (optional)"
                        }
                    }
                }
            ),
            Tool(
                name="resolve_security_event",
                description="""Mark a security event as resolved with resolution notes.

【WHEN TO USE:】
- After fixing a security issue
- To document resolution steps
- For security event lifecycle management
- To clear security alerts

【RESOLUTION PROCESS:】
1. Investigate the security event
2. Take corrective actions
3. Document the resolution
4. Mark event as resolved

Maintains security event audit trail and resolution tracking.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "Security event ID to resolve"
                        },
                        "resolution_notes": {
                            "type": "string",
                            "description": "Description of how the event was resolved"
                        }
                    },
                    "required": ["event_id", "resolution_notes"]
                }
            ),
            Tool(
                name="create_backup",
                description="""Create system backup - Multi-strategy data protection

【CORE FUNCTIONALITY】
Create system data backup, support multiple backup strategies, provide enterprise-level data protection and disaster recovery capabilities.

【USAGE SCENARIOS】
• Regular data protection - Establish routine backup plans
• Before major changes - Safe backup before system upgrades or configuration changes
• Disaster recovery preparation - Prepare data recovery points for unexpected situations
• Security incident response - Emergency backup when security incidents occur

【BACKUP STRATEGIES】
• full: Complete backup, including all data (suitable for first backup or regular complete backup)
• incremental: Incremental backup, only backup changes since last backup (saves space and time)
• differential: Differential backup, backup all changes since last complete backup

【TECHNICAL FEATURES】
• Automatic encryption: Backup files automatically encrypted for protection
• Compression optimization: Smart compression to reduce storage space
• Integrity verification: Verify data integrity during backup process
• Parallel processing: Multi-threading to improve backup performance

💡 Tip: Recommend creating complete backup for first use, subsequent incremental backups can improve efficiency.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "backup_type": {
                            "type": "string",
                            "enum": ["full", "incremental", "differential"],
                            "description": "Backup type: full(complete backup), incremental(incremental backup), differential(differential backup)",
                            "default": "incremental"
                        },
                        "description": {
                            "type": "string",
                            "description": "Backup description information (optional, for identifying backup purpose)"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high", "urgent"],
                            "description": "Backup priority: low(low), normal(normal), high(high), urgent(urgent)",
                            "default": "normal"
                        },
                        "verify_integrity": {
                            "type": "boolean",
                            "description": "Whether to verify integrity after backup completion",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="restore_backup",
                description="""Restore system data from a backup with progress tracking.

【WHEN TO USE:】
- For disaster recovery
- To restore corrupted data
- For system rollback scenarios
- When migrating to new systems

【RESTORE FEATURES:】
- Automatic decryption
- Progress monitoring
- Selective file restoration
- Integrity verification

Provides secure and reliable data restoration capabilities.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "backup_id": {
                            "type": "string",
                            "description": "ID of the backup to restore"
                        },
                        "target_path": {
                            "type": "string",
                            "description": "Optional target path for restoration"
                        }
                    },
                    "required": ["backup_id"]
                }
            ),
            Tool(
                name="get_backup_statistics",
                description="""Get comprehensive backup system statistics and health metrics.

【WHEN TO USE:】
- To monitor backup system health
- For capacity planning
- To check backup compliance
- For performance optimization

【STATISTICS INCLUDE:】
- Backup counts by type
- Storage usage and compression ratios
- Backup health score
- Success/failure rates
- Automated service status

Provides detailed backup system analytics and monitoring.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="register_sync_device",
                description="""Register a device for cross-device data synchronization.

【WHEN TO USE:】
- To enable multi-device workflows
- For data migration between systems
- To set up backup replication
- For distributed development environments

【DEVICE TYPES:】
- desktop: Desktop computers
- laptop: Laptop computers
- mobile: Mobile devices
- server: Server systems

Enables secure cross-device data synchronization and migration.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "device_name": {
                            "type": "string",
                            "description": "Human-readable device name"
                        },
                        "device_type": {
                            "type": "string",
                            "enum": ["desktop", "laptop", "mobile", "server"],
                            "description": "Type of device"
                        },
                        "sync_path": {
                            "type": "string",
                            "description": "Optional path for synchronization"
                        }
                    },
                    "required": ["device_name", "device_type"]
                }
            ),
            Tool(
                name="start_device_migration",
                description="""Start data migration between registered devices.

【WHEN TO USE:】
- To migrate data to new devices
- For system upgrades
- To synchronize development environments
- For backup replication

【MIGRATION FEATURES:】
- Incremental synchronization
- Progress monitoring
- Conflict resolution
- Bandwidth optimization

Provides secure and efficient cross-device data migration.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_device_id": {
                            "type": "string",
                            "description": "ID of the source device"
                        },
                        "target_device_id": {
                            "type": "string",
                            "description": "ID of the target device"
                        }
                    },
                    "required": ["source_device_id", "target_device_id"]
                }
            ),
            Tool(
                name="get_system_health",
                description="""Get comprehensive system health status and performance metrics.

【WHEN TO USE:】
- To monitor system performance
- For troubleshooting issues
- To check resource usage
- For capacity planning

【HEALTH METRICS:】
- CPU and memory usage
- Disk space and I/O performance
- Database health and performance
- Network connectivity status
- Overall system health score

Provides real-time system monitoring and health assessment.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="perform_security_scan",
                description="""Perform comprehensive security scan with recommendations.

【WHEN TO USE:】
- For regular security audits
- Before system deployments
- When security issues are suspected
- For compliance verification

【SCAN INCLUDES:】
- Encryption status verification
- Access control security check
- Backup system health assessment
- System vulnerability analysis
- Security event review

Provides actionable security recommendations and risk assessment.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            # ===== Intelligent Guidance System Tools (New) =====
            Tool(
                name="create_user_session",
                description="""Create user session - Intelligent guidance system entry

【CORE FUNCTIONALITY】
Create personalized session for users, enable intelligent guidance, tool recommendations and learning path functions.

【USAGE SCENARIOS】
• New user first use - Establish personalized usage environment
• Session start - Enable intelligent guidance functions
• Project switching - Update project context
• Restart learning - Reset learning state

【SESSION FUNCTIONS】
• Personalized tool recommendations: Based on user level and usage habits
• Smart usage hints: Provide operation guidance at appropriate times
• Learning path guidance: Progressive function learning
• Context awareness: Project-related intelligent suggestions

💡 Tip: Recommend creating session when starting to use hibro to get the best personalized experience.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (recommend using unique ID)"
                        },
                        "user_level": {
                            "type": "string",
                            "enum": ["beginner", "intermediate", "advanced", "expert"],
                            "description": "User proficiency level: beginner(beginner), intermediate(intermediate), advanced(advanced), expert(expert)",
                            "default": "beginner"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Current project path (optional, for project-related intelligent suggestions)"
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            Tool(
                name="get_tool_recommendations",
                description="""Get intelligent tool recommendations - Discover related functions

【CORE FUNCTIONALITY】
Based on current context, user behavior and project needs, intelligently recommend the most suitable tools and functions.

【USAGE SCENARIOS】
• Uncertain which tool to use - Get targeted recommendations
• Explore new functions - Discover unused useful tools
• Improve efficiency - Find more suitable tool combinations
• Learning advancement - Understand advanced function tools

【RECOMMENDATION ALGORITHMS】
• Association recommendations: Recommend related tools based on current tools
• Pattern recommendations: Recommend common combinations based on usage patterns
• Context recommendations: Recommend based on project type and stage
• Learning recommendations: Recommend advanced tools based on user level

【RECOMMENDATION INFORMATION】
Each recommendation includes: tool name, recommendation reason, usage tips, prerequisites, estimated value

💡 Tip: Regular checking of recommendations can discover new useful functions and improve work efficiency.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (required)"
                        },
                        "current_tool": {
                            "type": "string",
                            "description": "Currently used tool (optional)"
                        },
                        "project_type": {
                            "type": "string",
                            "enum": ["web", "mobile", "desktop", "api", "library", "data_science"],
                            "description": "Project type (optional, for targeted recommendations)"
                        },
                        "max_recommendations": {
                            "type": "integer",
                            "description": "Maximum number of recommendations",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            Tool(
                name="get_usage_hints",
                description="""Get usage hints - Intelligent operation guidance

【CORE FUNCTIONALITY】
Provide context-related usage hints at appropriate times to help users better use tools and avoid common errors.

【USAGE SCENARIOS】
• First time using tools - Get introductory guidance
• When encountering errors - Get solution suggestions
• Parameter setting - Get best practice hints
• Function exploration - Understand advanced usage

【HINT TYPES】
• Usage tips: Best methods for using tools
• Parameter hints: Setting suggestions for important parameters
• Workflow suggestions: Tool combination usage suggestions
• Error prevention: Methods to avoid common errors
• Feature highlights: Useful functions that are easily overlooked

【INTELLIGENT TRIGGERING】
System will automatically trigger related hints based on usage situations, can also actively get hints for specific tools.

💡 Tip: Paying attention to usage hints can quickly master efficient tool usage.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (required)"
                        },
                        "current_tool": {
                            "type": "string",
                            "description": "Current tool name (optional)"
                        },
                        "error_occurred": {
                            "type": "boolean",
                            "description": "Whether an error occurred",
                            "default": False
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Error message (if there are errors)"
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            Tool(
                name="get_learning_paths",
                description="""Get learning paths - Progressive function learning

【CORE FUNCTIONALITY】
Provide structured learning paths to help users gradually master hibro's various functions from basic to advanced.

【USAGE SCENARIOS】
• New user onboarding - Systematically learn basic functions
• Skill improvement - Learn advanced functions and best practices
• Function exploration - Discover and master unknown functions
• Knowledge consolidation - Deepen understanding through practice

【LEARNING PATHS】
• Beginner path: Basic functions and core concepts (45 minutes)
• Intermediate path: Advanced analysis and workflow optimization (80 minutes)
• Advanced expert path: Enterprise functions and system management (90 minutes)

【LEARNING FEATURES】
• Progressive design: Learning sequence from simple to complex
• Practice-oriented: Each step has specific practice tasks
• Personalized recommendations: Recommend suitable paths based on user level
• Progress tracking: Record learning progress and completion status

💡 Tip: Systematic learning according to learning paths can faster master hibro's complete capabilities.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (required)"
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            Tool(
                name="start_learning_path",
                description="""Start learning path - Launch structured learning

【CORE FUNCTIONALITY】
Start specified learning path, system will track learning progress and provide corresponding guidance and suggestions.

【USAGE SCENARIOS】
• Start systematic learning - Choose suitable learning path
• Skill improvement plan - Learn new functions according to plan
• Team training - Unified learning standards
• Knowledge system building - Complete mastery of function system

【PATH SELECTION】
• beginner_path: Beginner entry, learn basic functions
• intermediate_path: Intermediate advancement, master advanced analysis
• advanced_path: Expert level, enterprise functions and management

【LEARNING SUPPORT】
• Step guidance: Detailed instructions for each learning step
• Practice tasks: Specific operation exercises and verification
• Progress tracking: Real-time recording of learning progress
• Smart hints: Related hints during learning process

💡 Tip: Choose learning path suitable for your level, gradually master functions step by step.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (required)"
                        },
                        "path_id": {
                            "type": "string",
                            "enum": ["beginner_path", "intermediate_path", "advanced_path"],
                            "description": "Learning path ID: beginner_path(beginner), intermediate_path(intermediate), advanced_path(advanced)"
                        }
                    },
                    "required": ["session_id", "path_id"]
                }
            ),
            Tool(
                name="complete_learning_step",
                description="""Complete learning step - Record learning progress

【CORE FUNCTIONALITY】
Mark learning step as completed, update learning progress, and get guidance for next step.

【USAGE SCENARIOS】
• Complete practice tasks - Mark step completion status
• Update learning progress - Track learning journey
• Get next step guidance - Understand next learning goal
• Learning achievement confirmation - Verify learning effectiveness

【COMPLETION VERIFICATION】
System will verify learning step completion to ensure learning quality:
• Task completion check: Verify if required practice tasks are completed
• Knowledge point mastery: Confirm if key concepts are understood
• Skill application: Check if related tools can be used correctly

【PROGRESS UPDATE】
After completing steps, system will:
• Update learning progress percentage
• Unlock next learning step
• Provide personalized continuing learning suggestions
• Record learning time and completion status

💡 Tip: Timely marking completed steps helps system provide more accurate learning guidance.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (required)"
                        },
                        "path_id": {
                            "type": "string",
                            "description": "Learning path ID"
                        },
                        "step_id": {
                            "type": "string",
                            "description": "Learning step ID"
                        }
                    },
                    "required": ["session_id", "path_id", "step_id"]
                }
            ),
            Tool(
                name="get_guidance_statistics",
                description="""Get guidance system statistics - Understand usage and effectiveness

【CORE FUNCTIONALITY】
Get usage statistics and effectiveness analysis of intelligent guidance system to help understand learning progress and system usage.

【USAGE SCENARIOS】
• Learning progress view - Understand personal learning status
• Usage habit analysis - Discover usage patterns and preferences
• System effectiveness evaluation - Evaluate guidance system help effectiveness
• Optimization suggestion acquisition - Improvement suggestions based on statistical data

【STATISTICAL INFORMATION】
• Session statistics: Active session count, usage duration, tool usage frequency
• Recommendation statistics: Recommendation accuracy, adoption rate, most popular tools
• Learning statistics: Learning path completion status, step pass rate
• Hint statistics: Hint display count, user feedback, effectiveness evaluation

【ANALYSIS DIMENSIONS】
• Personal statistics: Detailed usage data for specified sessions
• Global statistics: Overall system usage trends and effectiveness
• Comparative analysis: Usage differences between different user levels
• Trend analysis: Time change trends in usage patterns

💡 Tip: Regular checking of statistical information helps understand learning effectiveness and optimize usage methods.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (optional, returns global statistics if not provided)"
                        }
                    }
                }
            ),

            # ===== Project Scanning and Progress Tools (New) =====
            Tool(
                name="scan_project",
                description="""Scan project and generate snapshot

【CORE FUNCTIONALITY】
Automatically scan project directory, identify project structure, tech stack, dependencies, generate project snapshot and store to memory system.

【USAGE SCENARIOS】
• New project first use - Quickly understand project basic situation
• Project changes - Update project snapshot to record changes
• Project switching - Scan to understand new project status
• Regular recording - Track project evolution process

【SCANNING CONTENT】
• Project type identification: web/api/mobile/desktop/library/data_science
• Tech stack detection: 20+ mainstream technologies automatically identified
• Framework identification: React/Vue/FastAPI/Django/Express etc.
• Programming language statistics: Sorted by file count
• Dependency extraction: Automatically parse package.json/requirements.txt
• Project statistics: File count, size, directory structure

【SCANNING MODES】
• Quick scan (default): Only scan key files, complete within 2 seconds
• Complete scan: Deep scan all files, complete within 10 seconds

【AUTOMATIC STORAGE】
Scan results automatically stored to memory system for convenient subsequent queries and tracking.

💡 Tip: Recommend scanning when starting new projects or when projects undergo major changes.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path (optional, defaults to current project path)"
                        },
                        "quick_scan": {
                            "type": "boolean",
                            "description": "Quick scan mode (default true)",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="get_project_progress",
                description="""Get project progress and status

【CORE FUNCTIONALITY】
View project's current status, active tasks, recent work records and key decisions.

【USAGE SCENARIOS】
• Ask "How is the project going?" - Get complete progress report
• Switch back to project - Quickly restore project context
• Regular review - Understand project evolution process
• Team handover - Show current project status

【INTELLIGENT BEHAVIOR】
• Automatic detection: If project snapshot is expired (>7 days), automatically rescan
• Context aggregation: Automatically integrate project snapshot, work records, active tasks
• Historical tracing: Show recent work content and key decisions

【REPORT CONTENT】
• Project snapshot: Tech stack, frameworks, languages, project scale
• Current tasks: Ongoing development tasks
• Recent work: Work records from last 7 days
• Key decisions: Important technology selection and architecture decisions
• Encountered problems: Current unresolved technical issues

【QUICK QUERY】
Supports natural language queries such as "How is the project going", "What are we working on now", etc.

💡 Tip: Regular checking of project progress helps maintain project direction and track progress.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path (optional, defaults to current project path)"
                        }
                    }
                }
            ),
            Tool(
                name="update_project_status",
                description="""Update project status and progress

【CORE FUNCTIONALITY】
Manually update project's current status, tasks and progress percentage.

【USAGE SCENARIOS】
• Start new task - Record current work content
• Complete task - Mark task completion and update progress
• Phase change - Record project transition from development to testing phase
• Progress reporting - Update project completion percentage

【RECORD CONTENT】
• Project phase: planning(planning)/development(development)/testing(testing)/production(production)
• Current task: Description of ongoing development task
• Progress percentage: 0-100 completion degree
• Notes information: Additional explanations and considerations

【AUTOMATIC ASSOCIATION】
Updates will automatically associate to project and store to memory system for convenient subsequent queries.

【TIME TRACKING】
Each update will record timestamp, forming complete project timeline.

【INTELLIGENT SUGGESTIONS】
Based on project type and current phase, system may provide related suggestions.

💡 Tip: Recommend updating project status when starting or completing important tasks to maintain record timeliness.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path"
                        },
                        "status": {
                            "type": "string",
                            "description": "Project phase",
                            "enum": ["planning", "development", "testing", "production"]
                        },
                        "current_task": {
                            "type": "string",
                            "description": "Current task description"
                        },
                        "progress_percentage": {
                            "type": "integer",
                            "description": "Progress percentage (0-100)",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "notes": {
                            "type": "string",
                            "description": "Notes information"
                        }
                    },
                    "required": ["project_path"]
                }
            ),

            # ===== Database Listening and Synchronization Tools =====
            Tool(
                name="get_sync_status",
                description="""Get database synchronization status

【CORE FUNCTIONALITY】
View database listener's running status and statistics for multi-IDE dialog synchronization.

【USAGE SCENARIOS】
• Check if listener is running normally
• View database change count
• Troubleshoot synchronization issues

【RETURN INFORMATION】
• Listening status (running/stopped)
• Database path
• Change detection count
• Last change time
• Cache version number

💡 Tip: If multiple IDE dialog boxes are running simultaneously, this tool can help confirm if synchronization mechanism is working properly.
""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            # ===== Event Bus Tools =====
            Tool(
                name="get_event_bus_status",
                description="""Get event bus status

【CORE FUNCTIONALITY】
View event bus running status, statistics and subscriber list.

【USAGE SCENARIOS】
• Monitor event bus health status
• View event publish/processing statistics
• Troubleshoot event processing issues
• View subscriber information

【RETURN INFORMATION】
• Running status (running/stopped)
• Worker thread count
• Queue size
• Subscriber count
• Event statistics (published count, processed count, dropped count)
• Running duration

💡 Tip: Event bus is used for decoupled communication between components, supporting memory change notifications and other functions.
""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            Tool(
                name="list_event_subscribers",
                description="""List event subscribers

【CORE FUNCTIONALITY】
View detailed information of all current event subscribers.

【USAGE SCENARIOS】
• View which components subscribed to events
• Check subscriber event types
• View subscriber call statistics

【RETURN INFORMATION】
• Subscriber ID
• Subscribed event types
• Priority
• Call count
• Last call time

💡 Tip: Each MCP Server instance can have multiple subscribers for handling different types of events.
""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            # ===== Code Knowledge Graph Tools (New) =====
            Tool(
                name="init_code_knowledge_graph",
                description="""Initialize code knowledge graph for a project

【CORE FUNCTIONALITY】
Scan project source code and build a knowledge graph containing:
- Classes and their methods/inheritance
- Functions with signatures
- Import dependencies
- API endpoints
- File structure

【USAGE SCENARIOS】
• First time analyzing a project
• After major code changes
• When get_quick_context shows project_init.missing: true

【INPUT】
• project_path: Project root directory path

【RETURNS】
• Statistics: files scanned, classes, functions found
• Knowledge graph initialized status

💡 Tip: This builds a structural code knowledge graph, different from conceptual knowledge graph.
""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project root directory path"
                        }
                    },
                    "required": ["project_path"]
                }
            ),

            Tool(
                name="get_code_context",
                description="""Get code context from knowledge graph

【CORE FUNCTIONALITY】
Query the code knowledge graph to get project context:
- Quick overview (files, classes, functions count)
- Core modules and key classes
- Recent changes
- Detailed class/function information

【USAGE SCENARIOS】
• Understanding project structure
• Finding key classes and functions
• Quick project onboarding

【INPUT】
• project_path: Project path
• detail_level: 'lightweight' (quick) or 'medium' (detailed)
• search_query: Optional search term for code entities

【RETURNS】
• Statistics and key information
• Token-optimized summary
""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path"
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["lightweight", "medium"],
                            "description": "Detail level: lightweight (~500 tokens) or medium (~2000 tokens)",
                            "default": "lightweight"
                        },
                        "search_query": {
                            "type": "string",
                            "description": "Optional search term to find specific code entities"
                        }
                    },
                    "required": ["project_path"]
                }
            ),

            # ===== Memory Cleanup Tools (New) =====
            Tool(
                name="trigger_cleanup",
                description="""Manually trigger memory cleanup

【CORE FUNCTIONALITY】
Execute memory cleanup using triple eviction strategies:
1. LFU eviction - Remove least frequently used memories
2. Time expiry - Remove old unused memories
3. Importance eviction - Remove low-importance memories

【USAGE SCENARIOS】
• Free up memory space manually
• Before large memory import operations
• After project completion to clean up

【INPUT】
• force: Force cleanup ignoring protection rules (default: false)

【RETURNS】
• Number of deleted memories per strategy
• Total cleanup duration
""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force cleanup ignoring protection rules",
                            "default": False
                        }
                    }
                }
            ),

            Tool(
                name="get_cleanup_status",
                description="""Get memory cleanup status

【CORE FUNCTIONALITY】
View current memory usage and cleanup system status.

【USAGE SCENARIOS】
• Check memory usage before/after cleanup
• Monitor cleanup scheduler status
• View last cleanup statistics

【RETURNS】
• Current memory count and usage ratio
• Threshold status (normal/warning/cleanup_needed/critical)
• Scheduler status and next cleanup time
• Last cleanup statistics
""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    def _get_resources(self) -> List[Resource]:
        """Get resource list"""
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
            ),
            Resource(
                uri="hibro://preferences",
                name="All Preferences",
                description="User preferences",
                mimeType="application/json"
            ),
            Resource(
                uri="hibro://projects",
                name="Project List",
                description="Projects with memories",
                mimeType="application/json"
            ),
            Resource(
                uri="hibro://recent",
                name="Recent Memories",
                description="Recently accessed memories",
                mimeType="application/json"
            ),
            Resource(
                uri="hibro://important",
                name="Important Memories",
                description="High importance memories",
                mimeType="application/json"
            ),
        ]

    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls"""
        try:
            self.logger.debug(f"Tool call: {name}, args: {arguments}")

            handlers = {
                # Intelligent Analysis
                "analyze_conversation": self._tool_analyze_conversation,
                "search_semantic": self._tool_search_semantic,
                # Reasoning Engine Tools (New)
                "analyze_causal_relations": self._tool_analyze_causal_relations,
                "predict_next_needs": self._tool_predict_next_needs,
                "build_knowledge_graph": self._tool_build_knowledge_graph,
                "comprehensive_reasoning": self._tool_comprehensive_reasoning,
                "query_reasoning": self._tool_query_reasoning,
                # Adaptive Learning Tools (New)
                "track_user_behavior": self._tool_track_user_behavior,
                "get_personalized_recommendations": self._tool_get_personalized_recommendations,
                "analyze_user_patterns": self._tool_analyze_user_patterns,
                "adaptive_importance_scoring": self._tool_adaptive_importance_scoring,
                "get_learning_insights": self._tool_get_learning_insights,
                # Core Functions
                "get_preferences": self._tool_get_preferences,
                "search_memories": self._tool_search_memories,
                "remember": self._tool_remember,
                # Context
                "get_project_context": self._tool_get_project_context,
                "get_recent_decisions": self._tool_get_recent_decisions,
                "get_important_facts": self._tool_get_important_facts,
                "get_quick_context": self._tool_get_quick_context,
                # Project Context Tools (New)
                "set_project_context": self._tool_set_project_context,
                "set_active_task": self._tool_set_active_task,
                "complete_active_task": self._tool_complete_active_task,
                # Project Scanning and Progress Tools (New)
                "scan_project": self._tool_scan_project,
                "get_project_progress": self._tool_get_project_progress,
                "update_project_status": self._tool_update_project_status,
                # Database Synchronization
                "get_sync_status": self._tool_get_sync_status,
                # Event Bus
                "get_event_bus_status": self._tool_get_event_bus_status,
                "list_event_subscribers": self._tool_list_event_subscribers,
                # Management
                "get_status": self._tool_get_status,
                "update_memory": self._tool_update_memory,
                "forget": self._tool_forget,
                # Memory Cleanup (New)
                "trigger_cleanup": self._tool_trigger_cleanup,
                "get_cleanup_status": self._tool_get_cleanup_status,
                # Code Knowledge Graph (New)
                "init_code_knowledge_graph": self._tool_init_code_knowledge_graph,
                "get_code_context": self._tool_get_code_context,
            }

            handler = handlers.get(name)
            if handler:
                result = await handler(arguments)
            else:
                result = {"success": False, "error": f"Unknown tool: {name}"}

            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        except Exception as e:
            self.logger.error(f"Tool call failed: {name}, error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
            )]

    # ==================== Adaptive Learning Tools (New) ====================

    async def _tool_track_user_behavior(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Track user behavior"""
        session_id = args.get("session_id")
        action_type = args.get("action_type")
        target_memory_id = args.get("target_memory_id")
        query_text = args.get("query_text")
        response_relevance = args.get("response_relevance", 0.0)
        user_feedback = args.get("user_feedback")
        interaction_duration = args.get("interaction_duration", 0)
        project_path = args.get("project_path")

        if not session_id or not action_type:
            return {"success": False, "error": "session_id and action_type are required"}

        try:
            # Check if adaptive learning components exist
            if not hasattr(self.memory_engine, 'behavior_analyzer'):
                return {
                    "success": False,
                    "error": "Adaptive learning components not available. Please ensure adaptive learning migration is applied."
                }

            # Create user behavior object
            from ..intelligence.behavior_analyzer import UserBehavior, ActionType, FeedbackType

            # Convert enum types
            action_enum = ActionType(action_type)
            feedback_enum = FeedbackType(user_feedback) if user_feedback else None

            behavior = UserBehavior(
                session_id=session_id,
                action_type=action_enum,
                target_memory_id=target_memory_id,
                query_text=query_text,
                response_relevance=response_relevance,
                user_feedback=feedback_enum,
                interaction_duration=interaction_duration,
                project_path=project_path
            )

            # Track behavior
            success = self.memory_engine.behavior_analyzer.track_user_behavior(behavior)

            if success:
                return {
                    "success": True,
                    "message": "User behavior tracked successfully",
                    "data": {
                        "session_id": session_id,
                        "action_type": action_type,
                        "tracked_at": behavior.timestamp.isoformat()
                    }
                }
            else:
                return {"success": False, "error": "Failed to track user behavior"}

        except Exception as e:
            self.logger.error(f"Failed to track user behavior: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_personalized_recommendations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized recommendations"""
        user_session = args.get("user_session")
        query_context = args.get("query_context")
        recommendation_type = args.get("recommendation_type", "hybrid")
        project_path = args.get("project_path")
        limit = args.get("limit", 5)
        include_explanation = args.get("include_explanation", True)

        if not user_session:
            return {"success": False, "error": "user_session is required"}

        try:
            # Check if recommender engine exists
            if not hasattr(self.memory_engine, 'recommender'):
                return {
                    "success": False,
                    "error": "Personalized recommender not available. Please ensure adaptive learning components are initialized."
                }

            # Call corresponding method based on recommendation type
            recommendations = []

            if recommendation_type == "collaborative":
                recommendations = self.memory_engine.recommender.collaborative_filtering_recommend(
                    user_session, limit=limit
                )
            elif recommendation_type == "content_based":
                if query_context:
                    recommendations = self.memory_engine.recommender.content_based_recommend(
                        query_context, limit=limit
                    )
                else:
                    return {"success": False, "error": "query_context required for content_based recommendations"}
            elif recommendation_type == "causal":
                if query_context:
                    recommendations = self.memory_engine.recommender.causal_based_recommend(
                        query_context, limit=limit
                    )
                else:
                    return {"success": False, "error": "query_context required for causal recommendations"}
            elif recommendation_type == "predictive":
                recommendations = self.memory_engine.recommender.predictive_recommend(
                    user_session, limit=limit
                )
            elif recommendation_type == "hybrid":
                recommendations = self.memory_engine.recommender.hybrid_recommend(
                    user_session, query_context or "", limit=limit
                )
            else:
                return {"success": False, "error": f"Unknown recommendation type: {recommendation_type}"}

            # Format recommendation results
            results = []
            for rec in recommendations:
                result = {
                    "memory_id": rec.get("memory_id"),
                    "confidence": rec.get("confidence", 0.0),
                    "recommendation_type": rec.get("recommendation_type", recommendation_type)
                }

                if include_explanation:
                    result["explanation"] = rec.get("explanation", "")
                    result["reasoning"] = rec.get("reasoning", "")

                results.append(result)

            return {
                "success": True,
                "user_session": user_session,
                "recommendation_type": recommendation_type,
                "recommendations": results,
                "total_count": len(results),
                "message": f"Generated {len(results)} personalized recommendations"
            }

        except Exception as e:
            self.logger.error(f"Failed to get personalized recommendations: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_analyze_user_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user patterns"""
        user_session = args.get("user_session")
        analysis_type = args.get("analysis_type", "query_patterns")
        time_window_days = args.get("time_window_days", 30)
        project_path = args.get("project_path")

        try:
            # Check if behavior analyzer exists
            if not hasattr(self.memory_engine, 'behavior_analyzer'):
                return {
                    "success": False,
                    "error": "Behavior analyzer not available. Please ensure adaptive learning components are initialized."
                }

            analyzer = self.memory_engine.behavior_analyzer
            results = {}

            if analysis_type == "query_patterns":
                patterns = analyzer.analyze_query_patterns(user_session, time_window_days)
                results = {
                    "patterns": [
                        {
                            "pattern_type": p.pattern_type,
                            "pattern_value": p.pattern_value,
                            "frequency": p.frequency,
                            "success_rate": p.success_rate,
                            "avg_relevance": p.avg_relevance
                        }
                        for p in patterns
                    ],
                    "total_patterns": len(patterns)
                }

            elif analysis_type == "attention_weights":
                # Get common topics
                topics = ["Python", "FastAPI", "Database", "Machine Learning", "Web Development", "Performance Optimization", "Security", "Testing"]
                weights = analyzer.calculate_attention_weights(topics, project_path)
                results = {
                    "attention_weights": [
                        {"topic": topic, "weight": weight}
                        for topic, weight in weights.items()
                    ],
                    "total_topics": len(weights)
                }

            elif analysis_type == "preference_drift":
                # Analyze preference changes for main topics
                main_topics = ["Python", "FastAPI", "Database"]
                drifts = []
                for topic in main_topics:
                    drift = analyzer.detect_preference_drift(topic, project_path, time_window_days)
                    if drift:
                        drifts.append({
                            "topic": drift.topic,
                            "old_weight": drift.old_weight,
                            "new_weight": drift.new_weight,
                            "drift_rate": drift.drift_rate,
                            "confidence": drift.confidence,
                            "evidence_count": drift.evidence_count
                        })
                results = {
                    "preference_drifts": drifts,
                    "total_drifts": len(drifts)
                }

            elif analysis_type == "learning_effectiveness":
                stats = analyzer.get_behavior_statistics(user_session, time_window_days)
                results = {
                    "behavior_statistics": stats,
                    "effectiveness_metrics": {
                        "avg_relevance": stats.get("average_relevance", 0.0),
                        "total_interactions": stats.get("total_behaviors", 0),
                        "feedback_ratio": len(stats.get("feedback_distribution", {})) / max(stats.get("total_behaviors", 1), 1)
                    }
                }

            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}

            return {
                "success": True,
                "user_session": user_session,
                "analysis_type": analysis_type,
                "time_window_days": time_window_days,
                "project_path": project_path,
                "results": results,
                "message": f"User pattern analysis completed for {analysis_type}"
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze user patterns: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_adaptive_importance_scoring(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive importance scoring"""
        content = args.get("content")
        memory_id = args.get("memory_id")
        user_session = args.get("user_session")
        category = args.get("category")
        project_path = args.get("project_path")
        feedback_data = args.get("feedback_data", [])

        if not content and not memory_id:
            return {"success": False, "error": "Either content or memory_id is required"}

        try:
            # Check if adaptive scorer exists
            if not hasattr(self.memory_engine, 'adaptive_scorer'):
                return {
                    "success": False,
                    "error": "Adaptive scorer not available. Please ensure adaptive learning components are initialized."
                }

            scorer = self.memory_engine.adaptive_scorer

            # Process feedback data learning
            if feedback_data:
                for feedback in feedback_data:
                    scorer.learn_from_feedback(
                        feedback.get("memory_id"),
                        feedback.get("feedback_score"),
                        feedback.get("feedback_type")
                    )

            # Calculate importance score
            if content:
                # New content scoring
                importance_score = scorer.calculate_importance_score(content, category or "general")

                return {
                    "success": True,
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "importance_score": round(importance_score, 3),
                    "category": category,
                    "user_session": user_session,
                    "message": f"Adaptive importance score calculated: {importance_score:.3f}"
                }

            elif memory_id:
                # Re-score existing memory
                memory = self.memory_engine.get_memory(memory_id)
                if not memory:
                    return {"success": False, "error": f"Memory {memory_id} not found"}

                new_score = scorer.calculate_importance_score(memory.content, memory.category or "general")
                old_score = memory.importance

                # Update memory importance
                memory.importance = new_score
                self.memory_engine.memory_repo.update_memory(memory)

                return {
                    "success": True,
                    "memory_id": memory_id,
                    "old_importance": round(old_score, 3),
                    "new_importance": round(new_score, 3),
                    "improvement": round(new_score - old_score, 3),
                    "message": f"Memory {memory_id} importance updated from {old_score:.3f} to {new_score:.3f}"
                }

        except Exception as e:
            self.logger.error(f"Failed to adaptive importance scoring: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_learning_insights(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get learning insights"""
        insight_type = args.get("insight_type", "recommendation_performance")
        user_session = args.get("user_session")
        project_path = args.get("project_path")
        time_period_days = args.get("time_period_days", 7)

        try:
            insights = {}

            if insight_type == "recommendation_performance":
                # Recommendation performance insights
                if hasattr(self.memory_engine, 'recommender'):
                    # Get recommendation history statistics
                    stats = self.memory_engine.recommender.get_recommendation_statistics(user_session)
                    insights = {
                        "recommendation_stats": stats,
                        "performance_summary": {
                            "total_recommendations": stats.get("total_recommendations", 0),
                            "avg_confidence": stats.get("avg_confidence", 0.0),
                            "recommendation_types": stats.get("recommendation_type_distribution", {}),
                            "user_feedback_rate": stats.get("feedback_rate", 0.0)
                        }
                    }
                else:
                    insights = {"error": "Recommender not available"}

            elif insight_type == "user_engagement":
                # User engagement insights
                if hasattr(self.memory_engine, 'behavior_analyzer'):
                    stats = self.memory_engine.behavior_analyzer.get_behavior_statistics(user_session, time_period_days)
                    insights = {
                        "engagement_stats": stats,
                        "engagement_summary": {
                            "total_interactions": stats.get("total_behaviors", 0),
                            "avg_relevance": stats.get("average_relevance", 0.0),
                            "action_distribution": stats.get("action_type_distribution", {}),
                            "feedback_quality": stats.get("feedback_distribution", {})
                        }
                    }
                else:
                    insights = {"error": "Behavior analyzer not available"}

            elif insight_type == "learning_progress":
                # Learning progress insights
                insights = {
                    "learning_metrics": {
                        "total_memories": self.memory_engine.get_statistics().get("total_memories", 0),
                        "adaptive_components_active": {
                            "behavior_analyzer": hasattr(self.memory_engine, 'behavior_analyzer'),
                            "adaptive_scorer": hasattr(self.memory_engine, 'adaptive_scorer'),
                            "recommender": hasattr(self.memory_engine, 'recommender')
                        },
                        "system_maturity": "developing" if time_period_days < 30 else "mature"
                    }
                }

            elif insight_type == "knowledge_gaps":
                # Knowledge gap insights
                if hasattr(self.memory_engine, 'recommender'):
                    gaps = self.memory_engine.recommender.analyze_knowledge_gaps(user_session)
                    insights = {
                        "knowledge_gaps": gaps,
                        "gap_summary": {
                            "total_gaps": len(gaps),
                            "priority_gaps": [gap for gap in gaps if gap.get("priority", 0) > 0.7],
                            "recommended_actions": [gap.get("recommendation", "") for gap in gaps[:3]]
                        }
                    }
                else:
                    insights = {"error": "Recommender not available for knowledge gap analysis"}

            else:
                return {"success": False, "error": f"Unknown insight type: {insight_type}"}

            return {
                "success": True,
                "insight_type": insight_type,
                "user_session": user_session,
                "project_path": project_path,
                "time_period_days": time_period_days,
                "insights": insights,
                "generated_at": datetime.now().isoformat(),
                "message": f"Learning insights generated for {insight_type}"
            }

        except Exception as e:
            self.logger.error(f"Failed to get learning insights: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Reasoning Engine Tools (New) ====================

    async def _tool_analyze_causal_relations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causal relations"""
        project_path = args.get("project_path")
        query = args.get("query")
        limit = args.get("limit", 10)

        try:
            # Get related memories
            if project_path:
                memories = self.memory_engine.get_project_memories(project_path, limit=100)
            else:
                memories = self.memory_engine.memory_repo.search_memories(limit=100)

            if not memories:
                return {
                    "success": True,
                    "causal_relations": [],
                    "message": "No memories found for causal analysis"
                }

            # Use causal analysis functionality
            causal_relations = self.memory_engine.analyze_causal_relations(memories)

            # If there's a query, filter relevant results
            if query:
                filtered_relations = []
                for relation in causal_relations[:limit]:
                    # Simple relevance check
                    if query.lower() in relation.get('evidence', '').lower():
                        filtered_relations.append(relation)
                causal_relations = filtered_relations

            results = []
            for relation in causal_relations[:limit]:
                results.append({
                    "cause_memory_id": relation.get('cause_memory_id'),
                    "effect_memory_id": relation.get('effect_memory_id'),
                    "causal_type": relation.get('causal_type'),
                    "strength": relation.get('strength', 0.0),
                    "confidence": relation.get('confidence', 0.0),
                    "evidence": relation.get('evidence', []),
                    "pattern_matched": relation.get('pattern_matched', '')
                })

            return {
                "success": True,
                "project_path": project_path,
                "query": query,
                "causal_relations": results,
                "total_found": len(results),
                "message": f"Found {len(results)} causal relations"
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze causal relations: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_predict_next_needs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict next needs"""
        project_path = args.get("project_path")
        current_context = args.get("current_context")
        prediction_types = args.get("prediction_types", ["next_need"])
        limit = args.get("limit", 5)

        try:
            # Build prediction context
            context = {"project_path": project_path}
            if current_context:
                context["current_context"] = current_context

            # Use prediction functionality
            predictions = self.memory_engine.predict_next_needs(context)

            # Filter prediction types
            if prediction_types:
                filtered_predictions = []
                for pred in predictions:
                    if pred.get('prediction_type') in prediction_types:
                        filtered_predictions.append(pred)
                predictions = filtered_predictions

            results = []
            for pred in predictions[:limit]:
                results.append({
                    "prediction_type": pred.get('prediction_type'),
                    "content": pred.get('content'),
                    "confidence": pred.get('confidence', 0.0),
                    "reasoning": pred.get('reasoning', ''),
                    "evidence": pred.get('evidence', []),
                    "related_memories": pred.get('related_memories', [])
                })

            return {
                "success": True,
                "project_path": project_path,
                "current_context": current_context,
                "prediction_types": prediction_types,
                "predictions": results,
                "total_found": len(results),
                "message": f"Generated {len(results)} predictions"
            }

        except Exception as e:
            self.logger.error(f"Failed to predict next needs: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_build_knowledge_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Build knowledge graph"""
        project_path = args.get("project_path")
        concept_types = args.get("concept_types", ["technology", "methodology", "domain"])
        min_frequency = args.get("min_frequency", 2)
        include_relations = args.get("include_relations", True)

        try:
            # Get related memories
            if project_path:
                memories = self.memory_engine.get_project_memories(project_path, limit=200)
            else:
                memories = self.memory_engine.memory_repo.search_memories(limit=200)

            if not memories:
                return {
                    "success": True,
                    "concepts": [],
                    "relations": [],
                    "message": "No memories found for knowledge graph construction"
                }

            # Build knowledge graph
            graph_data = self.memory_engine.build_concept_graph(memories)

            # Filter concepts
            concepts = []
            for concept in graph_data.get('concepts', []):
                if (concept.get('frequency', 0) >= min_frequency and
                    concept.get('concept_type') in concept_types):
                    concepts.append({
                        "name": concept.get('name'),
                        "concept_type": concept.get('concept_type'),
                        "frequency": concept.get('frequency'),
                        "importance": concept.get('importance', 0.0),
                        "related_memories": concept.get('related_memories', [])
                    })

            # Include relation information
            relations = []
            if include_relations:
                for relation in graph_data.get('relations', []):
                    relations.append({
                        "source_concept": relation.get('source_concept'),
                        "target_concept": relation.get('target_concept'),
                        "relation_type": relation.get('relation_type'),
                        "weight": relation.get('weight', 0.0),
                        "evidence": relation.get('evidence', [])
                    })

            return {
                "success": True,
                "project_path": project_path,
                "concept_types": concept_types,
                "concepts": concepts,
                "relations": relations if include_relations else [],
                "stats": {
                    "total_concepts": len(concepts),
                    "total_relations": len(relations),
                    "min_frequency": min_frequency
                },
                "message": f"Built knowledge graph with {len(concepts)} concepts and {len(relations)} relations"
            }

        except Exception as e:
            self.logger.error(f"Failed to build knowledge graph: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_comprehensive_reasoning(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive reasoning analysis"""
        project_path = args.get("project_path")
        focus_area = args.get("focus_area")
        analysis_depth = args.get("analysis_depth", "standard")
        include_insights = args.get("include_insights", True)

        try:
            # Get related memories
            if project_path:
                memories = self.memory_engine.get_project_memories(project_path, limit=150)
            else:
                memories = self.memory_engine.memory_repo.search_memories(limit=150)

            if not memories:
                return {
                    "success": True,
                    "analysis": {},
                    "message": "No memories found for comprehensive reasoning"
                }

            # Execute comprehensive reasoning analysis
            analysis_results = {}

            # 1. Causal analysis
            try:
                causal_relations = self.memory_engine.analyze_causal_relations(memories)
                analysis_results["causal_analysis"] = {
                    "relations_count": len(causal_relations),
                    "top_relations": causal_relations[:5],
                    "summary": f"Found {len(causal_relations)} causal relationships"
                }
            except Exception as e:
                analysis_results["causal_analysis"] = {"error": str(e)}

            # 2. Predictive analysis
            try:
                context = {"project_path": project_path}
                if focus_area:
                    context["focus_area"] = focus_area
                predictions = self.memory_engine.predict_next_needs(context)
                analysis_results["predictive_analysis"] = {
                    "predictions_count": len(predictions),
                    "top_predictions": predictions[:3],
                    "summary": f"Generated {len(predictions)} predictions"
                }
            except Exception as e:
                analysis_results["predictive_analysis"] = {"error": str(e)}

            # 3. Conceptual analysis
            try:
                graph_data = self.memory_engine.build_concept_graph(memories)
                concepts = graph_data.get('concepts', [])
                relations = graph_data.get('relations', [])
                analysis_results["conceptual_analysis"] = {
                    "concepts_count": len(concepts),
                    "relations_count": len(relations),
                    "top_concepts": concepts[:5],
                    "summary": f"Identified {len(concepts)} concepts with {len(relations)} relationships"
                }
            except Exception as e:
                analysis_results["conceptual_analysis"] = {"error": str(e)}

            # 4. Integrated insights
            integrated_insights = []
            if include_insights:
                try:
                    # Generate insights based on analysis results
                    if analysis_results.get("causal_analysis", {}).get("relations_count", 0) > 0:
                        integrated_insights.append({
                            "type": "causal_insight",
                            "title": "Causal Relationship Insights",
                            "description": f"Discovered {analysis_results['causal_analysis']['relations_count']} causal relationships, showing connections between decisions and outcomes",
                            "confidence": 0.8
                        })

                    if analysis_results.get("predictive_analysis", {}).get("predictions_count", 0) > 0:
                        integrated_insights.append({
                            "type": "predictive_insight",
                            "title": "Predictive Insights",
                            "description": f"Generated {analysis_results['predictive_analysis']['predictions_count']} prediction recommendations based on historical patterns",
                            "confidence": 0.7
                        })

                    if analysis_results.get("conceptual_analysis", {}).get("concepts_count", 0) > 0:
                        integrated_insights.append({
                            "type": "conceptual_insight",
                            "title": "Conceptual Insights",
                            "description": f"Identified {analysis_results['conceptual_analysis']['concepts_count']} core concepts and their associations",
                            "confidence": 0.9
                        })

                except Exception as e:
                    integrated_insights.append({"error": str(e)})

            return {
                "success": True,
                "project_path": project_path,
                "focus_area": focus_area,
                "analysis_depth": analysis_depth,
                "analysis": analysis_results,
                "integrated_insights": integrated_insights,
                "stats": {
                    "total_memories_analyzed": len(memories),
                    "analysis_types_completed": len([k for k in analysis_results.keys() if "error" not in analysis_results[k]]),
                    "insights_generated": len(integrated_insights)
                },
                "message": f"Comprehensive reasoning analysis completed for {len(memories)} memories"
            }

        except Exception as e:
            self.logger.error(f"Comprehensive reasoning analysis failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_query_reasoning(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query reasoning"""
        query = args.get("query")
        reasoning_type = args.get("reasoning_type", "integrated")
        project_path = args.get("project_path")
        confidence_threshold = args.get("confidence_threshold", 0.3)

        if not query:
            return {"success": False, "error": "query is required"}

        try:
            # Get related memories
            if project_path:
                memories = self.memory_engine.get_project_memories(project_path, limit=100)
            else:
                # Use semantic search to find related memories
                all_memories = self.memory_engine.memory_repo.search_memories(limit=200)
                memory_dicts = [m.to_dict() for m in all_memories]
                search_results = self.search_engine.search_memories(query, memory_dicts, limit=50, min_similarity=0.2)
                memories = [self.memory_engine.get_memory(m[0]['id']) for m, _ in search_results if m[0].get('id')]
                memories = [m for m in memories if m is not None]

            if not memories:
                return {
                    "success": True,
                    "query": query,
                    "reasoning_type": reasoning_type,
                    "analysis": "No relevant memories found for this query",
                    "confidence": 0.0,
                    "evidence": []
                }

            # Execute analysis based on reasoning type
            analysis_result = {}
            confidence = 0.0
            evidence = []

            if reasoning_type in ["causal", "integrated"]:
                try:
                    causal_relations = self.memory_engine.analyze_causal_relations(memories)
                    relevant_relations = [r for r in causal_relations if query.lower() in str(r).lower()]
                    if relevant_relations:
                        analysis_result["causal"] = {
                            "relations": relevant_relations[:3],
                            "summary": f"Found {len(relevant_relations)} causal relations related to your query"
                        }
                        confidence = max(confidence, 0.7)
                        evidence.extend([f"Causal relation: {r.get('pattern_matched', '')}" for r in relevant_relations[:2]])
                except Exception as e:
                    analysis_result["causal"] = {"error": str(e)}

            if reasoning_type in ["predictive", "integrated"]:
                try:
                    context = {"project_path": project_path, "query": query}
                    predictions = self.memory_engine.predict_next_needs(context)
                    if predictions:
                        analysis_result["predictive"] = {
                            "predictions": predictions[:3],
                            "summary": f"Generated {len(predictions)} predictions based on your query"
                        }
                        confidence = max(confidence, 0.6)
                        evidence.extend([f"Prediction: {p.get('content', '')[:50]}..." for p in predictions[:2]])
                except Exception as e:
                    analysis_result["predictive"] = {"error": str(e)}

            if reasoning_type in ["conceptual", "integrated"]:
                try:
                    graph_data = self.memory_engine.build_concept_graph(memories)
                    concepts = graph_data.get('concepts', [])
                    # Find concepts related to the query
                    relevant_concepts = [c for c in concepts if query.lower() in c.get('name', '').lower()]
                    if relevant_concepts:
                        analysis_result["conceptual"] = {
                            "concepts": relevant_concepts[:5],
                            "summary": f"Found {len(relevant_concepts)} concepts related to your query"
                        }
                        confidence = max(confidence, 0.8)
                        evidence.extend([f"Concept: {c.get('name', '')}" for c in relevant_concepts[:2]])
                except Exception as e:
                    analysis_result["conceptual"] = {"error": str(e)}

            # Generate comprehensive analysis text
            analysis_text = f"Query: {query}\n\n"
            for analysis_type, result in analysis_result.items():
                if "error" not in result:
                    analysis_text += f"{analysis_type.title()} Analysis:\n{result.get('summary', '')}\n\n"

            # Check confidence threshold
            if confidence < confidence_threshold:
                analysis_text += f"Note: Analysis confidence ({confidence:.2f}) is below threshold ({confidence_threshold}). Results may be less reliable."

            return {
                "success": True,
                "query": query,
                "reasoning_type": reasoning_type,
                "project_path": project_path,
                "analysis": analysis_text.strip(),
                "confidence": round(confidence, 2),
                "evidence": evidence,
                "detailed_results": analysis_result,
                "stats": {
                    "memories_analyzed": len(memories),
                    "analysis_types": list(analysis_result.keys()),
                    "confidence_threshold": confidence_threshold
                },
                "message": f"Query reasoning completed with {confidence:.2f} confidence"
            }

        except Exception as e:
            self.logger.error(f"Query reasoning failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Intelligent Analysis Tools ====================

    async def _tool_analyze_conversation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation and automatically extract memories

        This is the core tool for implementing "intelligent memory system"
        """
        conversation = args.get("conversation", "")
        context = args.get("context")
        auto_store = args.get("auto_store", True)

        if not conversation:
            return {"success": False, "error": "conversation is required"}

        # Use MemoryExtractor to extract memories
        extracted_memories = self.extractor.extract_memories(conversation, context)

        if not extracted_memories:
            return {
                "success": True,
                "extracted_count": 0,
                "message": "No valuable memories found in conversation"
            }

        results = []
        if auto_store:
            for memory in extracted_memories:
                memory_id = self.memory_engine.store_memory(
                    content=memory.content,
                    importance=memory.importance,
                    memory_type=memory.memory_type,
                    category=memory.category
                )
                results.append({
                    "id": memory_id,
                    "type": memory.memory_type,
                    "importance": round(memory.importance, 2),
                    "category": memory.category,
                    "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    "keywords": memory.keywords[:5] if memory.keywords else [],
                    "confidence": round(memory.confidence, 2)
                })

        return {
            "success": True,
            "extracted_count": len(extracted_memories),
            "stored": auto_store,
            "memories": results,
            "message": f"Extracted {len(extracted_memories)} memories from conversation"
        }

    async def _tool_search_semantic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Semantic search - using AI for similarity matching
        """
        query = args.get("query", "")
        limit = args.get("limit", 10)
        min_similarity = args.get("min_similarity", 0.3)
        memory_type = args.get("type")

        if not query:
            return {"success": False, "error": "query is required"}

        # Get candidate memories
        memories = self.memory_engine.memory_repo.search_memories(
            memory_type=memory_type,
            limit=1000  # Get enough candidates
        )

        if not memories:
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": "No memories to search"
            }

        # Convert to dictionary format
        memory_dicts = [m.to_dict() for m in memories]

        # Use semantic search engine
        search_results = self.search_engine.search_memories(
            query, memory_dicts, limit, min_similarity
        )

        results = []
        for memory_dict, similarity in search_results:
            results.append({
                "id": memory_dict.get("id"),
                "content": memory_dict.get("content", "")[:150],
                "type": memory_dict.get("memory_type"),
                "category": memory_dict.get("category"),
                "importance": memory_dict.get("importance"),
                "similarity": round(similarity, 3)
            })

        return {
            "success": True,
            "query": query,
            "search_type": "semantic",
            "result_count": len(results),
            "results": results
        }

    # ==================== Core Tools ====================

    async def _tool_get_preferences(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get user preferences"""
        category = args.get("category")

        memories = self.memory_engine.memory_repo.search_memories(
            memory_type="preference",
            limit=50
        )

        if category:
            memories = [m for m in memories if m.category == category]

        preferences = [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "category": m.category,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in memories
        ]

        return {
            "success": True,
            "data": preferences,
            "count": len(preferences),
            "message": f"Found {len(preferences)} preferences"
        }

    async def _tool_search_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Keyword search"""
        query = args.get("query", "")
        limit = args.get("limit", 10)
        memory_type = args.get("type")

        memories = self.memory_engine.recall_memories(
            query=query,
            limit=limit,
            memory_type=memory_type
        )

        results = [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "type": m.memory_type,
                "category": m.category,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in memories
        ]

        return {
            "success": True,
            "query": query,
            "search_type": "keyword",
            "result_count": len(results),
            "results": results
        }

    async def _tool_remember(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store new memory"""
        content = args.get("content")
        importance = args.get("importance", 0.5)
        memory_type = args.get("type", "conversation")
        category = args.get("category")

        if not content:
            return {"success": False, "error": "content is required"}

        memory_id = self.memory_engine.store_memory(
            content=content,
            importance=importance,
            memory_type=memory_type,
            category=category
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "type": memory_type,
            "importance": importance,
            "message": f"Memory stored with ID: {memory_id}"
        }

    # ==================== Context Tools ====================

    async def _tool_get_project_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get project context"""
        project_path = args.get("project_path")

        if not project_path:
            return {"success": False, "error": "project_path is required"}

        memories = self.memory_engine.get_project_memories(project_path, limit=30)

        results = [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "type": m.memory_type,
                "category": m.category
            }
            for m in memories
        ]

        return {
            "success": True,
            "project_path": project_path,
            "data": results,
            "count": len(results)
        }

    async def _tool_get_recent_decisions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get recent decisions"""
        limit = args.get("limit", 10)

        memories = self.memory_engine.memory_repo.search_memories(
            memory_type="decision",
            limit=limit
        )

        results = [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in memories
        ]

        return {
            "success": True,
            "data": results,
            "count": len(results)
        }

    async def _tool_get_important_facts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get important facts"""
        min_importance = args.get("min_importance", 0.7)
        limit = args.get("limit", 10)

        memories = self.memory_engine.memory_repo.search_memories(
            min_importance=min_importance,
            limit=limit
        )

        results = [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "type": m.memory_type,
                "category": m.category
            }
            for m in memories
        ]

        return {
            "success": True,
            "data": results,
            "count": len(results)
        }

    async def _tool_get_quick_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get quick context - enhanced version, auto initializes new projects with scan_project"""
        # Dynamically get current working directory if project_path not specified
        # This ensures correct project detection when MCP server is started from a different directory
        project_path = args.get("project_path")
        if not project_path:
            project_path = str(Path.cwd())

        # Normalize path for consistent comparison
        if project_path:
            project_path = str(Path(project_path).resolve())

        try:
            # Format global hot data from memory partition
            context = self.memory_partition.get_context_for_project(project_path)

            preferences = []
            decisions = []
            important_facts = []

            for memory in context['global_hot']:
                if memory.memory_type == 'preference':
                    preferences.append({
                        "content": memory.content,
                        "category": memory.category,
                        "lfu_score": memory.get_lfu_counter()
                    })
                elif memory.memory_type == 'decision':
                    decisions.append({
                        "content": memory.content,
                        "importance": memory.importance,
                        "lfu_score": memory.get_lfu_counter()
                    })
                elif memory.importance >= 0.8:
                    important_facts.append({
                        "content": memory.content,
                        "importance": memory.importance,
                        "lfu_score": memory.get_lfu_counter()
                    })

            # Build project context
            project_context = context.get('project_context', {})

            # Check if project is initialized: search for memory with category="project_init"
            # Use category parameter for exact match, project_path filter to limit to current project
            existing_init = self.memory_engine.memory_repo.search_memories(
                query="",
                memory_type="project",
                category="project_init",
                project_path=project_path,
                limit=1
            )

            has_initialized = len(existing_init) > 0

            self.logger.info(f"Project init check: project_path={project_path}, has_initialized={has_initialized}")

            # If not initialized, auto-call scan_project for full scan
            if not has_initialized:
                self.logger.info(f"Project not initialized, auto-scanning: {project_path}")
                scan_result = await self._tool_scan_project({
                    "project_path": project_path,
                    "quick_scan": False  # Use full scan
                })

                if scan_result.get("success"):
                    # Also initialize code knowledge graph
                    self.logger.info(f"Initializing code knowledge graph for: {project_path}")
                    kg_result = await self._tool_init_code_knowledge_graph({
                        "project_path": project_path
                    })

                    # Generate project workflow overview for human understanding
                    workflow_overview = self._generate_project_workflow(
                        project_path,
                        scan_result,
                        kg_result
                    )

                    # Store workflow as memory for future reference
                    from ..storage.models import Memory
                    workflow_memory = Memory(
                        content=workflow_overview,
                        memory_type="project",
                        category="project_workflow",
                        importance=0.95,
                        metadata={
                            "project_path": project_path,
                            "project_name": scan_result.get("project_name"),
                            "generated_at": datetime.now().isoformat()
                        }
                    )
                    self.memory_engine.memory_repo.create_memory(workflow_memory)
                    self.logger.info(f"Stored project workflow memory for: {project_path}")

                    # Re-fetch project_context since scan_project may have updated it
                    context = self.memory_partition.get_context_for_project(project_path)
                    project_context = context.get('project_context', {})

                    result = {
                        "success": True,
                        "preferences": preferences,
                        "recent_decisions": decisions,
                        "important_facts": important_facts,
                        "project_context": project_context,
                        "workflow_overview": workflow_overview,
                        "cache_version": self.memory_engine._cache_version,
                        "sync_stats": {
                            "database_watcher": self.memory_engine._db_watcher.get_stats() if self.memory_engine._db_watcher else None,
                            "event_bus": self.memory_engine._event_bus.get_stats() if self.memory_engine._event_bus else None
                        },
                        "stats": {
                            "total_memories": self.memory_engine.get_statistics()["total_memories"],
                            "global_hot_count": context['stats']['global_hot_count'],
                            "project_hot_count": context['stats']['project_hot_count'],
                            "project_path": project_path
                        },
                        "project_init": {
                            "initialized": True,
                            "auto_scanned": True,
                            "project_name": scan_result.get("project_name"),
                            "project_type": scan_result.get("project_type"),
                            "knowledge_graph": {
                                "initialized": kg_result.get("success", False),
                                "statistics": kg_result.get("statistics", {})
                            }
                        }
                    }
                else:
                    result = {
                        "success": True,
                        "preferences": preferences,
                        "recent_decisions": decisions,
                        "important_facts": important_facts,
                        "project_context": project_context,
                        "cache_version": self.memory_engine._cache_version,
                        "sync_stats": {
                            "database_watcher": self.memory_engine._db_watcher.get_stats() if self.memory_engine._db_watcher else None,
                            "event_bus": self.memory_engine._event_bus.get_stats() if self.memory_engine._event_bus else None
                        },
                        "stats": {
                            "total_memories": self.memory_engine.get_statistics()["total_memories"],
                            "global_hot_count": context['stats']['global_hot_count'],
                            "project_hot_count": context['stats']['project_hot_count'],
                            "project_path": project_path
                        },
                        "project_init": {
                            "initialized": False,
                            "auto_scan_failed": True,
                            "error": scan_result.get("error")
                        }
                    }
            else:
                # Already initialized, fetch workflow from memory
                workflow_memories = self.memory_engine.memory_repo.search_memories(
                    query="",
                    memory_type="project",
                    category="project_workflow",
                    project_path=project_path,
                    limit=1
                )

                workflow_overview = None
                if workflow_memories:
                    workflow_overview = workflow_memories[0].content
                    self.logger.info(f"Loaded project workflow from memory")

                result = {
                    "success": True,
                    "preferences": preferences,
                    "recent_decisions": decisions,
                    "important_facts": important_facts,
                    "project_context": project_context,
                    "workflow_overview": workflow_overview,
                    "cache_version": self.memory_engine._cache_version,
                    "sync_stats": {
                        "database_watcher": self.memory_engine._db_watcher.get_stats() if self.memory_engine._db_watcher else None,
                        "event_bus": self.memory_engine._event_bus.get_stats() if self.memory_engine._event_bus else None
                    },
                    "stats": {
                        "total_memories": self.memory_engine.get_statistics()["total_memories"],
                        "global_hot_count": context['stats']['global_hot_count'],
                        "project_hot_count": context['stats']['project_hot_count'],
                        "project_path": project_path
                    },
                    "project_init": {"initialized": True, "from_cache": True}
                }

            return result

        except Exception as e:
            self.logger.error(f"Get quick context failed: {e}")
            # Fallback to original implementation
            return await self._tool_get_quick_context_fallback()

    def _generate_project_workflow(
        self,
        project_path: str,
        scan_result: Dict[str, Any],
        kg_result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable project workflow overview

        This creates a comprehensive project understanding document
        that helps both humans and AI understand the project structure.

        Args:
            project_path: Project path
            scan_result: Result from scan_project
            kg_result: Result from init_code_knowledge_graph

        Returns:
            Markdown-formatted workflow overview
        """
        project_name = scan_result.get("project_name", Path(project_path).name)
        project_type = scan_result.get("project_type", "unknown")
        tech_stack = scan_result.get("tech_stack", [])

        # Handle languages - could be dict or list
        languages_raw = scan_result.get("languages", {})
        if isinstance(languages_raw, dict):
            languages_str = ', '.join(f"{k} ({v}%)" for k, v in languages_raw.items())
        elif isinstance(languages_raw, list):
            languages_str = ', '.join(languages_raw)
        else:
            languages_str = 'Pending analysis'

        # Handle kg_stats - could be dict or missing
        kg_stats = kg_result.get("statistics", {})
        if not isinstance(kg_stats, dict):
            kg_stats = {}

        workflow = f"""# {project_name} Project Workflow

## Project Overview

| Property | Value |
|----------|-------|
| **Project Path** | `{project_path}` |
| **Project Type** | {project_type} |
| **Tech Stack** | {', '.join(tech_stack) if tech_stack else 'Pending analysis'} |
| **Languages** | {languages_str} |

## Code Structure

| Type | Count |
|------|-------|
| Source Files | {kg_stats.get('files_processed', 0)} |
| Classes | {kg_stats.get('classes_added', 0)} |
| Functions | {kg_stats.get('functions_added', 0)} |
| API Endpoints | {kg_stats.get('api_endpoints_added', 0)} |

## hibro Workflow

### First Conversation (Project Initialization)

```
1. SessionStart Hook triggered
   │
   ▼
2. get_quick_context detects new project
   │
   ▼
3. Auto-execute scan_project
   ├─ Scan file structure
   ├─ Detect project type
   └─ Analyze tech stack
   │
   ▼
4. Auto-execute init_code_knowledge_graph
   ├─ Parse all source code
   ├─ Extract classes and functions
   └─ Build knowledge graph
   │
   ▼
5. Store project memories
   ├─ project_init memory
   ├─ code_kg_init memory
   └─ project_workflow memory (this document)
   │
   ▼
6. Return full context to Claude
```

### Subsequent Conversations

```
1. SessionStart Hook triggered
   │
   ▼
2. get_quick_context detects initialized project
   │
   ▼
3. Load project memories directly
   ├─ project_workflow (this document)
   ├─ User preferences
   └─ Recent decisions
   │
   ▼
4. Return context (no re-scan needed)
```

## Key Files

*Based on knowledge graph analysis, here are the most important files:*

{self._get_key_files_markdown(project_path)}

## Using hibro Tools

### Query Code Structure
```
mcp__hibro__get_code_context(
    project_path="{project_path}",
    detail_level="medium"
)
```

### Search Code
```
mcp__hibro__get_code_context(
    project_path="{project_path}",
    search_query="class or function name"
)
```

### Rescan Project
```
mcp__hibro__init_code_knowledge_graph(
    project_path="{project_path}"
)
```

---
*This document was auto-generated by hibro at {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""
        return workflow

    def _get_key_files_markdown(self, project_path: str) -> str:
        """Get markdown list of key files from knowledge graph"""
        try:
            from ..knowledge.graph_storage import GraphStorage, GraphNodeType

            storage = GraphStorage(self.memory_engine.db_manager)

            # Get file nodes sorted by importance
            file_nodes = storage.search_nodes(
                project_path=project_path,
                node_type=GraphNodeType.FILE,
                limit=5
            )

            if not file_nodes:
                return "*Waiting for knowledge graph analysis*"

            lines = []
            for node in file_nodes:
                lines.append(f"- `{node.file_path}` (importance: {node.importance:.1f})")

            return "\n".join(lines)

        except Exception as e:
            return f"*Failed to retrieve: {e}*"

    async def _tool_get_quick_context_fallback(self) -> Dict[str, Any]:
        """Fallback implementation for get quick context"""
        # Get preferences
        preferences = self.memory_engine.memory_repo.search_memories(
            memory_type="preference",
            limit=10
        )

        # Get recent decisions
        decisions = self.memory_engine.memory_repo.search_memories(
            memory_type="decision",
            limit=5
        )

        # Get important facts
        important = self.memory_engine.memory_repo.search_memories(
            min_importance=0.8,
            limit=5
        )

        return {
            "success": True,
            "preferences": [
                {"content": m.content, "category": m.category}
                for m in preferences
            ],
            "recent_decisions": [
                {"content": m.content, "importance": m.importance}
                for m in decisions
            ],
            "important_facts": [
                {"content": m.content, "importance": m.importance}
                for m in important
            ],
            "project_context": {},
            "stats": {
                "total_memories": self.memory_engine.get_statistics()["total_memories"]
            }
        }

    # ==================== Management Tools ====================

    async def _tool_get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        stats = self.memory_engine.get_statistics()

        return {
            "success": True,
            "data": {
                "total_memories": stats["total_memories"],
                "total_projects": stats["total_projects"],
                "total_preferences": stats["total_preferences"],
                "db_size_mb": round(stats["db_size_mb"], 2),
                "conversations_count": stats["conversations_count"],
                "contexts_count": stats["contexts_count"]
            }
        }

    async def _tool_update_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory"""
        memory_id = args.get("memory_id")
        content = args.get("content")
        importance = args.get("importance")

        if memory_id is None:
            return {"success": False, "error": "memory_id is required"}

        memory = self.memory_engine.get_memory(memory_id)
        if not memory:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        if content is not None:
            memory.content = content
        if importance is not None:
            memory.importance = importance

        self.memory_engine.memory_repo.update_memory(memory)

        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Memory {memory_id} updated"
        }

    async def _tool_forget(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete memory"""
        memory_id = args.get("memory_id")

        if memory_id is None:
            return {"success": False, "error": "memory_id is required"}

        success = self.memory_engine.delete_memory(memory_id)

        if success:
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        else:
            return {"success": False, "error": f"Memory {memory_id} not found"}

    # ==================== Resource Handling ====================

    async def _handle_resource_read(self, uri) -> str:
        """Handle resource reading"""
        try:
            # Convert AnyUrl to string if needed
            uri_str = str(uri)

            if uri_str == "hibro://system-prompt":
                return self._resource_system_prompt()
            elif uri_str == "hibro://quick-context":
                return self._resource_quick_context()
            elif uri_str == "hibro://status":
                return self._resource_status()
            elif uri_str == "hibro://preferences":
                return self._resource_preferences()
            elif uri_str == "hibro://projects":
                return self._resource_projects()
            elif uri_str == "hibro://recent":
                return self._resource_recent()
            elif uri_str == "hibro://important":
                return self._resource_important()
            elif uri_str.startswith("hibro://preferences/"):
                category = uri_str.split("/")[-1]
                return self._resource_preferences_by_category(category)
            elif uri_str.startswith("hibro://project/"):
                project_name = uri_str.split("/")[-1]
                return self._resource_project_context(project_name)
            else:
                return json.dumps({"error": f"Unknown resource: {uri_str}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _resource_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _resource_quick_context(self) -> str:
        preferences = self.memory_engine.memory_repo.search_memories(
            memory_type="preference", limit=10
        )
        decisions = self.memory_engine.memory_repo.search_memories(
            memory_type="decision", limit=5
        )
        important = self.memory_engine.memory_repo.search_memories(
            min_importance=0.8, limit=5
        )

        return json.dumps({
            "preferences": [{"content": m.content, "category": m.category} for m in preferences],
            "recent_decisions": [{"content": m.content} for m in decisions],
            "important_facts": [{"content": m.content} for m in important]
        }, ensure_ascii=False, indent=2)

    def _resource_status(self) -> str:
        stats = self.memory_engine.get_statistics()
        return json.dumps({
            "status": "running",
            "mode": "intelligent",
            "statistics": stats
        }, ensure_ascii=False, indent=2)

    def _resource_preferences(self) -> str:
        memories = self.memory_engine.memory_repo.search_memories(
            memory_type="preference", limit=50
        )
        return json.dumps([
            {"id": m.id, "content": m.content, "category": m.category}
            for m in memories
        ], ensure_ascii=False, indent=2)

    def _resource_projects(self) -> str:
        projects = self.memory_engine.project_repo.get_all_projects()
        return json.dumps(projects, ensure_ascii=False, indent=2)

    def _resource_recent(self) -> str:
        memories = self.memory_engine.memory_repo.search_memories(limit=20)
        return json.dumps([
            {"id": m.id, "content": m.content[:100], "type": m.memory_type}
            for m in memories
        ], ensure_ascii=False, indent=2)

    def _resource_important(self) -> str:
        memories = self.memory_engine.memory_repo.search_memories(
            min_importance=0.7, limit=20
        )
        return json.dumps([
            {"id": m.id, "content": m.content, "type": m.memory_type, "importance": m.importance}
            for m in memories
        ], ensure_ascii=False, indent=2)

    def _resource_preferences_by_category(self, category: str) -> str:
        memories = self.memory_engine.memory_repo.search_memories(
            memory_type="preference", limit=50
        )
        filtered = [m for m in memories if m.category == category]
        return json.dumps([
            {"id": m.id, "content": m.content}
            for m in filtered
        ], ensure_ascii=False, indent=2)

    def _resource_project_context(self, project_name: str) -> str:
        projects = self.memory_engine.project_repo.get_all_projects()
        project_path = None
        for p in projects:
            if project_name in p.get("path", "") or p.get("name") == project_name:
                project_path = p.get("path")
                break

        if not project_path:
            return json.dumps({"error": f"Project not found: {project_name}"})

        memories = self.memory_engine.get_project_memories(project_path, limit=50)
        return json.dumps([
            {"type": m.memory_type, "content": m.content}
            for m in memories
        ], ensure_ascii=False, indent=2)

    # ==================== Project Context Tools (New) ====================

    async def _tool_set_project_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set project context"""
        project_path = args.get("project_path")
        context_type = args.get("context_type")
        content = args.get("content")
        importance = args.get("importance", 0.8)

        if not project_path:
            return {"success": False, "error": "project_path is required"}
        if not context_type:
            return {"success": False, "error": "context_type is required"}
        if not content:
            return {"success": False, "error": "content is required"}

        try:
            # Determine memory type
            if context_type == "architecture":
                memory_type = "project_architecture"
            elif context_type == "command":
                memory_type = "project_command"
            else:
                return {"success": False, "error": f"Invalid context_type: {context_type}"}

            # Create project memory
            from ..storage.models import Memory
            memory = Memory(
                content=content,
                memory_type=memory_type,
                importance=importance
            )

            # Set project context information
            memory.setup_as_project_memory(
                project_path=project_path,
                context_type=context_type
            )

            # Set LFU counter
            memory.set_lfu_counter(50.0)  # Project context has higher priority

            # Save to database
            memory_id = self.memory_engine.memory_repo.create_memory(memory)

            self.logger.info(f"Project context set successfully: project={project_path}, type={context_type}, ID={memory_id}")

            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Project {context_type} context set successfully",
                "data": {
                    "project_path": project_path,
                    "context_type": context_type,
                    "content": content,
                    "importance": importance
                }
            }

        except Exception as e:
            self.logger.error(f"Set project context failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_set_active_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set active task"""
        project_path = args.get("project_path")
        task_content = args.get("task_content")
        importance = args.get("importance", 1.0)

        if not project_path:
            return {"success": False, "error": "project_path is required"}
        if not task_content:
            return {"success": False, "error": "task_content is required"}

        try:
            # Use active task manager to set task
            memory_id = self.active_task_manager.set_active_task(
                project_path=project_path,
                task_content=task_content,
                importance=importance
            )

            self.logger.info(f"Active task set successfully: project={project_path}, ID={memory_id}")

            return {
                "success": True,
                "memory_id": memory_id,
                "message": "Active task set successfully",
                "data": {
                    "project_path": project_path,
                    "task_content": task_content,
                    "importance": importance,
                    "note": "Previous active task (if any) has been deactivated"
                }
            }

        except Exception as e:
            self.logger.error(f"Set active task failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_scan_project(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scan project and create project_init memory to mark as initialized"""
        # Dynamically get current working directory if project_path not specified
        project_path = args.get("project_path")
        if not project_path:
            project_path = str(Path.cwd())
        quick_scan = args.get("quick_scan", True)

        if not project_path:
            return {
                "success": False,
                "error": "Project path not specified",
                "hint": "Please provide project_path parameter, or set current project first"
            }

        try:
            result = self.memory_engine.scan_and_remember_project(project_path, quick_scan)

            # Create project_init memory to mark project as initialized
            # So next get_quick_context call will return project_init.initialized: true
            from ..storage.models import Memory
            init_memory_content = f"Project initialized: {result['snapshot'].project_name}\nPath: {project_path}\nType: {result['snapshot'].project_type}"
            init_memory = Memory(
                content=init_memory_content,
                memory_type="project",
                category="project_init",
                importance=0.9,
                metadata={
                    "project_path": project_path,
                    "project_name": result['snapshot'].project_name,
                    "project_type": result['snapshot'].project_type,
                    "initialized_at": datetime.now().isoformat()
                }
            )
            self.memory_engine.memory_repo.create_memory(init_memory)
            self.logger.info(f"Created project_init memory for: {project_path}")

            return {
                "success": True,
                "project_name": result['snapshot'].project_name,
                "memory_id": result['memory_id'],
                "summary": result['summary'],
                "project_type": result['snapshot'].project_type,
                "tech_stack": result['snapshot'].tech_stack,
                "languages": result['snapshot'].languages,
                "statistics": result['snapshot'].statistics
            }

        except Exception as e:
            self.logger.error(f"Scan project failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_project_progress(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get project progress"""
        # Dynamically get current working directory if project_path not specified
        project_path = args.get("project_path")
        if not project_path:
            project_path = str(Path.cwd())

        if not project_path:
            return {
                "success": False,
                "error": "Project path not specified"
            }

        try:
            progress_report = self.memory_engine.get_project_progress(project_path)
            return progress_report

        except Exception as e:
            self.logger.error(f"Get project progress failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_update_project_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update project status"""
        project_path = args.get("project_path")
        status = args.get("status")
        current_task = args.get("current_task")
        progress_percentage = args.get("progress_percentage")
        notes = args.get("notes")

        if not project_path:
            return {"success": False, "error": "project_path is required"}

        try:
            result = self.memory_engine.update_project_status(
                project_path=project_path,
                status=status,
                current_task=current_task,
                progress_percentage=progress_percentage,
                notes=notes
            )

            return result

        except Exception as e:
            self.logger.error(f"Update project status failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_sync_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get database sync status"""
        try:
            # Get database listener status
            db_watcher = self.memory_engine._db_watcher

            if db_watcher:
                watcher_stats = db_watcher.get_stats()
                status = {
                    "success": True,
                    "sync_enabled": True,
                    "is_watching": watcher_stats['is_watching'],
                    "db_path": watcher_stats['db_path'],
                    "change_count": watcher_stats['change_count'],
                    "last_change_time": watcher_stats['last_change_time'],
                    "debounce_seconds": watcher_stats['debounce_seconds'],
                    "cache_version": self.memory_engine._cache_version,
                    "message": "Database listener running normally"
                }
            else:
                status = {
                    "success": True,
                    "sync_enabled": False,
                    "is_watching": False,
                    "db_path": str(self.memory_engine.db_manager.db_path),
                    "cache_version": self.memory_engine._cache_version,
                    "message": "Database listener not started (multi-session sync unavailable)"
                }

            return status

        except Exception as e:
            self.logger.error(f"Get sync status failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_event_bus_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get event bus status"""
        try:
            event_bus = self.memory_engine._event_bus

            if not event_bus:
                return {
                    "success": True,
                    "event_bus_enabled": False,
                    "message": "Event bus not initialized"
                }

            stats = event_bus.get_stats()

            return {
                "success": True,
                "event_bus_enabled": True,
                **stats,
                "message": "Event bus running normally" if stats['is_running'] else "Event bus stopped"
            }

        except Exception as e:
            self.logger.error(f"Get event bus status failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_list_event_subscribers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List event subscribers"""
        try:
            event_bus = self.memory_engine._event_bus

            if not event_bus:
                return {
                    "success": False,
                    "error": "Event bus not initialized"
                }

            subscribers = event_bus.get_subscribers_info()

            return {
                "success": True,
                "subscriber_count": len(subscribers),
                "subscribers": subscribers
            }

        except Exception as e:
            self.logger.error(f"List event subscribers failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_complete_active_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Complete active task"""
        project_path = args.get("project_path")

        if not project_path:
            return {"success": False, "error": "project_path is required"}

        try:
            # Use active task manager to complete task
            success = self.active_task_manager.complete_active_task(project_path)

            if success:
                self.logger.info(f"Active task completed: project={project_path}")
                return {
                    "success": True,
                    "message": "Active task completed successfully",
                    "data": {
                        "project_path": project_path,
                        "note": "Task is no longer active but memory is preserved"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No active task found for this project or completion failed"
                }

        except Exception as e:
            self.logger.error(f"Complete active task failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Memory Cleanup Tools (New) ====================

    async def _tool_trigger_cleanup(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Manually trigger memory cleanup"""
        force = args.get("force", False)

        try:
            # Check if cleanup system is available
            if not hasattr(self.memory_engine, 'cleaner'):
                return {
                    "success": False,
                    "error": "Memory cleanup system not available"
                }

            # Execute cleanup
            result = self.memory_engine.cleaner.execute_cleanup(force=force)

            return {
                "success": result.get("success", False),
                "deleted_count": result.get("deleted_count", 0),
                "strategies": result.get("strategies", {}),
                "duration_ms": result.get("duration_ms", 0),
                "forced": result.get("forced", False),
                "error": result.get("error"),
                "message": (
                    f"Cleanup completed: {result.get('deleted_count', 0)} memories deleted"
                    if result.get("success")
                    else f"Cleanup failed: {result.get('error', 'Unknown error')}"
                )
            }

        except Exception as e:
            self.logger.error(f"Trigger cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_cleanup_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory cleanup status"""
        try:
            # Get usage status
            usage_status = {}
            if hasattr(self.memory_engine, 'threshold_checker'):
                usage_status = self.memory_engine.threshold_checker.get_usage_status()

            # Get scheduler status
            scheduler_status = {}
            if hasattr(self.memory_engine, 'cleanup_scheduler'):
                scheduler_status = self.memory_engine.cleanup_scheduler.get_status()

            # Get last cleanup stats
            last_cleanup = None
            if hasattr(self.memory_engine, 'cleaner'):
                last_cleanup = self.memory_engine.cleaner.get_cleanup_stats()

            # Get cleanup preview
            preview = {}
            if hasattr(self.memory_engine, 'cleaner'):
                preview = self.memory_engine.cleaner.get_cleanup_preview()

            return {
                "success": True,
                "usage": usage_status,
                "scheduler": scheduler_status,
                "last_cleanup": last_cleanup,
                "preview": preview
            }

        except Exception as e:
            self.logger.error(f"Get cleanup status failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Code Knowledge Graph Tools (New) ====================

    async def _tool_init_code_knowledge_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize code knowledge graph for a project"""
        project_path = args.get("project_path")

        if not project_path:
            return {"success": False, "error": "project_path is required"}

        try:
            from ..knowledge.graph_storage import GraphStorage
            from ..knowledge.session_update import SessionUpdateManager

            # Initialize storage
            storage = GraphStorage(self.memory_engine.db_manager)

            # Initialize session update manager
            manager = SessionUpdateManager(
                project_path=project_path,
                storage=storage,
                max_workers=2
            )

            # Perform full scan
            self.logger.info(f"Initializing code knowledge graph for: {project_path}")
            stats = manager.perform_full_scan()

            # Mark project as knowledge graph initialized
            from ..storage.models import Memory
            init_memory = Memory(
                content=f"Code knowledge graph initialized: {project_path}",
                memory_type="project",
                category="code_kg_init",
                importance=0.9,
                metadata={
                    "project_path": project_path,
                    "initialized_at": datetime.now().isoformat(),
                    "stats": stats
                }
            )
            self.memory_engine.memory_repo.create_memory(init_memory)

            return {
                "success": True,
                "project_path": project_path,
                "statistics": stats,
                "message": f"Code knowledge graph initialized: {stats['files_processed']} files, {stats['classes_added']} classes, {stats['functions_added']} functions"
            }

        except Exception as e:
            self.logger.error(f"Init code knowledge graph failed: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_get_code_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get code context from knowledge graph"""
        project_path = args.get("project_path")
        detail_level = args.get("detail_level", "lightweight")
        search_query = args.get("search_query")

        if not project_path:
            return {"success": False, "error": "project_path is required"}

        try:
            from ..knowledge.graph_storage import GraphStorage
            from ..knowledge.code_query import KnowledgeGraphQuery

            # Initialize storage and query
            storage = GraphStorage(self.memory_engine.db_manager)
            query = KnowledgeGraphQuery(storage)

            # Check if knowledge graph has data
            nodes = storage.search_nodes(project_path=project_path, limit=1)
            if not nodes:
                return {
                    "success": False,
                    "error": "Code knowledge graph not initialized",
                    "hint": "Call init_code_knowledge_graph first to scan and build the graph"
                }

            # Perform search if query provided
            if search_query:
                result = query.search_code(project_path, search_query)
                return {
                    "success": result.success,
                    "query_time_ms": result.query_time_ms,
                    "token_estimate": result.token_estimate,
                    "results": result.data,
                    "error": result.error
                }

            # Get context based on detail level
            if detail_level == "medium":
                result = query.get_detailed_context(project_path)
            else:
                result = query.get_quick_project_context(project_path)

            if result.success:
                return {
                    "success": True,
                    "detail_level": detail_level,
                    "query_time_ms": result.query_time_ms,
                    "token_estimate": result.token_estimate,
                    "data": result.data.__dict__ if hasattr(result.data, '__dict__') else result.data
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }

        except Exception as e:
            self.logger.error(f"Get code context failed: {e}")
            return {"success": False, "error": str(e)}

    async def run(self):
        """Run MCP Server"""
        self.logger.info("Starting hibro MCP Server (Intelligent Mode)...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def run_server():
    """Start MCP Server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

    server = MCPServer()
    await server.run()


def main():
    """MCP server entry point"""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
