#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Conversation Interface Module
Provides ChatGPT-like natural language interaction
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..core.memory_engine import MemoryEngine
from ..intelligence import MemoryExtractor, SimilarityCalculator, SemanticSearchEngine
from ..utils.config import Config


class ConversationHandler:
    """Conversation Handler"""

    def __init__(self, config: Config):
        """
        Initialize conversation handler

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.conversation_handler')

        # Initialize core components
        self.memory_engine = MemoryEngine(config)
        self.memory_extractor = MemoryExtractor()
        self.similarity_calc = SimilarityCalculator()
        self.search_engine = SemanticSearchEngine(self.similarity_calc)

        # Conversation intent recognition patterns
        self.intent_patterns = {
            'remember': {
                'patterns': [
                    r'remember.*',
                    r'save.*',
                    r'store.*',
                    r'i want to record.*',
                    r'help me note.*'
                ],
                'keywords': ['remember', 'save', 'store', 'record', 'note']
            },
            'recall': {
                'patterns': [
                    r'recall.*',
                    r'remember.*',
                    r'find.*',
                    r'search.*',
                    r'look for.*memory',
                    r'.*do you remember.*',
                    r'tell me.*'
                ],
                'keywords': ['recall', 'remember', 'find', 'search', 'look for', 'do you remember', 'tell me']
            },
            'forget': {
                'patterns': [
                    r'forget.*',
                    r'delete.*',
                    r'remove.*',
                    r'don\'t want.*anymore'
                ],
                'keywords': ['forget', 'delete', 'remove', 'don\'t want']
            },
            'status': {
                'patterns': [
                    r'status.*',
                    r'statistics.*',
                    r'how many.*memor',
                    r'system.*status'
                ],
                'keywords': ['status', 'statistics', 'how many', 'situation']
            },
            'help': {
                'patterns': [
                    r'help.*',
                    r'how to.*',
                    r'how can.*',
                    r'what can.*do'
                ],
                'keywords': ['help', 'how to', 'how can', 'what can', 'what']
            }
        }

        # Initialize memory engine
        try:
            self.memory_engine.initialize()
        except Exception as e:
            self.logger.warning(f"Memory engine initialization failed: {e}")

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user message

        Args:
            message: User message
            context: Context information

        Returns:
            Reply message
        """
        if not message or not message.strip():
            return "Please tell me what you want to do."

        try:
            # Detect intent
            intent, confidence = self._detect_intent(message)

            self.logger.info(f"Detected intent: {intent} (confidence: {confidence:.2f})")

            # Handle based on intent
            if intent == 'remember':
                return self._handle_remember(message, context)
            elif intent == 'recall':
                return self._handle_recall(message, context)
            elif intent == 'forget':
                return self._handle_forget(message, context)
            elif intent == 'status':
                return self._handle_status(message, context)
            elif intent == 'help':
                return self._handle_help(message, context)
            else:
                # Default handling: try intelligent extraction or search
                return self._handle_general(message, context)

        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return f"Sorry, an error occurred while processing your request: {e}"

    def _detect_intent(self, message: str) -> Tuple[str, float]:
        """
        Detect user intent

        Args:
            message: User message

        Returns:
            (intent, confidence)
        """
        message_lower = message.lower()
        best_intent = 'general'
        best_confidence = 0.0

        for intent, config in self.intent_patterns.items():
            confidence = 0.0

            # Check pattern matching
            for pattern in config['patterns']:
                if re.search(pattern, message):
                    confidence += 0.5

            # Check keyword matching
            for keyword in config['keywords']:
                if keyword in message_lower:
                    confidence += 0.3

            # Normalize confidence
            confidence = min(confidence, 1.0)

            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence

        return best_intent, best_confidence

    def _handle_remember(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle remember request"""
        try:
            # Extract content to remember
            content = self._extract_content_to_remember(message)

            if not content:
                return "Please tell me what you want to remember."

            # Use intelligent extractor
            extracted_memories = self.memory_extractor.extract_memories(content)

            if not extracted_memories:
                # Store original content directly
                memory_id = self.memory_engine.store_memory(
                    content=content,
                    importance=0.5,
                    project_path=context.get('project_path') if context else None
                )
                return f"✅ Remembered: {content[:50]}{'...' if len(content) > 50 else ''} (ID: {memory_id})"

            # Store extracted memories
            results = []
            for extracted in extracted_memories:
                memory_id = self.memory_engine.store_memory(
                    content=extracted.content,
                    importance=extracted.importance,
                    category=extracted.category,
                    memory_type=extracted.memory_type,
                    project_path=context.get('project_path') if context else None
                )
                results.append(f"- {extracted.memory_type}: {extracted.content[:40]}... (ID: {memory_id})")

            return f"🧠 Intelligently extracted and saved {len(extracted_memories)} memories:\n" + "\n".join(results)

        except Exception as e:
            return f"Failed to save memory: {e}"

    def _handle_recall(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle recall request"""
        try:
            # Extract query content
            query = self._extract_query_content(message)

            if not query:
                return "Please tell me what you want to recall."

            # Use semantic search
            memories = self.memory_engine.recall_memories(query, limit=100)

            if not memories:
                return f"🤔 No memories found related to '{query}'."

            # Convert to dictionary format for semantic search
            memory_dicts = [memory.to_dict() for memory in memories]

            results = self.search_engine.search_memories(
                query, memory_dicts, top_k=5, min_similarity=0.3
            )

            if not results:
                return f"🤔 No memories found similar enough to '{query}'."

            # Format results
            response_lines = [f"📚 Found {len(results)} related memories:"]

            for i, (memory_dict, similarity) in enumerate(results, 1):
                content = memory_dict['content']
                if len(content) > 80:
                    content = content[:80] + "..."

                response_lines.append(
                    f"{i}. [{memory_dict['memory_type']}] {content} "
                    f"(similarity: {similarity:.2f}, importance: {memory_dict['importance']:.2f})"
                )

            return "\n".join(response_lines)

        except Exception as e:
            return f"Recall search failed: {e}"

    def _handle_forget(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle forget request"""
        try:
            # Extract content to forget
            content = self._extract_content_to_forget(message)

            if not content:
                return "Please tell me what you want to forget, or provide a memory ID."

            # Try to parse as ID
            try:
                memory_id = int(content)
                success = self.memory_engine.delete_memory(memory_id)
                if success:
                    return f"✅ Deleted memory ID: {memory_id}"
                else:
                    return f"❌ Memory ID not found: {memory_id}"
            except ValueError:
                # Not an ID, search for related memories
                memories = self.memory_engine.recall_memories(content, limit=5)

                if not memories:
                    return f"🤔 No memories found related to '{content}'."

                if len(memories) == 1:
                    # Only one result, delete directly
                    memory = memories[0]
                    success = self.memory_engine.delete_memory(memory.id)
                    if success:
                        return f"✅ Deleted memory: {memory.content[:50]}..."
                    else:
                        return "❌ Deletion failed"
                else:
                    # Multiple results, let user choose
                    response_lines = [f"Found {len(memories)} related memories, please specify the memory ID to delete:"]
                    for memory in memories:
                        content_preview = memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
                        response_lines.append(f"ID {memory.id}: {content_preview}")
                    return "\n".join(response_lines)

        except Exception as e:
            return f"Forget operation failed: {e}"

    def _handle_status(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle status query"""
        try:
            stats = self.memory_engine.get_statistics()

            response_lines = [
                "📊 hibro Memory System Status:",
                f"• Total memories: {stats['total_memories']}",
                f"• Projects: {stats['total_projects']}",
                f"• Database size: {stats['db_size_mb']:.1f} MB",
                f"• Conversations: {stats['conversations_count']}"
            ]

            # If there are memories, show distribution statistics
            if stats['total_memories'] > 0:
                memories = self.memory_engine.memory_repo.search_memories(limit=1000)
                if memories:
                    from ..intelligence.forgetting import ForgettingManager
                    forgetting_manager = ForgettingManager(self.config)
                    forgetting_stats = forgetting_manager.get_forgetting_statistics(memories)

                    response_lines.extend([
                        "",
                        "🧠 Memory Analysis:",
                        f"• High importance: {forgetting_stats['importance_distribution']['high']}",
                        f"• Medium importance: {forgetting_stats['importance_distribution']['medium']}",
                        f"• Low importance: {forgetting_stats['importance_distribution']['low']}",
                        f"• Forgetting candidates: {forgetting_stats['forgetting_candidates']}"
                    ])

            return "\n".join(response_lines)

        except Exception as e:
            return f"Failed to get status: {e}"

    def _handle_help(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle help request"""
        help_text = """
🤖 hibro Smart Memory Assistant

I can help you with:

📝 **Memory Management**
• "Remember this important decision..."
• "Save this code explanation..."
• "I want to record the project architecture design..."

🔍 **Smart Recall**
• "Recall discussions about React..."
• "Find memories about database design..."
• "Tell me about previous technology choices..."

🗑️ **Forget Management**
• "Forget memory ID 123"
• "Delete memories about old project..."

📊 **System Status**
• "Show system status"
• "Statistics"

💡 **Smart Features**
• Automatic extraction of important information
• Semantic similarity search
• Intelligent forgetting mechanism
• Project context management

Just tell me what you need in natural language!
        """
        return help_text.strip()

    def _handle_general(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Handle general message"""
        try:
            # Try intelligent extraction
            extracted_memories = self.memory_extractor.extract_memories(message)

            if extracted_memories:
                # Extracted memories, ask if user wants to save
                preview = []
                for i, extracted in enumerate(extracted_memories[:3], 1):
                    preview.append(f"{i}. [{extracted.memory_type}] {extracted.content[:50]}...")

                preview_text = "\n".join(preview)
                if len(extracted_memories) > 3:
                    preview_text += f"\n... and {len(extracted_memories) - 3} more"

                return f"🧠 I extracted some potentially important information from your message:\n\n{preview_text}\n\nWould you like to save these memories? Say 'remember' or 'save'."

            # No memories extracted, try searching for related content
            memories = self.memory_engine.recall_memories(message, limit=3)

            if memories:
                response_lines = ["🔍 I found some potentially related memories:"]
                for i, memory in enumerate(memories, 1):
                    content = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
                    response_lines.append(f"{i}. {content}")

                response_lines.append("\nNeed more detailed information?")
                return "\n".join(response_lines)

            # Nothing found, provide help
            return "🤔 I'm not sure what you want to do. You can:\n• Ask me to remember something\n• Search existing memories\n• Ask about system status\n• Say 'help' for more information"

        except Exception as e:
            return f"Error processing message: {e}"

    def _extract_content_to_remember(self, message: str) -> str:
        """Extract content to remember from message"""
        # Remove memory-related trigger words
        triggers = ['remember', 'save', 'store', 'record', 'note']

        content = message
        for trigger in triggers:
            if trigger in content.lower():
                # Find content after trigger word
                parts = content.split(trigger, 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    # Remove common connecting words
                    content = re.sub(r'^[：:，,。.！!？?]*', '', content).strip()
                    break

        return content if content != message else message

    def _extract_query_content(self, message: str) -> str:
        """Extract query content from message"""
        # Remove query-related trigger words
        triggers = ['recall', 'remember', 'find', 'search', 'look for', 'tell me']

        content = message
        for trigger in triggers:
            if trigger in content.lower():
                parts = content.split(trigger, 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    content = re.sub(r'^[：:，,。.！!？?]*', '', content).strip()
                    # Remove question words
                    content = re.sub(r'\?$', '', content).strip()
                    break

        return content if content != message else message

    def _extract_content_to_forget(self, message: str) -> str:
        """Extract content to forget from message"""
        # Remove forget-related trigger words
        triggers = ['forget', 'delete', 'remove']

        content = message
        for trigger in triggers:
            if trigger in content.lower():
                parts = content.split(trigger, 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    content = re.sub(r'^[：:，,。.！!？?]*', '', content).strip()
                    break

        return content if content != message else message


class ChatInterface:
    """Chat Interface"""

    def __init__(self, config: Config):
        """
        Initialize chat interface

        Args:
            config: Configuration object
        """
        self.config = config
        self.handler = ConversationHandler(config)
        self.logger = logging.getLogger('hibro.chat_interface')

    def start_chat(self):
        """Start chat interface"""
        print("🤖 hibro Smart Memory Assistant")
        print("Type 'quit', 'exit' or 'bye' to exit")
        print("Type 'help' for help")
        print("-" * 50)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Check exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break

                # Process message
                response = self.handler.process_message(user_input)
                print(f"\nhibro: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error occurred: {e}")
                self.logger.error(f"Chat interface error: {e}")

    def process_single_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process single message (for API calls)

        Args:
            message: User message
            context: Context information

        Returns:
            Reply message
        """
        return self.handler.process_message(message, context)