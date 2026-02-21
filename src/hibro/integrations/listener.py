#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Listener Module
Monitors IDE conversation file changes and captures conversation content in real-time
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..core.memory_engine import MemoryEngine
from ..intelligence import MemoryExtractor
from ..interfaces.automation import AutomationEngine
from ..utils.config import Config


class ConversationFileHandler(FileSystemEventHandler):
    """Conversation file event handler"""

    def __init__(self, listener: 'ConversationListener'):
        """
        Initialize file event handler

        Args:
            listener: Conversation listener instance
        """
        self.listener = listener
        self.logger = logging.getLogger('hibro.conversation_file_handler')

    def on_modified(self, event):
        """File modification event handler"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a conversation file
        if self._is_conversation_file(file_path):
            self.logger.debug(f"Detected conversation file change: {file_path}")
            self.listener.process_conversation_file(file_path)

    def on_created(self, event):
        """File creation event handler"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if self._is_conversation_file(file_path):
            self.logger.debug(f"Detected new conversation file: {file_path}")
            # Slight delay to ensure file write completion
            time.sleep(0.5)
            self.listener.process_conversation_file(file_path)

    def _is_conversation_file(self, file_path: Path) -> bool:
        """
        Determine if it's a conversation file

        Args:
            file_path: File path

        Returns:
            Whether it's a conversation file
        """
        # IDE conversation files are usually in JSON format
        if file_path.suffix.lower() not in ['.json', '.jsonl']:
            return False

        # Check if file path contains conversation-related directories
        path_parts = file_path.parts
        conversation_indicators = ['conversations', 'chat', 'claude', 'cursor', 'qoder', 'trae']

        return any(indicator in str(file_path).lower() for indicator in conversation_indicators)


class ConversationListener:
    """Conversation Listener"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize conversation listener

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.conversation_listener')

        # Initialize components
        self.memory_extractor = MemoryExtractor()
        self.automation_engine = AutomationEngine(config, memory_engine)

        # Monitoring configuration
        self.watch_directories = self._get_watch_directories()
        self.processed_files = set()  # Processed files to avoid duplicate processing
        self.last_processed_time = {}  # File last processed time

        # File listener
        self.observer = Observer()
        self.file_handler = ConversationFileHandler(self)

        # Callback functions
        self.on_conversation_detected: Optional[Callable] = None
        self.on_memory_extracted: Optional[Callable] = None

        # Running status
        self.is_running = False

    def _get_watch_directories(self) -> List[Path]:
        """Get list of directories to monitor"""
        directories = []

        # IDE-specific conversation directories (will be detected dynamically)
        # Check for available IDE integrations and use their directories
        from ..integrations import IDEManager
        ide_manager = IDEManager(self.config)
        active_integration = ide_manager.get_active_integration()

        if active_integration:
            ide_dirs = active_integration.get_conversation_directories()
            for ide_dir in ide_dirs:
                if ide_dir.exists():
                    directories.append(ide_dir)
                    self.logger.info(f"Found {active_integration.get_name()} directory: {ide_dir}")

                    # Add all sub-project directories
                    if ide_dir.is_dir():
                        for project_dir in ide_dir.iterdir():
                            if project_dir.is_dir():
                                directories.append(project_dir)
                                self.logger.info(f"Found {active_integration.get_name()} project: {project_dir.name}")

        # Legacy IDE conversation directories (for backward compatibility)
        legacy_dirs = [
            Path.home() / '.claude' / 'projects',
            Path.home() / '.claude' / 'conversations',
            Path.home() / '.claude' / 'chat',
            Path.home() / 'AppData' / 'Roaming' / 'Claude' / 'conversations',  # Windows
            Path.home() / 'Library' / 'Application Support' / 'Claude' / 'conversations',  # macOS
            Path.home() / '.cursor' / 'conversations',
            Path.home() / '.qoder' / 'chats',
            Path.home() / '.trae' / 'sessions',
        ]

        for directory in legacy_dirs:
            if directory.exists() and directory not in directories:
                directories.append(directory)
                self.logger.info(f"Found legacy IDE conversation directory: {directory}")

        # User custom directories
        custom_dirs = self.config.ide_integration.__dict__.get('watch_directories', [])
        for custom_dir in custom_dirs:
            dir_path = Path(custom_dir).expanduser()
            if dir_path.exists():
                directories.append(dir_path)

        if not directories:
            self.logger.warning("No IDE conversation directory found, will create default monitoring directory")
            default_dir = Path(self.config.data_directory) / 'conversations'
            default_dir.mkdir(parents=True, exist_ok=True)
            directories.append(default_dir)

        return directories

    def start_listening(self):
        """Start listening to conversation files"""
        if self.is_running:
            self.logger.warning("Conversation listener is already running")
            return

        try:
            # Setup file monitoring
            for directory in self.watch_directories:
                self.observer.schedule(
                    self.file_handler,
                    str(directory),
                    recursive=True
                )
                self.logger.info(f"Started monitoring directory: {directory}")

            # Start listener
            self.observer.start()
            self.is_running = True

            self.logger.info("Conversation listener started")

            # Process existing files
            self._process_existing_files()

        except Exception as e:
            self.logger.error(f"Failed to start conversation listener: {e}")
            raise

    def stop_listening(self):
        """Stop listening to conversation files"""
        if not self.is_running:
            return

        try:
            self.observer.stop()
            self.observer.join()
            self.is_running = False

            self.logger.info("Conversation listener stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop conversation listener: {e}")

    def _process_existing_files(self):
        """Process existing conversation files"""
        self.logger.info("Started processing existing conversation files...")

        for directory in self.watch_directories:
            try:
                # Find conversation files
                for file_path in directory.rglob('*.json'):
                    if self._should_process_file(file_path):
                        self.process_conversation_file(file_path)

                for file_path in directory.rglob('*.jsonl'):
                    if self._should_process_file(file_path):
                        self.process_conversation_file(file_path)

            except Exception as e:
                self.logger.warning(f"Failed to process existing files in directory {directory}: {e}")

    def _should_process_file(self, file_path: Path) -> bool:
        """
        Determine if file should be processed

        Args:
            file_path: File path

        Returns:
            Whether file should be processed
        """
        # Check if file has been processed
        file_key = str(file_path)
        if file_key in self.processed_files:
            return False

        # Check file modification time
        try:
            file_mtime = file_path.stat().st_mtime
            last_processed = self.last_processed_time.get(file_key, 0)

            if file_mtime <= last_processed:
                return False

        except Exception:
            return False

        return True

    def process_conversation_file(self, file_path: Path):
        """
        Process conversation file

        Args:
            file_path: Conversation file path
        """
        try:
            # Read file content
            conversation_data = self._read_conversation_file(file_path)

            if not conversation_data:
                return

            # Extract conversation content
            conversations = self._extract_conversations(conversation_data)

            if not conversations:
                return

            # Detect if it's IDE JSONL format (items in list have type field)
            is_ide_jsonl_format = (
                conversations and
                isinstance(conversations[0], dict) and
                'type' in conversations[0]
            )

            if is_ide_jsonl_format:
                # IDE JSONL format: Process entire session merged
                self._process_ide_session(conversations, file_path)
            else:
                # Traditional format: Process individually
                for conversation in conversations:
                    self._process_single_conversation(conversation, file_path)

            # Mark file as processed
            file_key = str(file_path)
            self.processed_files.add(file_key)
            self.last_processed_time[file_key] = time.time()

            self.logger.info(f"Processed conversation file: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to process conversation file {file_path}: {e}")

    def _read_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read conversation file content

        Args:
            file_path: File path

        Returns:
            Conversation data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format, one JSON object per line
                    lines = f.readlines()
                    return {'conversations': [json.loads(line) for line in lines if line.strip()]}
                else:
                    # Standard JSON format
                    return json.load(f)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def _extract_conversations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract conversation content from data

        Args:
            data: Raw data

        Returns:
            Conversation list
        """
        conversations = []

        try:
            # Try different data structures
            if 'conversations' in data:
                conversations.extend(data['conversations'])
            elif 'messages' in data:
                conversations.append({'messages': data['messages']})
            elif isinstance(data, list):
                conversations.extend(data)
            elif 'content' in data or 'text' in data:
                conversations.append(data)

        except Exception as e:
            self.logger.warning(f"Failed to extract conversation content: {e}")

        return conversations

    def _process_ide_session(self, messages: List[Dict[str, Any]], source_file: Path):
        """
        Process IDE session (entire JSONL file)

        Merge and process entire session, extract key information from user messages and assistant responses

        Args:
            messages: List of all messages in session
            source_file: Source file path
        """
        try:
            # Merge all text content
            all_text_parts = []
            user_messages = []

            for msg in messages:
                text = self._extract_text_from_conversation(msg)
                if text and len(text.strip()) > 5:
                    all_text_parts.append(text)

                    # Collect user messages for memory extraction
                    if msg.get('type') == 'user':
                        content = msg.get('message', {}).get('content', '')
                        if isinstance(content, str):
                            user_messages.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    user_messages.append(item.get('text', ''))

            if not all_text_parts:
                return

            # Merge text
            combined_text = '\n\n'.join(all_text_parts)

            # If content too long, only process user messages (more likely to contain preferences and decisions)
            if len(combined_text) > 10000:
                combined_text = '\n\n'.join(user_messages)

            if len(combined_text.strip()) < 10:
                return

            # Detect project context
            project_context = self._detect_project_context(source_file)

            self.logger.debug(f"Processing IDE session: {source_file.name}, "
                             f"Message count: {len(messages)}, Text length: {len(combined_text)}")

            # Process using automation engine
            automation_result = self.automation_engine.process_memory(
                combined_text,
                memory_type='conversation',
                context=project_context
            )

            # Apply automation results
            if automation_result:
                self._apply_automation_result(combined_text, automation_result, project_context)

            # Trigger callback
            if self.on_conversation_detected:
                self.on_conversation_detected(combined_text, messages, source_file)

        except Exception as e:
            self.logger.error(f"Failed to process IDE session: {e}")

    def _process_single_conversation(self, conversation: Dict[str, Any], source_file: Path):
        """
        Process single conversation

        Args:
            conversation: Conversation data
            source_file: Source file path
        """
        try:
            # Extract text content
            text_content = self._extract_text_from_conversation(conversation)

            if not text_content or len(text_content.strip()) < 10:
                return

            # Detect project context
            project_context = self._detect_project_context(source_file)

            # Process using automation engine
            automation_result = self.automation_engine.process_memory(
                text_content,
                memory_type='conversation',
                context=project_context
            )

            # Apply automation results
            if automation_result:
                self._apply_automation_result(text_content, automation_result, project_context)

            # Trigger callback
            if self.on_conversation_detected:
                self.on_conversation_detected(text_content, conversation, source_file)

        except Exception as e:
            self.logger.error(f"Failed to process single conversation: {e}")

    def _extract_text_from_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Extract text content from conversation data
        Supports IDE JSONL format

        Args:
            conversation: Conversation data

        Returns:
            Extracted text content
        """
        text_parts = []

        try:
            # Process IDE JSONL format
            # Each line is independent JSON object, containing type, message, timestamp
            if 'type' in conversation and 'message' in conversation:
                msg_type = conversation.get('type')
                message = conversation.get('message', {})

                # Extract user message
                if msg_type == 'user':
                    content = message.get('content', '')
                    if isinstance(content, str):
                        text_parts.append(f"[User]: {content}")
                    elif isinstance(content, list):
                        # Process structured content (e.g., image + text)
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(f"[User]: {item.get('text', '')}")
                            elif isinstance(item, str):
                                text_parts.append(f"[User]: {item}")

                # Extract assistant response
                elif msg_type == 'assistant':
                    content = message.get('content', '')
                    if isinstance(content, str):
                        text_parts.append(f"[Assistant]: {content}")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(f"[Assistant]: {item.get('text', '')}")

                # Extract tool call results (may contain important information)
                elif msg_type == 'tool_result':
                    content = message.get('content', '')
                    if isinstance(content, str) and len(content) < 2000:  # Limit length
                        text_parts.append(f"[Tool Result]: {content[:500]}")

            # Process traditional message list format
            elif 'messages' in conversation:
                for message in conversation['messages']:
                    content = self._extract_message_content(message)
                    if content:
                        role = message.get('role', 'unknown')
                        text_parts.append(f"[{role}]: {content}")

            # Process direct content
            elif 'content' in conversation:
                text_parts.append(str(conversation['content']))
            elif 'text' in conversation:
                text_parts.append(str(conversation['text']))

            # Process other possible fields
            for key in ['user_message', 'assistant_message', 'response']:
                if key in conversation:
                    text_parts.append(str(conversation[key]))

        except Exception as e:
            self.logger.warning(f"Failed to extract text content: {e}")

        return '\n'.join(text_parts)

    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        """
        Extract content from message

        Args:
            message: Message data

        Returns:
            Message content
        """
        # Try different content fields
        content_fields = ['content', 'text', 'message', 'body']

        for field in content_fields:
            if field in message:
                content = message[field]
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Process structured content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return '\n'.join(text_parts)

        return ""

    def _detect_project_context(self, source_file: Path) -> Dict[str, Any]:
        """
        Detect project context

        Args:
            source_file: Source file path

        Returns:
            Project context information
        """
        context = {
            'source_file': str(source_file),
            'detected_at': datetime.now().isoformat()
        }

        try:
            # Try to infer project from file path
            path_parts = source_file.parts

            # Find possible project directories
            for i, part in enumerate(path_parts):
                if part in ['projects', 'workspace', 'code', 'dev']:
                    if i + 1 < len(path_parts):
                        context['project_name'] = path_parts[i + 1]
                        context['project_path'] = str(Path(*path_parts[:i + 2]))
                        break

        except Exception as e:
            self.logger.warning(f"Failed to detect project context: {e}")

        return context

    def _apply_automation_result(self, content: str, result: Dict[str, Any], context: Dict[str, Any]):
        """
        Apply automation processing results

        Args:
            content: Original content
            result: Automation processing result
            context: Context information
        """
        try:
            # Store memory
            if result.get('store_memory', True):
                memory_id = self.memory_engine.store_memory(
                    content=result.get('modified_content', content),
                    importance=0.5 + result.get('importance_boost', 0.0),
                    category=result.get('categories', [None])[0] if result.get('categories') else None,
                    memory_type='conversation',
                    project_path=result.get('project_path') or context.get('project_path')
                )

                self.logger.debug(f"Auto-stored memory: ID={memory_id}")

            # Process extracted memories
            if 'extracted_memories' in result:
                for extracted in result['extracted_memories']:
                    self.memory_engine.store_memory(
                        content=extracted['content'],
                        importance=extracted['importance'],
                        category=extracted.get('category'),
                        memory_type=extracted['type'],
                        project_path=context.get('project_path')
                    )

            # Trigger callback
            if self.on_memory_extracted:
                self.on_memory_extracted(result, context)

        except Exception as e:
            self.logger.error(f"Failed to apply automation results: {e}")

    def get_listening_status(self) -> Dict[str, Any]:
        """
        Get listening status

        Returns:
            Listening status information
        """
        return {
            'is_running': self.is_running,
            'watch_directories': [str(d) for d in self.watch_directories],
            'processed_files_count': len(self.processed_files),
            'last_activity': max(self.last_processed_time.values()) if self.last_processed_time else None
        }

    def set_callbacks(self, on_conversation_detected: Optional[Callable] = None,
                     on_memory_extracted: Optional[Callable] = None):
        """
        Set callback functions

        Args:
            on_conversation_detected: Callback when conversation detected
            on_memory_extracted: Callback when memory extracted
        """
        self.on_conversation_detected = on_conversation_detected
        self.on_memory_extracted = on_memory_extracted