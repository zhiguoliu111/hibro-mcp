"""
Generic Conversation Parser
Provides standardized conversation file parsing functionality
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re

logger = logging.getLogger(__name__)


class ConversationParser:
    """Generic conversation parser"""

    @staticmethod
    def parse_json_conversation(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse JSON format conversation file (Claude Code format)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Standardize format
            conversation = {
                'id': data.get('id', str(file_path.stem)),
                'title': data.get('name', data.get('title', 'Untitled')),
                'created_at': ConversationParser._parse_timestamp(data.get('created_at')),
                'updated_at': ConversationParser._parse_timestamp(data.get('updated_at')),
                'messages': ConversationParser._extract_messages_from_json(data),
                'project_path': data.get('project_path'),
                'metadata': {
                    'format': 'json',
                    'source': 'claude_code',
                    'file_path': str(file_path),
                    'raw_data': data
                }
            }

            return conversation

        except Exception as e:
            logger.error(f"Failed to parse JSON conversation file {file_path}: {e}")
            return None

    @staticmethod
    def parse_jsonl_conversation(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse JSONL format conversation file
        """
        try:
            messages = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        message_data = json.loads(line)
                        messages.append(message_data)

            if not messages:
                return None

            # Infer information from filename and messages
            conversation = {
                'id': str(file_path.stem),
                'title': ConversationParser._extract_title_from_messages(messages),
                'created_at': ConversationParser._parse_timestamp(messages[0].get('timestamp')) if messages else datetime.now(),
                'updated_at': ConversationParser._parse_timestamp(messages[-1].get('timestamp')) if messages else datetime.now(),
                'messages': messages,
                'project_path': None,
                'metadata': {
                    'format': 'jsonl',
                    'source': 'unknown',
                    'file_path': str(file_path),
                    'message_count': len(messages)
                }
            }

            return conversation

        except Exception as e:
            logger.error(f"Failed to parse JSONL conversation file {file_path}: {e}")
            return None

    @staticmethod
    def parse_text_conversation(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse plain text format conversation file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple text parsing
            messages = ConversationParser._parse_text_messages(content)

            conversation = {
                'id': str(file_path.stem),
                'title': ConversationParser._extract_title_from_text(content),
                'created_at': datetime.fromtimestamp(file_path.stat().st_ctime),
                'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime),
                'messages': messages,
                'project_path': None,
                'metadata': {
                    'format': 'text',
                    'source': 'unknown',
                    'file_path': str(file_path),
                    'content_length': len(content)
                }
            }

            return conversation

        except Exception as e:
            logger.error(f"Failed to parse text conversation file {file_path}: {e}")
            return None

    @staticmethod
    def auto_parse(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Automatically detect file format and parse
        """
        if not file_path.exists():
            return None

        # Select parser based on file extension
        suffix = file_path.suffix.lower()

        if suffix == '.json':
            return ConversationParser.parse_json_conversation(file_path)
        elif suffix == '.jsonl':
            return ConversationParser.parse_jsonl_conversation(file_path)
        elif suffix in ['.txt', '.md', '.chat']:
            return ConversationParser.parse_text_conversation(file_path)
        else:
            # Try parsing as JSON
            result = ConversationParser.parse_json_conversation(file_path)
            if result:
                return result

            # Try parsing as text
            return ConversationParser.parse_text_conversation(file_path)

    @staticmethod
    def _parse_timestamp(timestamp: Union[str, int, float, None]) -> datetime:
        """Parse timestamp"""
        if timestamp is None:
            return datetime.now()

        try:
            if isinstance(timestamp, str):
                # Try ISO format
                if 'T' in timestamp:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Try other formats
                return datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
        except Exception:
            pass

        return datetime.now()

    @staticmethod
    def _extract_messages_from_json(data: Dict) -> List[Dict]:
        """Extract messages from JSON data"""
        messages = []

        # Claude Code format
        if 'chat_messages' in data:
            for msg in data['chat_messages']:
                messages.append({
                    'role': msg.get('sender', 'unknown'),
                    'content': msg.get('text', ''),
                    'timestamp': ConversationParser._parse_timestamp(msg.get('created_at')),
                    'metadata': msg
                })

        # Other possible formats
        elif 'messages' in data:
            for msg in data['messages']:
                messages.append({
                    'role': msg.get('role', msg.get('sender', 'unknown')),
                    'content': msg.get('content', msg.get('text', '')),
                    'timestamp': ConversationParser._parse_timestamp(msg.get('timestamp', msg.get('created_at'))),
                    'metadata': msg
                })

        return messages

    @staticmethod
    def _extract_title_from_messages(messages: List[Dict]) -> str:
        """Extract title from messages"""
        if not messages:
            return "Untitled Conversation"

        # Use first 50 characters of first user message as title
        for msg in messages:
            if msg.get('role') in ['user', 'human']:
                content = msg.get('content', '')
                if content:
                    title = content[:50].strip()
                    if len(content) > 50:
                        title += "..."
                    return title

        return "Untitled Conversation"

    @staticmethod
    def _extract_title_from_text(content: str) -> str:
        """Extract title from text content"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                title = line[:50]
                if len(line) > 50:
                    title += "..."
                return title

        return "Untitled Conversation"

    @staticmethod
    def _parse_text_messages(content: str) -> List[Dict]:
        """Parse plain text format messages"""
        messages = []

        # Simple split strategy: split by blank lines or specific markers
        sections = re.split(r'\n\s*\n', content)

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Try to identify role (supports bilingual role markers: Chinese and English)
            role = 'unknown'
            # Check for user role markers (bilingual)
            if section.lower().startswith(('user:', 'human:', '用户:')):
                role = 'user'
                section = re.sub(r'^(user|human|用户):\s*', '', section, flags=re.IGNORECASE)
            # Check for assistant role markers (bilingual)
            elif section.lower().startswith(('assistant:', 'ai:', 'claude:', '助手:', 'AI:', '助手')):
                role = 'assistant'
                section = re.sub(r'^(assistant|ai|claude|助手|AI):\s*', '', section, flags=re.IGNORECASE)

            messages.append({
                'role': role,
                'content': section,
                'timestamp': datetime.now(),
                'metadata': {'index': i}
            })

        return messages