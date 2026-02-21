"""
Cursor IDE Integration
Shows how to adapt Cursor editor
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform

from ..base.ide_integration import IDEIntegration
from ..base.conversation_parser import ConversationParser
from ..base.context_injector import ContextInjectorFactory

logger = logging.getLogger(__name__)


class CursorIntegration(IDEIntegration):
    """Cursor IDE integration"""

    def __init__(self, config):
        super().__init__(config)
        self.context_injector = ContextInjectorFactory.create_injector(
            'file_injection', config
        )

    def get_name(self) -> str:
        return "Cursor"

    def get_version(self) -> str:
        """Get Cursor version information"""
        try:
            # Cursor usually has configuration files in user directory
            cursor_config_dirs = self._get_cursor_config_dirs()
            for config_dir in cursor_config_dirs:
                version_file = config_dir / 'version.json'
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        version_data = json.load(f)
                        return version_data.get('version', 'Unknown')
            return "Unknown"
        except Exception:
            return "Unknown"

    def is_available(self) -> bool:
        """Check if Cursor is available"""
        try:
            # Check if Cursor configuration directory or executable exists
            cursor_dirs = self.get_conversation_directories()
            if any(directory.exists() for directory in cursor_dirs):
                return True

            # Check if Cursor executable exists
            cursor_executables = self._get_cursor_executables()
            return any(exe.exists() for exe in cursor_executables)
        except Exception:
            return False

    def _get_cursor_config_dirs(self) -> List[Path]:
        """Get Cursor configuration directories"""
        home = Path.home()
        config_dirs = []

        if platform.system() == "Windows":
            config_dirs.extend([
                home / 'AppData' / 'Roaming' / 'Cursor',
                home / 'AppData' / 'Local' / 'Cursor'
            ])
        elif platform.system() == "Darwin":
            config_dirs.extend([
                home / 'Library' / 'Application Support' / 'Cursor',
                home / '.cursor'
            ])
        else:  # Linux
            config_dirs.extend([
                home / '.config' / 'cursor',
                home / '.cursor'
            ])

        return config_dirs

    def _get_cursor_executables(self) -> List[Path]:
        """Get Cursor executable file paths"""
        executables = []

        if platform.system() == "Windows":
            executables.extend([
                Path("C:/Users") / Path.home().name / "AppData/Local/Programs/cursor/Cursor.exe",
                Path("C:/Program Files/Cursor/Cursor.exe")
            ])
        elif platform.system() == "Darwin":
            executables.append(Path("/Applications/Cursor.app"))
        else:  # Linux
            executables.extend([
                Path("/usr/bin/cursor"),
                Path("/usr/local/bin/cursor"),
                Path.home() / ".local/bin/cursor"
            ])

        return executables

    def get_conversation_directories(self) -> List[Path]:
        """Get Cursor conversation file directories"""
        directories = []

        # Cursor may store conversations in different locations
        config_dirs = self._get_cursor_config_dirs()
        for config_dir in config_dirs:
            # Common conversation storage locations
            potential_dirs = [
                config_dir / 'conversations',
                config_dir / 'chats',
                config_dir / 'ai_conversations',
                config_dir / 'User' / 'conversations'
            ]

            for dir_path in potential_dirs:
                if dir_path.exists():
                    directories.append(dir_path)

        return directories

    def get_file_patterns(self) -> List[str]:
        """Get Cursor conversation file matching patterns"""
        return [
            '*.json',      # JSON format conversations
            '*.jsonl',     # JSONL format conversations
            '*.chat',      # Possible chat file format
            '**/chat_*.json'  # Nested conversation files
        ]

    def parse_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse Cursor conversation file"""
        try:
            # Use generic parser
            conversation = ConversationParser.auto_parse(file_path)

            if conversation:
                # Add Cursor specific metadata
                conversation['metadata']['source'] = 'cursor'
                conversation['metadata']['ide_version'] = self.get_version()

                # Cursor may have specific file format, add special handling here
                if file_path.suffix == '.chat':
                    # Handle Cursor's proprietary .chat format
                    conversation = self._parse_cursor_chat_format(file_path, conversation)

            return conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Cursor conversation file {file_path}: {e}")
            return None

    def _parse_cursor_chat_format(self, file_path: Path, base_conversation: Dict) -> Dict:
        """Parse Cursor's proprietary .chat format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Assume Cursor uses specific format, adjust according to actual format
            # This is just an example implementation
            messages = []
            current_message = None

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('User:'):
                    if current_message:
                        messages.append(current_message)
                    current_message = {
                        'role': 'user',
                        'content': line[5:].strip(),
                        'timestamp': datetime.now()
                    }
                elif line.startswith('Assistant:'):
                    if current_message:
                        messages.append(current_message)
                    current_message = {
                        'role': 'assistant',
                        'content': line[10:].strip(),
                        'timestamp': datetime.now()
                    }
                elif current_message and line:
                    current_message['content'] += '\n' + line

            if current_message:
                messages.append(current_message)

            base_conversation['messages'] = messages
            return base_conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Cursor .chat format: {e}")
            return base_conversation

    def inject_context(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context into Cursor"""
        try:
            if target_path is None:
                # Create Cursor specific context file
                cursor_dirs = self._get_cursor_config_dirs()
                for cursor_dir in cursor_dirs:
                    if cursor_dir.exists():
                        context_file = cursor_dir / 'hibro_context.json'
                        target_path = context_file
                        break

            if target_path and self.context_injector:
                # Wrap context in Cursor recognizable format
                cursor_context = {
                    'type': 'hibro_context',
                    'timestamp': datetime.now().isoformat(),
                    'content': context
                }

                formatted_context = json.dumps(cursor_context, indent=2, ensure_ascii=False)
                return self.context_injector.inject(formatted_context, target_path)

            return False

        except Exception as e:
            self.logger.error(f"Failed to inject context into Cursor: {e}")
            return False

    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """Get Cursor project context"""
        context = {
            'project_name': project_path.name,
            'project_path': str(project_path),
            'workspace_files': [],
            'active_files': [],
            'git_info': {}
        }

        try:
            # Find Cursor workspace configuration
            cursor_workspace_files = [
                project_path / '.cursor' / 'workspace.json',
                project_path / '.vscode' / 'settings.json',  # Cursor compatible with VSCode config
                project_path / 'cursor.json'
            ]

            for workspace_file in cursor_workspace_files:
                if workspace_file.exists():
                    try:
                        with open(workspace_file, 'r', encoding='utf-8') as f:
                            workspace_data = json.load(f)

                            # Extract workspace file list
                            if 'files' in workspace_data:
                                context['workspace_files'] = workspace_data['files']
                            elif 'folders' in workspace_data:
                                # VSCode format
                                context['workspace_files'] = [
                                    folder.get('path', '') for folder in workspace_data['folders']
                                ]

                            break
                    except Exception:
                        continue

            # Get Git information (similar to Claude Code)
            git_dir = project_path / '.git'
            if git_dir.exists():
                try:
                    head_file = git_dir / 'HEAD'
                    if head_file.exists():
                        with open(head_file, 'r') as f:
                            head_content = f.read().strip()
                            if head_content.startswith('ref: refs/heads/'):
                                context['git_info']['branch'] = head_content[16:]
                            else:
                                context['git_info']['branch'] = head_content[:8]

                    config_file = git_dir / 'config'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_content = f.read()
                            for line in config_content.split('\n'):
                                if 'url =' in line:
                                    context['git_info']['remote_url'] = line.split('=', 1)[1].strip()
                                    break

                except Exception as e:
                    self.logger.debug(f"Failed to get Git information: {e}")

        except Exception as e:
            self.logger.error(f"Failed to get Cursor project context: {e}")

        return context

    def get_supported_features(self) -> List[str]:
        """Get Cursor supported features"""
        return [
            'conversation_monitoring',
            'context_injection',
            'project_integration',
            'file_injection'
        ]

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """Validate Cursor configuration"""
        is_valid, errors = super().validate_configuration()

        # Cursor specific validation
        try:
            # Check if Cursor is installed
            if not self.is_available():
                errors.append("Cursor not installed or not accessible")

            # Check configuration directory permissions
            config_dirs = self._get_cursor_config_dirs()
            accessible_dirs = [d for d in config_dirs if d.exists()]
            if not accessible_dirs:
                errors.append("Unable to access Cursor configuration directory")

        except Exception as e:
            errors.append(f"Cursor configuration validation failed: {e}")

        return len(errors) == 0, errors

    def __str__(self) -> str:
        return f"Cursor Integration (v{self.get_version()})"