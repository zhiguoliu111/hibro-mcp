"""
Claude Code IDE Integration
Refactor existing Claude Code specific functionality into plugin form
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


class ClaudeCodeIntegration(IDEIntegration):
    """Claude Code IDE integration"""

    def __init__(self, config):
        super().__init__(config)
        self.context_injector = ContextInjectorFactory.create_injector(
            'resource_update', config
        )

    def get_name(self) -> str:
        return "Claude Code"

    def get_version(self) -> str:
        """Get Claude Code version information"""
        try:
            # Try to get version from Claude Code configuration
            claude_dirs = self.get_conversation_directories()
            for claude_dir in claude_dirs:
                version_file = claude_dir.parent / 'version.json'
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        version_data = json.load(f)
                        return version_data.get('version', 'Unknown')
            return "Unknown"
        except Exception:
            return "Unknown"

    def is_available(self) -> bool:
        """Check if Claude Code is available"""
        try:
            # Check if Claude Code configuration directory exists
            claude_dirs = self.get_conversation_directories()
            return any(directory.exists() for directory in claude_dirs)
        except Exception:
            return False

    def get_conversation_directories(self) -> List[Path]:
        """Get Claude Code conversation file directories"""
        directories = []
        home = Path.home()

        # Claude Code project directory
        claude_projects = home / '.claude' / 'projects'
        if claude_projects.exists():
            directories.append(claude_projects)

        # Claude Code conversation directory (old version)
        claude_conversations = home / '.claude' / 'conversations'
        if claude_conversations.exists():
            directories.append(claude_conversations)

        # Platform-specific directories
        if platform.system() == "Windows":
            # Windows AppData directory
            appdata_conversations = home / 'AppData' / 'Roaming' / 'Claude' / 'conversations'
            if appdata_conversations.exists():
                directories.append(appdata_conversations)
        elif platform.system() == "Darwin":
            # macOS Application Support directory
            app_support_conversations = home / 'Library' / 'Application Support' / 'Claude' / 'conversations'
            if app_support_conversations.exists():
                directories.append(app_support_conversations)

        return directories

    def get_file_patterns(self) -> List[str]:
        """Get Claude Code conversation file matching patterns"""
        return [
            '*.json',      # Claude Code project files
            '*.jsonl',     # Conversation log files
            '**/chat.json' # Nested conversation files
        ]

    def parse_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse Claude Code conversation file"""
        try:
            # Use generic parser
            conversation = ConversationParser.auto_parse(file_path)

            if conversation:
                # Add Claude Code specific metadata
                conversation['metadata']['source'] = 'claude_code'
                conversation['metadata']['ide_version'] = self.get_version()

                # Try to extract project path (Claude Code specific)
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'project_path' in data:
                            conversation['project_path'] = data['project_path']

            return conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Claude Code conversation file {file_path}: {e}")
            return None

    def inject_context(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context into Claude Code"""
        if self.context_injector:
            return self.context_injector.inject(context, target_path)
        else:
            self.logger.error("Context injector not initialized")
            return False

    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """Get Claude Code project context"""
        context = {
            'project_name': project_path.name,
            'project_path': str(project_path),
            'workspace_files': [],
            'active_files': [],
            'git_info': {}
        }

        try:
            # Find Claude Code project configuration
            claude_projects_dir = Path.home() / '.claude' / 'projects'
            if claude_projects_dir.exists():
                for project_dir in claude_projects_dir.iterdir():
                    if project_dir.is_dir():
                        # Find project configuration file
                        config_files = list(project_dir.glob('*.json'))
                        for config_file in config_files:
                            try:
                                with open(config_file, 'r', encoding='utf-8') as f:
                                    project_data = json.load(f)
                                    if project_data.get('project_path') == str(project_path):
                                        # Found matching project
                                        context['workspace_files'] = project_data.get('workspace_files', [])
                                        context['active_files'] = project_data.get('active_files', [])
                                        break
                            except Exception:
                                continue

            # Get Git information
            git_dir = project_path / '.git'
            if git_dir.exists():
                try:
                    # Read current branch
                    head_file = git_dir / 'HEAD'
                    if head_file.exists():
                        with open(head_file, 'r') as f:
                            head_content = f.read().strip()
                            if head_content.startswith('ref: refs/heads/'):
                                context['git_info']['branch'] = head_content[16:]
                            else:
                                context['git_info']['branch'] = head_content[:8]  # Short hash

                    # Read remote repository information
                    config_file = git_dir / 'config'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_content = f.read()
                            # Simple parsing of remote repository URL
                            for line in config_content.split('\n'):
                                if 'url =' in line:
                                    context['git_info']['remote_url'] = line.split('=', 1)[1].strip()
                                    break

                except Exception as e:
                    self.logger.debug(f"Failed to get Git information: {e}")

        except Exception as e:
            self.logger.error(f"Failed to get Claude Code project context: {e}")

        return context

    def get_supported_features(self) -> List[str]:
        """Get Claude Code supported features"""
        return [
            'conversation_monitoring',
            'context_injection',
            'project_integration',
            'real_time_sync',
            'resource_updates'
        ]

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """Validate Claude Code configuration"""
        is_valid, errors = super().validate_configuration()

        # Claude Code specific validation
        try:
            # Check if there are accessible conversation directories
            accessible_dirs = [d for d in self.get_conversation_directories() if d.exists()]
            if not accessible_dirs:
                errors.append("No accessible Claude Code conversation directory found")

            # Check permission configuration
            claude_settings = Path.home() / '.claude' / 'settings.local.json'
            if claude_settings.exists():
                try:
                    with open(claude_settings, 'r') as f:
                        settings = json.load(f)
                        permissions = settings.get('permissions', {}).get('allow', [])

                        # Check if there are hibro related permissions
                        hibro_permissions = [p for p in permissions if 'hibro' in p]
                        if not hibro_permissions:
                            errors.append("No hibro related permissions found in Claude Code permissions configuration")

                except Exception as e:
                    errors.append(f"Unable to read Claude Code permissions configuration: {e}")

        except Exception as e:
            errors.append(f"Claude Code configuration validation failed: {e}")

        return len(errors) == 0, errors

    def monitor_conversations(self, callback):
        """Monitor Claude Code conversation file changes"""
        # Use parent class's default implementation, based on watchdog
        return super().monitor_conversations(callback)

    def __str__(self) -> str:
        return f"Claude Code Integration (v{self.get_version()})"