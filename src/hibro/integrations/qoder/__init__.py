"""
Qoder IDE Integration
Shows how to adapt Qoder editor
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


class QoderIntegration(IDEIntegration):
    """Qoder IDE integration"""

    def __init__(self, config):
        super().__init__(config)
        self.context_injector = ContextInjectorFactory.create_injector(
            'api_call', config,
            api_endpoint=self._get_qoder_api_endpoint()
        )

    def get_name(self) -> str:
        return "Qoder"

    def get_version(self) -> str:
        """Get Qoder version information"""
        try:
            # Try to get version information through API
            import requests
            api_endpoint = self._get_qoder_api_endpoint()
            if api_endpoint:
                response = requests.get(f"{api_endpoint}/version", timeout=5)
                if response.status_code == 200:
                    return response.json().get('version', 'Unknown')
            return "Unknown"
        except Exception:
            return "Unknown"

    def _get_qoder_api_endpoint(self) -> Optional[str]:
        """Get Qoder API endpoint"""
        # Qoder may communicate through local API service
        potential_endpoints = [
            "http://localhost:8080",
            "http://localhost:3000",
            "http://127.0.0.1:8080"
        ]

        for endpoint in potential_endpoints:
            try:
                import requests
                response = requests.get(f"{endpoint}/health", timeout=2)
                if response.status_code == 200:
                    return endpoint
            except Exception:
                continue

        return None

    def is_available(self) -> bool:
        """Check if Qoder is available"""
        try:
            # Check Qoder configuration directory
            qoder_dirs = self.get_conversation_directories()
            if any(directory.exists() for directory in qoder_dirs):
                return True

            # Check if Qoder API is available
            return self._get_qoder_api_endpoint() is not None
        except Exception:
            return False

    def get_conversation_directories(self) -> List[Path]:
        """Get Qoder conversation file directories"""
        directories = []
        home = Path.home()

        # Qoder possible configuration directories
        if platform.system() == "Windows":
            qoder_dirs = [
                home / 'AppData' / 'Roaming' / 'Qoder' / 'chats',
                home / 'AppData' / 'Local' / 'Qoder' / 'conversations'
            ]
        elif platform.system() == "Darwin":
            qoder_dirs = [
                home / 'Library' / 'Application Support' / 'Qoder' / 'chats',
                home / '.qoder' / 'chats'
            ]
        else:  # Linux
            qoder_dirs = [
                home / '.config' / 'qoder' / 'chats',
                home / '.qoder' / 'chats'
            ]

        for qoder_dir in qoder_dirs:
            if qoder_dir.exists():
                directories.append(qoder_dir)

        return directories

    def get_file_patterns(self) -> List[str]:
        """Get Qoder conversation file matching patterns"""
        return [
            '*.json',
            '*.chat',
            '*.qoder',
            '**/session_*.json'
        ]

    def parse_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse Qoder conversation file"""
        try:
            conversation = ConversationParser.auto_parse(file_path)

            if conversation:
                conversation['metadata']['source'] = 'qoder'
                conversation['metadata']['ide_version'] = self.get_version()

                # Qoder specific format handling
                if file_path.suffix == '.qoder':
                    conversation = self._parse_qoder_format(file_path, conversation)

            return conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Qoder conversation file {file_path}: {e}")
            return None

    def _parse_qoder_format(self, file_path: Path, base_conversation: Dict) -> Dict:
        """Parse Qoder proprietary format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Assume Qoder uses specific JSON structure
            messages = []
            if 'dialogue' in data:
                for entry in data['dialogue']:
                    messages.append({
                        'role': entry.get('speaker', 'unknown'),
                        'content': entry.get('message', ''),
                        'timestamp': self._parse_qoder_timestamp(entry.get('timestamp')),
                        'metadata': entry
                    })

            base_conversation['messages'] = messages
            base_conversation['title'] = data.get('session_name', base_conversation['title'])

            return base_conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Qoder format: {e}")
            return base_conversation

    def _parse_qoder_timestamp(self, timestamp) -> datetime:
        """Parse Qoder timestamp"""
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp)
            except Exception:
                pass
        elif isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp)
            except Exception:
                pass
        return datetime.now()

    def inject_context(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context into Qoder"""
        if self.context_injector:
            return self.context_injector.inject(context, target_path)
        else:
            # Fallback: file injection
            try:
                qoder_dirs = self.get_conversation_directories()
                if qoder_dirs:
                    context_file = qoder_dirs[0] / 'hibro_context.json'
                    qoder_context = {
                        'type': 'hibro_context',
                        'timestamp': datetime.now().isoformat(),
                        'content': context
                    }

                    with open(context_file, 'w', encoding='utf-8') as f:
                        json.dump(qoder_context, f, indent=2, ensure_ascii=False)

                    return True
            except Exception as e:
                self.logger.error(f"Failed to inject context into Qoder: {e}")

        return False

    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """Get Qoder project context"""
        context = {
            'project_name': project_path.name,
            'project_path': str(project_path),
            'workspace_files': [],
            'active_files': [],
            'git_info': {}
        }

        try:
            # Find Qoder project configuration
            qoder_config_files = [
                project_path / '.qoder' / 'project.json',
                project_path / 'qoder.config.json'
            ]

            for config_file in qoder_config_files:
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            project_data = json.load(f)
                            context['workspace_files'] = project_data.get('files', [])
                            context['active_files'] = project_data.get('active_files', [])
                            break
                    except Exception:
                        continue

            # Get Git information (standard implementation)
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
                except Exception as e:
                    self.logger.debug(f"Failed to get Git information: {e}")

        except Exception as e:
            self.logger.error(f"Failed to get Qoder project context: {e}")

        return context

    def get_supported_features(self) -> List[str]:
        """Get Qoder supported features"""
        features = ['conversation_monitoring', 'project_integration']

        # Add features based on API availability
        if self._get_qoder_api_endpoint():
            features.extend(['context_injection', 'real_time_sync'])
        else:
            features.append('file_injection')

        return features

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """Validate Qoder configuration"""
        is_valid, errors = super().validate_configuration()

        try:
            if not self.is_available():
                errors.append("Qoder not installed or not accessible")

            # Check API connection
            api_endpoint = self._get_qoder_api_endpoint()
            if api_endpoint:
                try:
                    import requests
                    response = requests.get(f"{api_endpoint}/health", timeout=5)
                    if response.status_code != 200:
                        errors.append("Qoder API connection error")
                except Exception:
                    errors.append("Unable to connect to Qoder API")

        except Exception as e:
            errors.append(f"Qoder configuration validation failed: {e}")

        return len(errors) == 0, errors

    def __str__(self) -> str:
        return f"Qoder Integration (v{self.get_version()})"