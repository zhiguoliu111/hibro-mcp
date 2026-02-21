"""
Trae IDE Integration
Shows how to adapt Trae editor
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


class TraeIntegration(IDEIntegration):
    """Trae IDE integration"""

    def __init__(self, config):
        super().__init__(config)
        self.context_injector = ContextInjectorFactory.create_injector(
            'websocket', config,
            websocket_url=self._get_trae_websocket_url()
        )

    def get_name(self) -> str:
        return "Trae"

    def get_version(self) -> str:
        """Get Trae version information"""
        try:
            # Try to get version from Trae configuration file
            trae_config_dirs = self._get_trae_config_dirs()
            for config_dir in trae_config_dirs:
                version_file = config_dir / 'app.json'
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        app_data = json.load(f)
                        return app_data.get('version', 'Unknown')
            return "Unknown"
        except Exception:
            return "Unknown"

    def _get_trae_config_dirs(self) -> List[Path]:
        """Get Trae configuration directories"""
        home = Path.home()
        config_dirs = []

        if platform.system() == "Windows":
            config_dirs.extend([
                home / 'AppData' / 'Roaming' / 'Trae',
                home / 'AppData' / 'Local' / 'Trae'
            ])
        elif platform.system() == "Darwin":
            config_dirs.extend([
                home / 'Library' / 'Application Support' / 'Trae',
                home / '.trae'
            ])
        else:  # Linux
            config_dirs.extend([
                home / '.config' / 'trae',
                home / '.trae'
            ])

        return config_dirs

    def _get_trae_websocket_url(self) -> Optional[str]:
        """Get Trae WebSocket URL"""
        # Trae may communicate through WebSocket for real-time communication
        potential_urls = [
            "ws://localhost:9000/ws",
            "ws://localhost:8765/ws",
            "ws://127.0.0.1:9000/ws"
        ]

        for url in potential_urls:
            try:
                import websocket
                ws = websocket.create_connection(url, timeout=2)
                ws.close()
                return url
            except Exception:
                continue

        return None

    def is_available(self) -> bool:
        """Check if Trae is available"""
        try:
            # Check Trae configuration directory
            trae_dirs = self.get_conversation_directories()
            if any(directory.exists() for directory in trae_dirs):
                return True

            # Check WebSocket connection
            return self._get_trae_websocket_url() is not None
        except Exception:
            return False

    def get_conversation_directories(self) -> List[Path]:
        """Get Trae conversation file directories"""
        directories = []

        config_dirs = self._get_trae_config_dirs()
        for config_dir in config_dirs:
            # Trae possible session storage locations
            potential_dirs = [
                config_dir / 'sessions',
                config_dir / 'conversations',
                config_dir / 'ai_sessions',
                config_dir / 'data' / 'sessions'
            ]

            for dir_path in potential_dirs:
                if dir_path.exists():
                    directories.append(dir_path)

        return directories

    def get_file_patterns(self) -> List[str]:
        """Get Trae conversation file matching patterns"""
        return [
            '*.json',
            '*.session',
            '*.trae',
            '**/session_*.json',
            '**/chat_*.session'
        ]

    def parse_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse Trae conversation file"""
        try:
            conversation = ConversationParser.auto_parse(file_path)

            if conversation:
                conversation['metadata']['source'] = 'trae'
                conversation['metadata']['ide_version'] = self.get_version()

                # Trae specific format handling
                if file_path.suffix == '.session':
                    conversation = self._parse_trae_session_format(file_path, conversation)

            return conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Trae conversation file {file_path}: {e}")
            return None

    def _parse_trae_session_format(self, file_path: Path, base_conversation: Dict) -> Dict:
        """Parse Trae's proprietary .session format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Assume Trae uses specific session format
            messages = []
            if 'interactions' in data:
                for interaction in data['interactions']:
                    # User input
                    if 'user_input' in interaction:
                        messages.append({
                            'role': 'user',
                            'content': interaction['user_input'],
                            'timestamp': self._parse_trae_timestamp(interaction.get('timestamp')),
                            'metadata': {'interaction_id': interaction.get('id')}
                        })

                    # AI response
                    if 'ai_response' in interaction:
                        messages.append({
                            'role': 'assistant',
                            'content': interaction['ai_response'],
                            'timestamp': self._parse_trae_timestamp(interaction.get('response_timestamp')),
                            'metadata': {'interaction_id': interaction.get('id')}
                        })

            base_conversation['messages'] = messages
            base_conversation['title'] = data.get('session_title', base_conversation['title'])
            base_conversation['project_path'] = data.get('workspace_path')

            return base_conversation

        except Exception as e:
            self.logger.error(f"Failed to parse Trae .session format: {e}")
            return base_conversation

    def _parse_trae_timestamp(self, timestamp) -> datetime:
        """Parse Trae timestamp"""
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                pass
        elif isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp / 1000)  # Assume millisecond timestamp
            except Exception:
                pass
        return datetime.now()

    def inject_context(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context into Trae"""
        if self.context_injector:
            return self.context_injector.inject(context, target_path)
        else:
            # Fallback: file injection
            try:
                trae_dirs = self.get_conversation_directories()
                if trae_dirs:
                    context_file = trae_dirs[0] / 'hibro_context.session'
                    trae_context = {
                        'type': 'context_injection',
                        'source': 'hibro',
                        'timestamp': datetime.now().isoformat(),
                        'content': context,
                        'metadata': {
                            'injected_by': 'hibro_mcp',
                            'version': '2.0.0'
                        }
                    }

                    with open(context_file, 'w', encoding='utf-8') as f:
                        json.dump(trae_context, f, indent=2, ensure_ascii=False)

                    return True
            except Exception as e:
                self.logger.error(f"Failed to inject context into Trae: {e}")

        return False

    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """Get Trae project context"""
        context = {
            'project_name': project_path.name,
            'project_path': str(project_path),
            'workspace_files': [],
            'active_files': [],
            'git_info': {}
        }

        try:
            # Find Trae workspace configuration
            trae_workspace_files = [
                project_path / '.trae' / 'workspace.json',
                project_path / 'trae.workspace.json',
                project_path / '.trae.json'
            ]

            for workspace_file in trae_workspace_files:
                if workspace_file.exists():
                    try:
                        with open(workspace_file, 'r', encoding='utf-8') as f:
                            workspace_data = json.load(f)

                            context['workspace_files'] = workspace_data.get('included_files', [])
                            context['active_files'] = workspace_data.get('open_files', [])

                            # Trae may have special project settings
                            if 'project_settings' in workspace_data:
                                context['trae_settings'] = workspace_data['project_settings']

                            break
                    except Exception:
                        continue

            # Git information retrieval (standard implementation)
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
            self.logger.error(f"Failed to get Trae project context: {e}")

        return context

    def get_supported_features(self) -> List[str]:
        """Get Trae supported features"""
        features = ['conversation_monitoring', 'project_integration']

        # Add features based on WebSocket availability
        if self._get_trae_websocket_url():
            features.extend(['context_injection', 'real_time_sync', 'websocket_communication'])
        else:
            features.append('file_injection')

        return features

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """Validate Trae configuration"""
        is_valid, errors = super().validate_configuration()

        try:
            if not self.is_available():
                errors.append("Trae not installed or not accessible")

            # Check WebSocket connection
            websocket_url = self._get_trae_websocket_url()
            if websocket_url:
                try:
                    import websocket
                    ws = websocket.create_connection(websocket_url, timeout=5)
                    ws.close()
                except Exception:
                    errors.append("Trae WebSocket connection error")

        except Exception as e:
            errors.append(f"Trae configuration validation failed: {e}")

        return len(errors) == 0, errors

    def monitor_conversations(self, callback):
        """Monitor Trae conversation file changes"""
        # If WebSocket connection exists, use real-time monitoring
        websocket_url = self._get_trae_websocket_url()
        if websocket_url:
            return self._monitor_via_websocket(callback)
        else:
            # Use file system monitoring
            return super().monitor_conversations(callback)

    def _monitor_via_websocket(self, callback):
        """Monitor Trae session changes through WebSocket"""
        try:
            import websocket
            import threading

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if data.get('type') == 'session_update':
                        # Handle session update event
                        session_file = Path(data.get('session_file', ''))
                        if session_file.exists():
                            callback(session_file, 'modified')
                except Exception as e:
                    self.logger.error(f"Failed to handle WebSocket message: {e}")

            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket connection closed")

            def on_open(ws):
                # Subscribe to session update events
                subscribe_message = {
                    'type': 'subscribe',
                    'events': ['session_update', 'conversation_change']
                }
                ws.send(json.dumps(subscribe_message))

            websocket_url = self._get_trae_websocket_url()
            ws = websocket.WebSocketApp(
                websocket_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            # Run WebSocket in background thread
            def run_websocket():
                ws.run_forever()

            thread = threading.Thread(target=run_websocket, daemon=True)
            thread.start()

            return ws

        except Exception as e:
            self.logger.error(f"Failed to start WebSocket monitoring: {e}")
            # Fallback to file system monitoring
            return super().monitor_conversations(callback)

    def __str__(self) -> str:
        return f"Trae Integration (v{self.get_version()})"