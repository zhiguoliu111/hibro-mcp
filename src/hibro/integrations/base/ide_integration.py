"""
IDE Integration Abstract Base Class
Defines interfaces that all IDE integrations must implement
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IDEIntegration(ABC):
    """Abstract base class for IDE integration"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_name(self) -> str:
        """Get IDE name"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get IDE version information"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if IDE is available (installed and accessible)"""
        pass

    @abstractmethod
    def get_conversation_directories(self) -> List[Path]:
        """Get list of conversation file directories"""
        pass

    @abstractmethod
    def get_file_patterns(self) -> List[str]:
        """Get matching patterns for conversation files"""
        pass

    @abstractmethod
    def parse_conversation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse conversation file, return standardized format

        Return format:
        {
            'id': str,                    # Conversation ID
            'title': str,                 # Conversation title
            'created_at': datetime,       # Creation time
            'updated_at': datetime,       # Update time
            'messages': List[Dict],       # Message list
            'project_path': Optional[str], # Project path
            'metadata': Dict              # Other metadata
        }
        """
        pass

    @abstractmethod
    def inject_context(self, context: str, target_path: Optional[Path] = None) -> bool:
        """
        Inject context information into IDE

        Args:
            context: Context content to inject
            target_path: Target path (optional)

        Returns:
            bool: Whether injection was successful
        """
        pass

    @abstractmethod
    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """
        Get project context information

        Returns:
            Dict containing project related information:
            - project_name: Project name
            - project_path: Project path
            - workspace_files: Workspace file list
            - active_files: Currently open files
            - git_info: Git information (if available)
        """
        pass

    def get_supported_features(self) -> List[str]:
        """
        Get list of supported features

        Possible features:
        - conversation_monitoring: Conversation monitoring
        - context_injection: Context injection
        - project_integration: Project integration
        - real_time_sync: Real-time synchronization
        """
        return []

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate if configuration is correct

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_message_list)
        """
        errors = []

        # Check if conversation directories exist
        for directory in self.get_conversation_directories():
            if not directory.exists():
                errors.append(f"Conversation directory does not exist: {directory}")

        return len(errors) == 0, errors

    def get_conversation_files(self) -> List[Path]:
        """Get all conversation files"""
        files = []
        patterns = self.get_file_patterns()

        for directory in self.get_conversation_directories():
            if not directory.exists():
                continue

            for pattern in patterns:
                files.extend(directory.glob(pattern))

        return files

    def monitor_conversations(self, callback):
        """
        Monitor conversation file changes (default implementation)
        Subclasses can override to provide more efficient monitoring methods
        """
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ConversationHandler(FileSystemEventHandler):
            def __init__(self, integration, callback):
                self.integration = integration
                self.callback = callback

            def on_modified(self, event):
                if event.is_directory:
                    return

                file_path = Path(event.src_path)
                if any(file_path.match(pattern) for pattern in self.integration.get_file_patterns()):
                    self.callback(file_path, 'modified')

            def on_created(self, event):
                if event.is_directory:
                    return

                file_path = Path(event.src_path)
                if any(file_path.match(pattern) for pattern in self.integration.get_file_patterns()):
                    self.callback(file_path, 'created')

        observer = Observer()
        handler = ConversationHandler(self, callback)

        for directory in self.get_conversation_directories():
            if directory.exists():
                observer.schedule(handler, str(directory), recursive=True)

        observer.start()
        return observer

    def __str__(self) -> str:
        return f"{self.get_name()} Integration"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.get_name()}', available={self.is_available()})>"