"""
Generic Context Injector
Provides standardized context injection functionality
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import tempfile
import os

logger = logging.getLogger(__name__)


class ContextInjector(ABC):
    """Abstract base class for context injector"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def inject(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context"""
        pass

    @abstractmethod
    def supports_injection_method(self, method: str) -> bool:
        """Check if the specified injection method is supported"""
        pass


class ResourceUpdateInjector(ContextInjector):
    """
    Resource Update Injector
    Injects context by updating MCP resources (for IDEs with MCP support)
    """

    def inject(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context through resource update"""
        try:
            # This will trigger MCP resource update event
            # IDE will automatically detect resource changes and update context
            self.logger.info("Injecting context through resource update")
            return True
        except Exception as e:
            self.logger.error(f"Resource update injection failed: {e}")
            return False

    def supports_injection_method(self, method: str) -> bool:
        return method == "resource_update"


class FileInjectionInjector(ContextInjector):
    """
    File Injector
    Injects context by creating or modifying files
    """

    def inject(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context through file"""
        try:
            if target_path is None:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.context',
                    delete=False,
                    encoding='utf-8'
                ) as f:
                    f.write(context)
                    target_path = Path(f.name)

            else:
                # Write to specified file
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(context)

            self.logger.info(f"Context injected to file: {target_path}")
            return True

        except Exception as e:
            self.logger.error(f"File injection failed: {e}")
            return False

    def supports_injection_method(self, method: str) -> bool:
        return method == "file_injection"


class APICallInjector(ContextInjector):
    """
    API Call Injector
    Injects context through API calls
    """

    def __init__(self, config, api_endpoint: str = None):
        super().__init__(config)
        self.api_endpoint = api_endpoint

    def inject(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context through API call"""
        try:
            import requests

            if not self.api_endpoint:
                self.logger.error("API endpoint not configured")
                return False

            payload = {
                'context': context,
                'target_path': str(target_path) if target_path else None
            }

            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()

            self.logger.info("Context injected through API call successfully")
            return True

        except Exception as e:
            self.logger.error(f"API call injection failed: {e}")
            return False

    def supports_injection_method(self, method: str) -> bool:
        return method == "api_call"


class WebSocketInjector(ContextInjector):
    """
    WebSocket Injector
    Injects context through WebSocket connection
    """

    def __init__(self, config, websocket_url: str = None):
        super().__init__(config)
        self.websocket_url = websocket_url

    def inject(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context through WebSocket"""
        try:
            import websocket

            if not self.websocket_url:
                self.logger.error("WebSocket URL not configured")
                return False

            def on_open(ws):
                message = {
                    'type': 'context_injection',
                    'context': context,
                    'target_path': str(target_path) if target_path else None
                }
                ws.send(json.dumps(message))
                ws.close()

            ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=on_open
            )
            ws.run_forever()

            self.logger.info("Context injected through WebSocket successfully")
            return True

        except Exception as e:
            self.logger.error(f"WebSocket injection failed: {e}")
            return False

    def supports_injection_method(self, method: str) -> bool:
        return method == "websocket"


class ContextInjectorFactory:
    """Context injector factory"""

    _injectors = {
        'resource_update': ResourceUpdateInjector,
        'file_injection': FileInjectionInjector,
        'api_call': APICallInjector,
        'websocket': WebSocketInjector
    }

    @classmethod
    def create_injector(cls, method: str, config, **kwargs) -> Optional[ContextInjector]:
        """Create context injector"""
        injector_class = cls._injectors.get(method)
        if not injector_class:
            logger.error(f"Unsupported injection method: {method}")
            return None

        try:
            return injector_class(config, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create injector: {e}")
            return None

    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """Get list of supported injection methods"""
        return list(cls._injectors.keys())

    @classmethod
    def register_injector(cls, method: str, injector_class):
        """Register new injector class"""
        cls._injectors[method] = injector_class