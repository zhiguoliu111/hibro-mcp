"""
Integration module
Provides dynamic loading, management, and coordination of IDE integrations
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type
from .base.ide_integration import IDEIntegration

logger = logging.getLogger(__name__)


class IDEManager:
    """IDE integration manager"""

    def __init__(self, config):
        self.config = config
        self.integrations: Dict[str, IDEIntegration] = {}
        self.available_integrations: Dict[str, Type[IDEIntegration]] = {}
        self._load_integrations()

    def _load_integrations(self):
        """Dynamically load all available IDE integrations"""
        integration_modules = [
            'claude_code',
            'cursor',
            'qoder',
            'trae'
        ]

        for module_name in integration_modules:
            try:
                # Dynamically import module
                module_path = f'hibro.integrations.{module_name}'
                module = importlib.import_module(module_path)

                # Get integration class (naming convention: {ModuleName}Integration)
                # Special handling for claude_code -> ClaudeCode
                if module_name == 'claude_code':
                    class_name = 'ClaudeCodeIntegration'
                else:
                    # For others: cursor -> Cursor, qoder -> Qoder, trae -> Trae
                    class_name = f'{module_name.title()}Integration'

                integration_class = getattr(module, class_name)

                # Register available integration class
                self.available_integrations[module_name] = integration_class

                # Try to create instance and check availability
                try:
                    integration = integration_class(self.config)
                    if integration.is_available():
                        self.integrations[module_name] = integration
                        logger.info(f"Successfully loaded IDE integration: {module_name}")
                    else:
                        logger.debug(f"IDE integration not available: {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to create IDE integration instance {module_name}: {e}")

            except ImportError as e:
                logger.debug(f"IDE integration module does not exist: {module_name} ({e})")
            except AttributeError as e:
                logger.warning(f"IDE integration class does not exist: {module_name}.{class_name} ({e})")
            except Exception as e:
                logger.error(f"Failed to load IDE integration: {module_name} ({e})")

    def get_active_integration(self) -> Optional[IDEIntegration]:
        """Get current active IDE integration"""
        ide_type = getattr(self.config.ide, 'type', 'auto')

        if ide_type == "auto":
            # Auto-detect: return first available integration by priority
            priority_order = ['claude_code', 'cursor', 'qoder', 'trae']

            for ide_name in priority_order:
                if ide_name in self.integrations:
                    integration = self.integrations[ide_name]
                    if integration.is_available():
                        logger.info(f"Auto-selected IDE integration: {ide_name}")
                        return integration

            # If not in priority list, return any available one
            for integration in self.integrations.values():
                if integration.is_available():
                    logger.info(f"Auto-selected IDE integration: {integration.get_name()}")
                    return integration

        else:
            # Use specified IDE
            integration = self.integrations.get(ide_type)
            if integration and integration.is_available():
                return integration
            else:
                logger.warning(f"Specified IDE integration not available: {ide_type}")

        return None

    def get_integration(self, ide_name: str) -> Optional[IDEIntegration]:
        """Get specified IDE integration"""
        return self.integrations.get(ide_name)

    def get_available_integrations(self) -> Dict[str, IDEIntegration]:
        """Get all available IDE integrations"""
        return {name: integration for name, integration in self.integrations.items()
                if integration.is_available()}

    def get_integration_info(self) -> List[Dict[str, str]]:
        """Get information for all IDE integrations"""
        info = []

        for name, integration_class in self.available_integrations.items():
            try:
                integration = self.integrations.get(name)
                if integration:
                    available = integration.is_available()
                    version = integration.get_version()
                else:
                    # Try creating temporary instance to get info
                    temp_integration = integration_class(self.config)
                    available = temp_integration.is_available()
                    version = temp_integration.get_version() if available else "Unknown"

                info.append({
                    'name': name,
                    'display_name': integration_class.__name__.replace('Integration', ''),
                    'available': str(available),
                    'version': version,
                    'status': 'Active' if name in self.integrations and available else 'Inactive'
                })
            except Exception as e:
                info.append({
                    'name': name,
                    'display_name': name.title(),
                    'available': 'False',
                    'version': 'Unknown',
                    'status': f'Error: {e}'
                })

        return info

    def reload_integrations(self):
        """Reload all IDE integrations"""
        logger.info("Reloading IDE integrations...")
        self.integrations.clear()
        self.available_integrations.clear()
        self._load_integrations()

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration for all IDE integrations"""
        validation_results = {}

        for name, integration in self.integrations.items():
            is_valid, errors = integration.validate_configuration()
            if not is_valid:
                validation_results[name] = errors

        return validation_results

    def get_supported_features(self) -> Dict[str, List[str]]:
        """Get features supported by all IDE integrations"""
        features = {}

        for name, integration in self.integrations.items():
            features[name] = integration.get_supported_features()

        return features

    def start_monitoring(self, callback):
        """Start monitoring conversation files for all available IDEs"""
        active_integration = self.get_active_integration()
        if active_integration:
            logger.info(f"Starting to monitor conversation files for {active_integration.get_name()}")
            return active_integration.monitor_conversations(callback)
        else:
            logger.warning("No available IDE integration, cannot start monitoring")
            return None

    def inject_context_to_active_ide(self, context: str, target_path: Optional[Path] = None) -> bool:
        """Inject context to current active IDE"""
        active_integration = self.get_active_integration()
        if active_integration:
            return active_integration.inject_context(context, target_path)
        else:
            logger.warning("No available IDE integration, cannot inject context")
            return False

    def get_project_context_from_active_ide(self, project_path: Path) -> Optional[Dict]:
        """Get project context from current active IDE"""
        active_integration = self.get_active_integration()
        if active_integration:
            return active_integration.get_project_context(project_path)
        else:
            logger.warning("No available IDE integration, cannot get project context")
            return None

    def __str__(self) -> str:
        active = self.get_active_integration()
        active_name = active.get_name() if active else "None"
        return f"IDEManager(active={active_name}, available={len(self.integrations)})"

    def __repr__(self) -> str:
        return f"<IDEManager(integrations={list(self.integrations.keys())})>"
