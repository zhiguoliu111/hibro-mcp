#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management module
Handles configuration file loading and management for the hibro system
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Memory-related configuration"""
    auto_learn: bool = True
    importance_threshold: float = 0.7
    max_memories: int = 100000


@dataclass
class ForgettingConfig:
    """Forgetting mechanism configuration"""
    time_decay_rate: float = 0.1
    min_importance: float = 0.3
    cleanup_interval_days: int = 7
    # Scheduled cleanup
    cleanup_time: str = "03:00"          # Daily cleanup time (HH:MM)
    cleanup_enabled: bool = True         # Enable scheduled cleanup
    # Threshold triggers
    threshold_warning: float = 0.7       # Warning threshold (70%)
    threshold_cleanup: float = 0.85      # Trigger cleanup threshold (85%)
    threshold_critical: float = 0.95     # Critical threshold (95%, block storage)


@dataclass
class IDEConfig:
    """IDE integration configuration"""
    type: str = "auto"  # auto, claude_code, cursor, qoder, trae
    auto_inject: bool = True
    context_limit_kb: int = 200
    monitor_conversations: bool = True
    injection_strategy: str = "smart"  # smart, always, manual


@dataclass
class IDESpecificConfig:
    """IDE-specific configuration"""
    conversation_dirs: list = field(default_factory=list)
    file_patterns: list = field(default_factory=list)
    context_injection: dict = field(default_factory=dict)


@dataclass
class IDEIntegrationConfig:
    """IDE integration configuration"""
    auto_inject: bool = True
    context_limit_kb: int = 200
    monitor_conversations: bool = True


@dataclass
class StorageConfig:
    """Storage configuration"""
    database_path: str = "~/.hibro/memories.db"
    max_size_gb: int = 10
    backup_enabled: bool = True
    backup_interval_hours: int = 24


@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_enabled: bool = True
    auto_cleanup_days: int = 365
    sensitive_data_filter: bool = True


class Config:
    """Main configuration class"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Configuration file path, if None use default path
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config_dir = Path.home() / '.hibro'

        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)

        # Load configuration
        self._load_config()

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve configuration file path"""
        if config_path:
            return Path(config_path).expanduser()
        return Path.home() / '.hibro' / 'config.yaml'

    def _load_config(self):
        """Load configuration file"""
        # Default configuration
        default_config = {
            'memory': {
                'auto_learn': True,
                'importance_threshold': 0.7,
                'max_memories': 100000
            },
            'forgetting': {
                'time_decay_rate': 0.1,
                'min_importance': 0.3,
                'cleanup_interval_days': 7,
                'cleanup_time': '03:00',
                'cleanup_enabled': True,
                'threshold_warning': 0.7,
                'threshold_cleanup': 0.85,
                'threshold_critical': 0.95
            },
            'ide': {
                'type': 'auto',
                'auto_inject': True,
                'context_limit_kb': 200,
                'monitor_conversations': True,
                'injection_strategy': 'smart'
            },
            'ide_specific': {
                'claude_code': {
                    'conversation_dirs': ['~/.claude/projects', '~/.claude/conversations'],
                    'file_patterns': ['*.json', '*.jsonl'],
                    'context_injection': {'method': 'resource_update'}
                },
                'cursor': {
                    'conversation_dirs': ['~/.cursor/conversations'],
                    'file_patterns': ['*.json', '*.chat'],
                    'context_injection': {'method': 'file_injection'}
                },
                'qoder': {
                    'conversation_dirs': ['~/.qoder/chats'],
                    'file_patterns': ['*.json', '*.qoder'],
                    'context_injection': {'method': 'api_call'}
                },
                'trae': {
                    'conversation_dirs': ['~/.trae/sessions'],
                    'file_patterns': ['*.json', '*.session'],
                    'context_injection': {'method': 'websocket'}
                }
            },
            'ide_integration': {
                'auto_inject': True,
                'context_limit_kb': 200,
                'monitor_conversations': True
            },
            'storage': {
                'database_path': '~/.hibro/memories.db',
                'max_size_gb': 10,
                'backup_enabled': True,
                'backup_interval_hours': 24
            },
            'security': {
                'encryption_enabled': True,
                'auto_cleanup_days': 365,
                'sensitive_data_filter': True
            }
        }

        # If config file exists, load and merge
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                # Deep merge configuration
                self._config_data = self._deep_merge(default_config, user_config)
            except Exception as e:
                print(f"Warning: Failed to load configuration file {self.config_path}: {e}")
                self._config_data = default_config
        else:
            # Use default configuration and create config file
            self._config_data = default_config
            self._create_default_config()

        # Create configuration objects
        self.memory = MemoryConfig(**self._config_data['memory'])
        self.forgetting = ForgettingConfig(**self._config_data['forgetting'])
        self.ide_integration = IDEIntegrationConfig(**self._config_data['ide_integration'])
        self.ide = IDEConfig(**self._config_data.get('ide', {}))
        self.ide_specific = {
            name: IDESpecificConfig(**config)
            for name, config in self._config_data.get('ide_specific', {}).items()
        }
        self.storage = StorageConfig(**self._config_data['storage'])
        self.security = SecurityConfig(**self._config_data['security'])

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _create_default_config(self):
        """Create default configuration file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False,
                         allow_unicode=True, indent=2)
            print(f"✅ Created default configuration file: {self.config_path}")
        except Exception as e:
            print(f"Warning: Failed to create configuration file: {e}")

    def get_database_path(self) -> Path:
        """Get database file path"""
        # If data_directory is set, use it as base path
        if hasattr(self, '_data_directory') and self._data_directory:
            return Path(self._data_directory) / 'hibro.db'

        db_path = Path(self.storage.database_path).expanduser()
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    @property
    def data_directory(self) -> str:
        """Get data directory"""
        if hasattr(self, '_data_directory') and self._data_directory:
            return self._data_directory
        return str(Path.home() / '.hibro')

    @data_directory.setter
    def data_directory(self, value: str):
        """Set data directory"""
        self._data_directory = value
        # Also update database path in storage configuration
        self.storage.database_path = str(Path(value) / 'hibro.db')

    def get_conversations_dir(self) -> Path:
        """Get conversation history directory"""
        conv_dir = self.config_dir / 'conversations'
        conv_dir.mkdir(exist_ok=True)
        return conv_dir

    def get_contexts_dir(self) -> Path:
        """Get context directory"""
        ctx_dir = self.config_dir / 'contexts'
        ctx_dir.mkdir(exist_ok=True)
        return ctx_dir

    def save(self):
        """Save current configuration to file"""
        config_data = {
            'memory': {
                'auto_learn': self.memory.auto_learn,
                'importance_threshold': self.memory.importance_threshold,
                'max_memories': self.memory.max_memories
            },
            'forgetting': {
                'time_decay_rate': self.forgetting.time_decay_rate,
                'min_importance': self.forgetting.min_importance,
                'cleanup_interval_days': self.forgetting.cleanup_interval_days,
                'cleanup_time': self.forgetting.cleanup_time,
                'cleanup_enabled': self.forgetting.cleanup_enabled,
                'threshold_warning': self.forgetting.threshold_warning,
                'threshold_cleanup': self.forgetting.threshold_cleanup,
                'threshold_critical': self.forgetting.threshold_critical
            },
            'ide_integration': {
                'auto_inject': self.ide_integration.auto_inject,
                'context_limit_kb': self.ide_integration.context_limit_kb,
                'monitor_conversations': self.ide_integration.monitor_conversations
            },
            'storage': {
                'database_path': self.storage.database_path,
                'max_size_gb': self.storage.max_size_gb,
                'backup_enabled': self.storage.backup_enabled,
                'backup_interval_hours': self.storage.backup_interval_hours
            },
            'security': {
                'encryption_enabled': self.security.encryption_enabled,
                'auto_cleanup_days': self.security.auto_cleanup_days,
                'sensitive_data_filter': self.security.sensitive_data_filter
            }
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False,
                         allow_unicode=True, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save configuration: {e}")