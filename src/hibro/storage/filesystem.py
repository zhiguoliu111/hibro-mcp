#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File system storage management module
Handles storage management for conversation history, context snapshots and export files
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict

from ..utils.config import Config
from ..utils.helpers import ensure_directory, get_file_size_mb, format_timestamp


@dataclass
class ConversationFile:
    """Conversation file information"""
    file_path: Path
    date: date
    project_name: Optional[str] = None
    size_mb: float = 0.0
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.file_path.exists():
            stat = self.file_path.stat()
            self.size_mb = stat.st_size / (1024 * 1024)
            self.created_at = datetime.fromtimestamp(stat.st_ctime)
            self.modified_at = datetime.fromtimestamp(stat.st_mtime)


@dataclass
class ContextSnapshot:
    """Context snapshot information"""
    project_path: str
    snapshot_time: datetime
    memories_count: int
    file_path: Path
    size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'project_path': self.project_path,
            'snapshot_time': self.snapshot_time.isoformat(),
            'memories_count': self.memories_count,
            'file_path': str(self.file_path),
            'size_mb': self.size_mb
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextSnapshot':
        """Create object from dictionary"""
        return cls(
            project_path=data['project_path'],
            snapshot_time=datetime.fromisoformat(data['snapshot_time']),
            memories_count=data['memories_count'],
            file_path=Path(data['file_path']),
            size_mb=data.get('size_mb', 0.0)
        )


class FileSystemManager:
    """File system storage manager"""

    def __init__(self, config: Config):
        """
        Initialize file system manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.filesystem')

        # Initialize directory structure
        self.base_dir = Path.home() / '.hibro'
        self.conversations_dir = self.base_dir / 'conversations'
        self.contexts_dir = self.base_dir / 'contexts'
        self.exports_dir = self.base_dir / 'exports'
        self.backups_dir = self.base_dir / 'backups'
        self.logs_dir = self.base_dir / 'logs'

        # Ensure all directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.base_dir,
            self.conversations_dir,
            self.contexts_dir,
            self.exports_dir,
            self.backups_dir,
            self.logs_dir
        ]

        for directory in directories:
            if ensure_directory(directory):
                self.logger.debug(f"Directory ensured to exist: {directory}")
            else:
                self.logger.error(f"Failed to create directory: {directory}")

    def save_conversation(self, content: str, project_name: Optional[str] = None,
                         conversation_date: Optional[date] = None) -> Path:
        """
        Save conversation content to file

        Args:
            content: Conversation content
            project_name: Project name
            conversation_date: Conversation date, uses current date if None

        Returns:
            Saved file path
        """
        if conversation_date is None:
            conversation_date = date.today()

        try:
            # Create date directory
            date_dir = self.conversations_dir / conversation_date.strftime('%Y-%m-%d')
            ensure_directory(date_dir)

            # Generate filename
            timestamp = datetime.now().strftime('%H%M%S')
            if project_name:
                filename = f"{project_name}_{timestamp}.md"
            else:
                filename = f"conversation_{timestamp}.md"

            file_path = date_dir / filename

            # Save content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Conversation Record\n\n")
                f.write(f"**Date**: {conversation_date}\n")
                f.write(f"**Time**: {format_timestamp()}\n")
                if project_name:
                    f.write(f"**Project**: {project_name}\n")
                f.write(f"\n---\n\n")
                f.write(content)

            self.logger.info(f"Conversation saved: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            raise

    def load_conversation(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Load conversation content

        Args:
            file_path: File path

        Returns:
            Conversation content, returns None if failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.warning(f"Conversation file does not exist: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.logger.info(f"Conversation loaded: {file_path}")
            return content

        except Exception as e:
            self.logger.error(f"Failed to load conversation: {e}")
            return None

    def list_conversations(self, project_name: Optional[str] = None,
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> List[ConversationFile]:
        """
        List conversation files

        Args:
            project_name: Project name filter
            start_date: Start date
            end_date: End date

        Returns:
            Conversation file list
        """
        conversations = []

        try:
            for date_dir in self.conversations_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                try:
                    conv_date = datetime.strptime(date_dir.name, '%Y-%m-%d').date()
                except ValueError:
                    continue

                # Date filtering
                if start_date and conv_date < start_date:
                    continue
                if end_date and conv_date > end_date:
                    continue

                # Iterate through files in this date directory
                for file_path in date_dir.glob('*.md'):
                    # Project name filtering
                    if project_name:
                        if not file_path.name.startswith(f"{project_name}_"):
                            continue

                    # Extract project name
                    file_project_name = None
                    if '_' in file_path.stem:
                        file_project_name = file_path.stem.split('_')[0]
                        if file_project_name == 'conversation':
                            file_project_name = None

                    conv_file = ConversationFile(
                        file_path=file_path,
                        date=conv_date,
                        project_name=file_project_name
                    )

                    conversations.append(conv_file)

            # Sort by date and time
            conversations.sort(key=lambda x: (x.date, x.created_at or datetime.min), reverse=True)

            self.logger.info(f"Found {len(conversations)} conversation files")
            return conversations

        except Exception as e:
            self.logger.error(f"Failed to list conversation files: {e}")
            return []

    def save_context_snapshot(self, project_path: str, memories: List[Dict[str, Any]]) -> Path:
        """
        Save project context snapshot

        Args:
            project_path: Project path
            memories: Memory list

        Returns:
            Snapshot file path
        """
        try:
            # Create project directory
            project_name = Path(project_path).name
            project_dir = self.contexts_dir / project_name
            ensure_directory(project_dir)

            # Generate snapshot filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_file = project_dir / f"context_{timestamp}.json"

            # Prepare snapshot data
            snapshot_data = {
                'project_path': project_path,
                'project_name': project_name,
                'snapshot_time': datetime.now().isoformat(),
                'memories_count': len(memories),
                'memories': memories,
                'metadata': {
                    'created_by': 'hibro',
                    'version': '1.0.0'
                }
            }

            # Save snapshot
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Context snapshot saved: {snapshot_file}")
            return snapshot_file

        except Exception as e:
            self.logger.error(f"Failed to save context snapshot: {e}")
            raise

    def load_context_snapshot(self, snapshot_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load context snapshot

        Args:
            snapshot_path: Snapshot file path

        Returns:
            Snapshot data, returns None if failed
        """
        try:
            snapshot_path = Path(snapshot_path)
            if not snapshot_path.exists():
                self.logger.warning(f"Snapshot file does not exist: {snapshot_path}")
                return None

            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)

            self.logger.info(f"Context snapshot loaded: {snapshot_path}")
            return snapshot_data

        except Exception as e:
            self.logger.error(f"Failed to load context snapshot: {e}")
            return None

    def list_context_snapshots(self, project_name: Optional[str] = None) -> List[ContextSnapshot]:
        """
        List context snapshots

        Args:
            project_name: Project name filter

        Returns:
            Snapshot list
        """
        snapshots = []

        try:
            search_dirs = []
            if project_name:
                project_dir = self.contexts_dir / project_name
                if project_dir.exists():
                    search_dirs.append(project_dir)
            else:
                search_dirs = [d for d in self.contexts_dir.iterdir() if d.is_dir()]

            for project_dir in search_dirs:
                for snapshot_file in project_dir.glob('context_*.json'):
                    try:
                        # Extract time from filename
                        timestamp_str = snapshot_file.stem.replace('context_', '')
                        snapshot_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

                        # Read snapshot basic information
                        with open(snapshot_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        snapshot = ContextSnapshot(
                            project_path=data.get('project_path', ''),
                            snapshot_time=snapshot_time,
                            memories_count=data.get('memories_count', 0),
                            file_path=snapshot_file,
                            size_mb=get_file_size_mb(snapshot_file)
                        )

                        snapshots.append(snapshot)

                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Skip invalid snapshot file {snapshot_file}: {e}")
                        continue

            # Sort by time
            snapshots.sort(key=lambda x: x.snapshot_time, reverse=True)

            self.logger.info(f"Found {len(snapshots)} context snapshots")
            return snapshots

        except Exception as e:
            self.logger.error(f"Failed to list context snapshots: {e}")
            return []

    def export_memories(self, memories: List[Dict[str, Any]],
                       export_format: str = 'json',
                       filename: Optional[str] = None) -> Path:
        """
        Export memory data

        Args:
            memories: Memory list
            export_format: Export format ('json', 'markdown', 'csv')
            filename: Filename, auto-generated if None

        Returns:
            Export file path
        """
        try:
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"memories_export_{timestamp}.{export_format}"

            export_file = self.exports_dir / filename

            # Export by format
            if export_format == 'json':
                self._export_json(memories, export_file)
            elif export_format == 'markdown':
                self._export_markdown(memories, export_file)
            elif export_format == 'csv':
                self._export_csv(memories, export_file)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            self.logger.info(f"Memories exported: {export_file}")
            return export_file

        except Exception as e:
            self.logger.error(f"Failed to export memories: {e}")
            raise

    def _export_json(self, memories: List[Dict[str, Any]], file_path: Path):
        """Export as JSON format"""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'memories_count': len(memories),
            'memories': memories,
            'metadata': {
                'exported_by': 'hibro',
                'version': '1.0.0'
            }
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def _export_markdown(self, memories: List[Dict[str, Any]], file_path: Path):
        """Export as Markdown format"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# hibro Memory Export\n\n")
            f.write(f"**Export Time**: {format_timestamp()}\n")
            f.write(f"**Memory Count**: {len(memories)}\n\n")
            f.write("---\n\n")

            for i, memory in enumerate(memories, 1):
                f.write(f"## Memory {i}\n\n")
                f.write(f"**Type**: {memory.get('memory_type', 'unknown')}\n")
                f.write(f"**Importance**: {memory.get('importance', 0.0):.2f}\n")
                if memory.get('category'):
                    f.write(f"**Category**: {memory['category']}\n")
                f.write(f"**Created**: {memory.get('created_at', 'unknown')}\n\n")
                f.write(f"**Content**:\n{memory.get('content', '')}\n\n")
                f.write("---\n\n")

    def _export_csv(self, memories: List[Dict[str, Any]], file_path: Path):
        """Export as CSV format"""
        import csv

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if not memories:
                return

            # Get all fields
            fieldnames = set()
            for memory in memories:
                fieldnames.update(memory.keys())

            fieldnames = sorted(fieldnames)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(memories)

    def cleanup_old_files(self, days_to_keep: int = 30):
        """
        Clean up old files

        Args:
            days_to_keep: Days to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleaned_count = 0

            # Clean up old conversation files
            for date_dir in self.conversations_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                try:
                    dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d')
                    if dir_date < cutoff_date:
                        shutil.rmtree(date_dir)
                        cleaned_count += 1
                        self.logger.info(f"Cleaned old conversation directory: {date_dir}")
                except ValueError:
                    continue

            # Clean up old log files
            for log_file in self.logs_dir.glob('*.log'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    cleaned_count += 1
                    self.logger.info(f"Cleaned old log file: {log_file}")

            self.logger.info(f"Cleanup completed, cleaned {cleaned_count} files/directories")

        except Exception as e:
            self.logger.error(f"Failed to clean up old files: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Storage statistics
        """
        try:
            stats = {
                'base_dir': str(self.base_dir),
                'total_size_mb': 0.0,
                'conversations': {
                    'count': 0,
                    'size_mb': 0.0
                },
                'contexts': {
                    'count': 0,
                    'size_mb': 0.0
                },
                'exports': {
                    'count': 0,
                    'size_mb': 0.0
                },
                'backups': {
                    'count': 0,
                    'size_mb': 0.0
                }
            }

            # Statistics for each directory
            directories = {
                'conversations': self.conversations_dir,
                'contexts': self.contexts_dir,
                'exports': self.exports_dir,
                'backups': self.backups_dir
            }

            for name, directory in directories.items():
                if directory.exists():
                    count = 0
                    size = 0.0

                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            count += 1
                            size += get_file_size_mb(file_path)

                    stats[name]['count'] = count
                    stats[name]['size_mb'] = size
                    stats['total_size_mb'] += size

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}