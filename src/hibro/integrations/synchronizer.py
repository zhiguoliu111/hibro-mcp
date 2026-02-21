#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Synchronizer Module
Implements real-time synchronization functionality between memory system and Claude Code
"""

import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty

from ..core.memory_engine import MemoryEngine
from ..intelligence import MemoryExtractor, ImportanceScorer
from ..interfaces.automation import AutomationEngine
from .listener import ConversationListener
from .injector import MemoryInjector
from ..utils.config import Config


class SyncEvent:
    """Sync Event"""

    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Initialize sync event

        Args:
            event_type: Event type
            data: Event data
            timestamp: Timestamp
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.processed = False


class RealTimeSynchronizer:
    """Real-time Synchronizer"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize real-time synchronizer

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.realtime_synchronizer')

        # Initialize components
        self.conversation_listener = ConversationListener(config, memory_engine)
        self.memory_injector = MemoryInjector(config, memory_engine)
        self.automation_engine = AutomationEngine(config, memory_engine)
        self.memory_extractor = MemoryExtractor()
        self.importance_scorer = ImportanceScorer(config)

        # Sync configuration
        self.sync_config = {
            'enabled': True,
            'batch_size': 10,
            'sync_interval_seconds': 5,
            'max_queue_size': 1000,
            'auto_backup_interval_hours': 24,
            'context_refresh_interval_minutes': 30
        }

        # Event queue and processing
        self.event_queue = Queue(maxsize=self.sync_config['max_queue_size'])
        self.processing_thread = None
        self.is_running = False

        # Statistics
        self.sync_stats = {
            'events_processed': 0,
            'memories_synced': 0,
            'last_sync_time': None,
            'errors_count': 0,
            'start_time': None
        }

        # Callback functions
        self.on_memory_synced: Optional[Callable] = None
        self.on_context_updated: Optional[Callable] = None
        self.on_sync_error: Optional[Callable] = None

        # Setup listener callbacks
        self._setup_listener_callbacks()

    def _setup_listener_callbacks(self):
        """Setup listener callback functions"""
        self.conversation_listener.set_callbacks(
            on_conversation_detected=self._handle_conversation_detected,
            on_memory_extracted=self._handle_memory_extracted
        )

    def start_sync(self):
        """Start real-time sync"""
        if self.is_running:
            self.logger.warning("Real-time synchronizer is already running")
            return

        try:
            # Start conversation listener
            self.conversation_listener.start_listening()

            # Start event processing thread
            self.processing_thread = threading.Thread(
                target=self._process_events_loop,
                daemon=True
            )
            self.processing_thread.start()

            # Start periodic tasks thread
            self.periodic_thread = threading.Thread(
                target=self._periodic_tasks_loop,
                daemon=True
            )
            self.periodic_thread.start()

            self.is_running = True
            self.sync_stats['start_time'] = datetime.now()

            self.logger.info("Real-time synchronizer started")

        except Exception as e:
            self.logger.error(f"Failed to start real-time synchronizer: {e}")
            raise

    def stop_sync(self):
        """Stop real-time sync"""
        if not self.is_running:
            return

        try:
            # Stop listener
            self.conversation_listener.stop_listening()

            # Stop sync
            self.is_running = False

            # Wait for processing threads to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            if self.periodic_thread and self.periodic_thread.is_alive():
                self.periodic_thread.join(timeout=5)

            self.logger.info("Real-time synchronizer stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop real-time synchronizer: {e}")

    def _handle_conversation_detected(self, content: str, conversation_data: Dict[str, Any], source_file: Path):
        """Handle detected conversation"""
        try:
            event = SyncEvent(
                event_type='conversation_detected',
                data={
                    'content': content,
                    'conversation_data': conversation_data,
                    'source_file': str(source_file)
                }
            )

            self._queue_event(event)

        except Exception as e:
            self.logger.error(f"Failed to handle conversation detection event: {e}")

    def _handle_memory_extracted(self, extraction_result: Dict[str, Any], context: Dict[str, Any]):
        """Handle memory extraction event"""
        try:
            event = SyncEvent(
                event_type='memory_extracted',
                data={
                    'extraction_result': extraction_result,
                    'context': context
                }
            )

            self._queue_event(event)

        except Exception as e:
            self.logger.error(f"Failed to handle memory extraction event: {e}")

    def _queue_event(self, event: SyncEvent):
        """Add event to queue"""
        try:
            self.event_queue.put(event, timeout=1)
        except Exception as e:
            self.logger.warning(f"Event queue full, dropping event: {event.event_type}")
            self.sync_stats['errors_count'] += 1

    def _process_events_loop(self):
        """Event processing loop"""
        self.logger.info("Event processing thread started")

        while self.is_running:
            try:
                # Batch process events
                events = self._get_batch_events()

                if events:
                    self._process_event_batch(events)

                # Short sleep
                time.sleep(self.sync_config['sync_interval_seconds'])

            except Exception as e:
                self.logger.error(f"Event processing loop error: {e}")
                self.sync_stats['errors_count'] += 1
                time.sleep(1)

    def _get_batch_events(self) -> List[SyncEvent]:
        """Get batch of events"""
        events = []
        batch_size = self.sync_config['batch_size']

        try:
            # Get first event (blocking)
            first_event = self.event_queue.get(timeout=1)
            events.append(first_event)

            # Get remaining events (non-blocking)
            for _ in range(batch_size - 1):
                try:
                    event = self.event_queue.get_nowait()
                    events.append(event)
                except Empty:
                    break

        except Empty:
            pass  # No events, return empty list

        return events

    def _process_event_batch(self, events: List[SyncEvent]):
        """Process event batch"""
        for event in events:
            try:
                self._process_single_event(event)
                event.processed = True
                self.sync_stats['events_processed'] += 1

            except Exception as e:
                self.logger.error(f"Failed to process event {event.event_type}: {e}")
                self.sync_stats['errors_count'] += 1

                if self.on_sync_error:
                    self.on_sync_error(event, e)

        self.sync_stats['last_sync_time'] = datetime.now()

    def _process_single_event(self, event: SyncEvent):
        """Process single event"""
        if event.event_type == 'conversation_detected':
            self._process_conversation_event(event)
        elif event.event_type == 'memory_extracted':
            self._process_memory_extraction_event(event)
        elif event.event_type == 'context_update':
            self._process_context_update_event(event)
        elif event.event_type == 'importance_update':
            self._process_importance_update_event(event)
        else:
            self.logger.warning(f"Unknown event type: {event.event_type}")

    def _process_conversation_event(self, event: SyncEvent):
        """Process conversation event"""
        data = event.data
        content = data['content']
        source_file = data['source_file']

        # Intelligently extract memories
        extracted_memories = self.memory_extractor.extract_memories(content)

        if extracted_memories:
            # Detect project context
            project_context = self._detect_project_from_path(source_file)

            # Store extracted memories
            for extracted in extracted_memories:
                memory_id = self.memory_engine.store_memory(
                    content=extracted.content,
                    importance=extracted.importance,
                    category=extracted.category,
                    memory_type=extracted.memory_type,
                    project_path=project_context.get('project_path')
                )

                self.sync_stats['memories_synced'] += 1

                # Trigger callback
                if self.on_memory_synced:
                    self.on_memory_synced(memory_id, extracted, project_context)

        self.logger.debug(f"Processed conversation event: Extracted {len(extracted_memories)} memories")

    def _process_memory_extraction_event(self, event: SyncEvent):
        """Process memory extraction event"""
        data = event.data
        extraction_result = data['extraction_result']
        context = data['context']

        # Apply automation rules results
        if 'extracted_memories' in extraction_result:
            for memory_data in extraction_result['extracted_memories']:
                memory_id = self.memory_engine.store_memory(
                    content=memory_data['content'],
                    importance=memory_data['importance'],
                    category=memory_data.get('category'),
                    memory_type=memory_data['type'],
                    project_path=context.get('project_path')
                )

                self.sync_stats['memories_synced'] += 1

    def _process_context_update_event(self, event: SyncEvent):
        """Process context update event"""
        data = event.data
        project_path = data.get('project_path')

        if project_path:
            # Create project context snapshot
            snapshot_text = self.memory_injector.create_project_context_snapshot(project_path)

            # Save snapshot
            self.memory_engine.fs_manager.save_context_snapshot(project_path, [])

            if self.on_context_updated:
                self.on_context_updated(project_path, snapshot_text)

    def _process_importance_update_event(self, event: SyncEvent):
        """Process importance update event"""
        data = event.data
        memory_id = data.get('memory_id')
        feedback_type = data.get('feedback_type')

        if memory_id and feedback_type:
            memory = self.memory_engine.get_memory(memory_id)
            if memory:
                # Update importance
                new_importance = self.importance_scorer.update_importance_by_feedback(
                    memory, feedback_type
                )

                # Save update
                self.memory_engine.memory_repo.update_memory(memory)

    def _periodic_tasks_loop(self):
        """Periodic tasks loop"""
        self.logger.info("Periodic tasks thread started")

        last_backup = datetime.now()
        last_context_refresh = datetime.now()

        while self.is_running:
            try:
                current_time = datetime.now()

                # Auto backup
                if (current_time - last_backup).total_seconds() >= self.sync_config['auto_backup_interval_hours'] * 3600:
                    self._perform_auto_backup()
                    last_backup = current_time

                # Context refresh
                if (current_time - last_context_refresh).total_seconds() >= self.sync_config['context_refresh_interval_minutes'] * 60:
                    self._refresh_project_contexts()
                    last_context_refresh = current_time

                # Clean old events
                self._cleanup_old_events()

                # Sleep
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Periodic task error: {e}")
                time.sleep(60)

    def _perform_auto_backup(self):
        """Perform auto backup"""
        try:
            backup_path = self.memory_engine.db_manager.backup_database()
            self.logger.info(f"Auto backup completed: {backup_path}")

        except Exception as e:
            self.logger.error(f"Auto backup failed: {e}")

    def _refresh_project_contexts(self):
        """Refresh project contexts"""
        try:
            # Get active projects
            active_projects = self._get_active_projects()

            for project_path in active_projects:
                # Create context update event
                event = SyncEvent(
                    event_type='context_update',
                    data={'project_path': project_path}
                )
                self._queue_event(event)

            self.logger.debug(f"Refreshed {len(active_projects)} project contexts")

        except Exception as e:
            self.logger.error(f"Failed to refresh project contexts: {e}")

    def _get_active_projects(self) -> List[str]:
        """Get active project list"""
        try:
            # Get recently accessed projects
            recent_cutoff = datetime.now() - timedelta(days=7)

            memories = self.memory_engine.memory_repo.search_memories(limit=1000)
            project_paths = set()

            for memory in memories:
                if (memory.last_accessed and memory.last_accessed >= recent_cutoff and
                    memory.metadata and 'project_path' in memory.metadata):
                    project_paths.add(memory.metadata['project_path'])

            return list(project_paths)

        except Exception as e:
            self.logger.error(f"Failed to get active projects: {e}")
            return []

    def _cleanup_old_events(self):
        """Clean old events"""
        # Can implement event history cleanup logic here
        pass

    def _detect_project_from_path(self, file_path: str) -> Dict[str, Any]:
        """Detect project information from file path"""
        try:
            path_obj = Path(file_path)
            path_parts = path_obj.parts

            # Find project root directory
            for i, part in enumerate(path_parts):
                if part in ['projects', 'workspace', 'code', 'dev', 'src']:
                    if i + 1 < len(path_parts):
                        project_name = path_parts[i + 1]
                        project_path = str(Path(*path_parts[:i + 2]))

                        return {
                            'project_name': project_name,
                            'project_path': project_path,
                            'detected_from': 'file_path'
                        }

            return {}

        except Exception as e:
            self.logger.warning(f"Failed to detect project from path: {e}")
            return {}

    def trigger_manual_sync(self, sync_type: str = 'full'):
        """Trigger manual sync"""
        try:
            if sync_type == 'full':
                # Full sync
                self._perform_full_sync()
            elif sync_type == 'incremental':
                # Incremental sync
                self._perform_incremental_sync()
            elif sync_type == 'context_only':
                # Context only sync
                self._refresh_project_contexts()

            self.logger.info(f"Manual sync completed: {sync_type}")

        except Exception as e:
            self.logger.error(f"Manual sync failed: {e}")
            raise

    def _perform_full_sync(self):
        """Perform full sync"""
        # Re-process all conversation files
        self.conversation_listener._process_existing_files()

        # Refresh all project contexts
        self._refresh_project_contexts()

    def _perform_incremental_sync(self):
        """Perform incremental sync"""
        # Process recently modified files
        cutoff_time = datetime.now() - timedelta(hours=24)

        for directory in self.conversation_listener.watch_directories:
            for file_path in directory.rglob('*.json'):
                try:
                    if file_path.stat().st_mtime > cutoff_time.timestamp():
                        self.conversation_listener.process_conversation_file(file_path)
                except Exception:
                    continue

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return {
            'is_running': self.is_running,
            'stats': self.sync_stats.copy(),
            'config': self.sync_config.copy(),
            'queue_size': self.event_queue.qsize(),
            'listener_status': self.conversation_listener.get_listening_status()
        }

    def update_sync_config(self, **kwargs) -> bool:
        """Update sync configuration"""
        try:
            for key, value in kwargs.items():
                if key in self.sync_config:
                    self.sync_config[key] = value
                    self.logger.info(f"Sync config updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update sync config: {e}")
            return False

    def set_callbacks(self, on_memory_synced: Optional[Callable] = None,
                     on_context_updated: Optional[Callable] = None,
                     on_sync_error: Optional[Callable] = None):
        """Set callback functions"""
        self.on_memory_synced = on_memory_synced
        self.on_context_updated = on_context_updated
        self.on_sync_error = on_sync_error