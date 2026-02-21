#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Bus Module
Implements publish/subscribe pattern for decoupled inter-component communication and real-time notifications
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from enum import Enum


class EventType(Enum):
    """Event type enumeration"""
    # Memory-related events
    MEMORY_STORED = "memory_stored"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"

    # Project-related events
    PROJECT_SCANNED = "project_scanned"
    PROJECT_STATUS_UPDATED = "project_status_updated"

    # Preference-related events
    PREFERENCE_CHANGED = "preference_changed"

    # Database-related events
    DATABASE_CHANGED = "database_changed"
    DATABASE_SYNC = "database_sync"

    # System events
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_RESTORE = "system_restore"


@dataclass
class Event:
    """Event object"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"  # Event source identifier
    priority: int = 0  # Priority (0=normal, 1=high, 2=urgent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority
        }


class EventSubscriber:
    """Event subscriber"""

    def __init__(self,
                 callback: Callable[[Event], None],
                 subscriber_id: str,
                 event_types: Optional[List[EventType]] = None,
                 priority: int = 0):
        """
        Initialize event subscriber

        Args:
            callback: Event callback function
            subscriber_id: Subscriber ID
            event_types: List of event types to subscribe (None means subscribe to all)
            priority: Priority
        """
        self.callback = callback
        self.subscriber_id = subscriber_id
        self.event_types = event_types
        self.priority = priority
        self.call_count = 0
        self.last_called = None

    def should_handle(self, event: Event) -> bool:
        """Determine if this event should be handled"""
        if self.event_types is None:
            return True
        return event.event_type in self.event_types

    def handle(self, event: Event):
        """Handle event"""
        try:
            self.callback(event)
            self.call_count += 1
            self.last_called = datetime.now()
        except Exception as e:
            # Log error but don't interrupt other subscribers
            logging.getLogger('hibro.event_bus').error(
                f"Subscriber {self.subscriber_id} failed to handle event: {e}"
            )


class EventBus:
    """Event bus - publish/subscribe pattern"""

    def __init__(self, max_queue_size: int = 1000, worker_threads: int = 2):
        """
        Initialize event bus

        Args:
            max_queue_size: Maximum event queue size
            worker_threads: Number of worker threads
        """
        self.logger = logging.getLogger('hibro.event_bus')
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads

        # Subscriber list
        self.subscribers: List[EventSubscriber] = []
        self._subscriber_lock = threading.Lock()

        # Event queue (for async processing)
        self.event_queue: Queue = Queue(maxsize=max_queue_size)

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.is_running = False

        # Statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'start_time': None
        }

    def start(self):
        """Start event bus"""
        if self.is_running:
            self.logger.warning("Event bus is already running")
            return

        self.is_running = True
        self.stats['start_time'] = datetime.now()

        # Start worker threads
        for i in range(self.worker_threads):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"EventBus-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        self.logger.info(
            f"Event bus started (worker threads: {self.worker_threads}, "
            f"queue size: {self.max_queue_size})"
        )

    def stop(self):
        """Stop event bus"""
        if not self.is_running:
            return

        self.is_running = False

        # Send stop signal
        for _ in self.workers:
            self.event_queue.put(None)

        # Wait for worker threads to finish
        for worker in self.workers:
            worker.join(timeout=5)

        self.workers.clear()
        self.logger.info("Event bus stopped")

    def subscribe(self,
                  callback: Callable[[Event], None],
                  subscriber_id: str,
                  event_types: Optional[List[EventType]] = None,
                  priority: int = 0) -> str:
        """
        Subscribe to events

        Args:
            callback: Event callback function
            subscriber_id: Subscriber ID
            event_types: List of event types to subscribe (None means subscribe to all)
            priority: Priority

        Returns:
            Subscriber ID
        """
        with self._subscriber_lock:
            subscriber = EventSubscriber(
                callback=callback,
                subscriber_id=subscriber_id,
                event_types=event_types,
                priority=priority
            )
            self.subscribers.append(subscriber)
            # Sort by priority (higher priority first)
            self.subscribers.sort(key=lambda s: s.priority, reverse=True)

        self.logger.info(
            f"New subscriber: {subscriber_id}, "
            f"event types: {[e.value for e in event_types] if event_types else 'all'}"
        )
        return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from events

        Args:
            subscriber_id: Subscriber ID

        Returns:
            Whether unsubscription was successful
        """
        with self._subscriber_lock:
            for i, subscriber in enumerate(self.subscribers):
                if subscriber.subscriber_id == subscriber_id:
                    self.subscribers.pop(i)
                    self.logger.info(f"Subscriber unsubscribed: {subscriber_id}")
                    return True

        self.logger.warning(f"Subscriber not found: {subscriber_id}")
        return False

    def publish(self,
                event_type: EventType,
                data: Dict[str, Any],
                source: str = "unknown",
                priority: int = 0,
                async_mode: bool = True) -> bool:
        """
        Publish event

        Args:
            event_type: Event type
            data: Event data
            source: Event source
            priority: Priority
            async_mode: Whether to process asynchronously

        Returns:
            Whether publishing was successful
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            priority=priority
        )

        self.stats['events_published'] += 1

        if async_mode:
            # Async processing (add to queue)
            try:
                self.event_queue.put(event, block=False)
                self.logger.debug(f"Event published (async): {event_type.value}")
                return True
            except:
                # Queue full, drop event
                self.stats['events_dropped'] += 1
                self.logger.warning(f"Event queue full, dropping event: {event_type.value}")
                return False
        else:
            # Sync processing (execute immediately)
            self._dispatch_event(event)
            return True

    def _worker_loop(self):
        """Worker thread loop"""
        while self.is_running:
            try:
                # Get event from queue
                event = self.event_queue.get(timeout=1)

                # Stop signal
                if event is None:
                    break

                # Dispatch event
                self._dispatch_event(event)
                self.stats['events_processed'] += 1

            except:
                # Queue empty, continue waiting
                continue

    def _dispatch_event(self, event: Event):
        """Dispatch event to subscribers"""
        self.logger.debug(f"Dispatching event: {event.event_type.value}")

        with self._subscriber_lock:
            subscribers = self.subscribers.copy()

        # Call subscribers by priority
        for subscriber in subscribers:
            if subscriber.should_handle(event):
                try:
                    subscriber.handle(event)
                except Exception as e:
                    self.logger.error(
                        f"Subscriber {subscriber.subscriber_id} failed to handle event: {e}"
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()

        return {
            'is_running': self.is_running,
            'worker_threads': self.worker_threads,
            'queue_size': self.event_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'subscriber_count': len(self.subscribers),
            'events_published': self.stats['events_published'],
            'events_processed': self.stats['events_processed'],
            'events_dropped': self.stats['events_dropped'],
            'uptime_seconds': uptime
        }

    def get_subscribers_info(self) -> List[Dict[str, Any]]:
        """Get subscriber information"""
        with self._subscriber_lock:
            return [
                {
                    'subscriber_id': s.subscriber_id,
                    'event_types': [e.value for e in s.event_types] if s.event_types else 'all',
                    'priority': s.priority,
                    'call_count': s.call_count,
                    'last_called': s.last_called.isoformat() if s.last_called else None
                }
                for s in self.subscribers
            ]

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False
