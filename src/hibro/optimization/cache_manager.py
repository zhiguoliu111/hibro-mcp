#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache manager
Provides multi-level caching mechanism to optimize data access performance
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from ..utils.config import Config


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1_MEMORY = "l1_memory"      # Memory cache
    L2_DISK = "l2_disk"          # Disk cache
    L3_COMPRESSED = "l3_compressed"  # Compressed cache


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    level: CacheLevel = CacheLevel.L1_MEMORY


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0


class LRUCache:
    """LRU cache implementation"""

    def __init__(self, max_size: int, max_memory_mb: int = 100):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage (MB)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if self._is_expired(entry):
                del self.cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None

            # Update access information
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set cache value"""
        with self.lock:
            # Calculate value size
            size_bytes = self._calculate_size(value)

            # Check if exceeds single entry size limit
            if size_bytes > self.max_memory_bytes // 2:
                return False

            now = datetime.now()

            # If key exists, update it
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size_bytes -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )

            # Ensure enough space
            while (len(self.cache) >= self.max_size or
                   self.stats.total_size_bytes + size_bytes > self.max_memory_bytes):
                if not self._evict_lru():
                    return False

            # Add to cache
            self.cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self.cache)

            return True

    def remove(self, key: str) -> bool:
        """Remove cache entry"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.entry_count = len(self.cache)
                return True
            return False

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()

    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self.cache:
            return False

        # Get least recently used entry
        lru_key, lru_entry = next(iter(self.cache.items()))
        del self.cache[lru_key]

        self.stats.total_size_bytes -= lru_entry.size_bytes
        self.stats.evictions += 1
        self.stats.entry_count = len(self.cache)

        return True

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired"""
        if entry.ttl_seconds is None:
            return False

        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds

    def _calculate_size(self, value: Any) -> int:
        """Calculate value size"""
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            # Simple estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v)
                          for k, v in value.items())
            else:
                return 1024  # Default 1KB

    def cleanup_expired(self):
        """Clean up expired entries"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                self.remove(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats.hits / (self.stats.hits + self.stats.misses)
                       if (self.stats.hits + self.stats.misses) > 0 else 0)

            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': hit_rate,
                'evictions': self.stats.evictions,
                'entry_count': self.stats.entry_count,
                'total_size_mb': self.stats.total_size_bytes / (1024 * 1024),
                'max_size': self.max_size,
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }


class CacheManager:
    """Cache manager"""

    def __init__(self, config: Config):
        """
        Initialize cache manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.cache_manager')

        # Cache configuration
        self.cache_config = {
            'l1_max_entries': 1000,
            'l1_max_memory_mb': 100,
            'l1_default_ttl_seconds': 3600,  # 1 hour
            'l2_max_entries': 5000,
            'l2_max_memory_mb': 500,
            'l2_default_ttl_seconds': 86400,  # 24 hours
            'cleanup_interval_seconds': 300,  # 5 minutes
            'enable_compression': True,
            'compression_threshold_bytes': 1024  # 1KB
        }

        # Multi-level cache
        self.l1_cache = LRUCache(
            self.cache_config['l1_max_entries'],
            self.cache_config['l1_max_memory_mb']
        )

        self.l2_cache = LRUCache(
            self.cache_config['l2_max_entries'],
            self.cache_config['l2_max_memory_mb']
        )

        # Cache key prefixes
        self.cache_prefixes = {
            'memory': 'mem:',
            'similarity': 'sim:',
            'embedding': 'emb:',
            'search': 'search:',
            'project': 'proj:',
            'statistics': 'stats:'
        }

        # Start cleanup thread
        self.cleanup_thread = None
        self.running = False
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start cleanup thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()

    def _cleanup_loop(self):
        """Cleanup loop"""
        while self.running:
            try:
                time.sleep(self.cache_config['cleanup_interval_seconds'])
                self._cleanup_expired_entries()
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")

    def _cleanup_expired_entries(self):
        """Clean up expired entries"""
        self.l1_cache.cleanup_expired()
        self.l2_cache.cleanup_expired()

    def _make_key(self, cache_type: str, key: str) -> str:
        """Generate cache key"""
        prefix = self.cache_prefixes.get(cache_type, 'misc:')
        return f"{prefix}{key}"

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Get cache value

        Args:
            cache_type: Cache type
            key: Cache key

        Returns:
            Cache value, returns None if not exists
        """
        cache_key = self._make_key(cache_type, key)

        # Check L1 cache first
        value = self.l1_cache.get(cache_key)
        if value is not None:
            return self._decompress_if_needed(value)

        # Then check L2 cache
        value = self.l2_cache.get(cache_key)
        if value is not None:
            # Promote to L1 cache
            decompressed_value = self._decompress_if_needed(value)
            self.l1_cache.put(
                cache_key,
                decompressed_value,
                self.cache_config['l1_default_ttl_seconds']
            )
            return decompressed_value

        return None

    def put(self, cache_type: str, key: str, value: Any,
           ttl_seconds: Optional[int] = None) -> bool:
        """
        Set cache value

        Args:
            cache_type: Cache type
            key: Cache key
            value: Cache value
            ttl_seconds: Time to live

        Returns:
            Whether set was successful
        """
        cache_key = self._make_key(cache_type, key)

        if ttl_seconds is None:
            ttl_seconds = self.cache_config['l1_default_ttl_seconds']

        # Compress large values
        compressed_value = self._compress_if_needed(value)

        # Try to store in L1 cache
        if self.l1_cache.put(cache_key, compressed_value, ttl_seconds):
            return True

        # L1 failed, try L2 cache
        l2_ttl = self.cache_config['l2_default_ttl_seconds']
        return self.l2_cache.put(cache_key, compressed_value, l2_ttl)

    def remove(self, cache_type: str, key: str) -> bool:
        """
        Remove cache entry

        Args:
            cache_type: Cache type
            key: Cache key

        Returns:
            Whether removal was successful
        """
        cache_key = self._make_key(cache_type, key)

        removed_l1 = self.l1_cache.remove(cache_key)
        removed_l2 = self.l2_cache.remove(cache_key)

        return removed_l1 or removed_l2

    def clear(self, cache_type: Optional[str] = None):
        """
        Clear cache

        Args:
            cache_type: Cache type, None means clear all
        """
        if cache_type is None:
            self.l1_cache.clear()
            self.l2_cache.clear()
        else:
            # Clear specific type of cache
            prefix = self.cache_prefixes.get(cache_type, 'misc:')
            self._clear_by_prefix(prefix)

    def _clear_by_prefix(self, prefix: str):
        """Clear cache by prefix"""
        # L1 cache
        keys_to_remove = [key for key in self.l1_cache.cache.keys()
                         if key.startswith(prefix)]
        for key in keys_to_remove:
            self.l1_cache.remove(key)

        # L2 cache
        keys_to_remove = [key for key in self.l2_cache.cache.keys()
                         if key.startswith(prefix)]
        for key in keys_to_remove:
            self.l2_cache.remove(key)

    def _compress_if_needed(self, value: Any) -> Any:
        """Compress value if needed"""
        if not self.cache_config['enable_compression']:
            return value

        try:
            import pickle
            import gzip

            # Serialize value
            pickled = pickle.dumps(value)

            # Check if compression is needed
            if len(pickled) < self.cache_config['compression_threshold_bytes']:
                return value

            # Compress
            compressed = gzip.compress(pickled)

            # If compressed is larger, don't compress
            if len(compressed) >= len(pickled):
                return value

            return {'_compressed': True, '_data': compressed}

        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return value

    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if needed"""
        if not isinstance(value, dict) or not value.get('_compressed'):
            return value

        try:
            import pickle
            import gzip

            compressed_data = value['_data']
            pickled = gzip.decompress(compressed_data)
            return pickle.loads(pickled)

        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return value

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()

        total_hits = l1_stats['hits'] + l2_stats['hits']
        total_misses = l1_stats['misses'] + l2_stats['misses']
        overall_hit_rate = (total_hits / (total_hits + total_misses)
                           if (total_hits + total_misses) > 0 else 0)

        return {
            'overall': {
                'hit_rate': overall_hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_entries': l1_stats['entry_count'] + l2_stats['entry_count'],
                'total_size_mb': l1_stats['total_size_mb'] + l2_stats['total_size_mb']
            },
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'config': self.cache_config.copy()
        }

    def warm_up_cache(self, warm_up_func: Callable[[], List[Tuple[str, str, Any]]]):
        """
        Warm up cache

        Args:
            warm_up_func: Function returning (cache_type, key, value) list
        """
        try:
            entries = warm_up_func()
            warmed_count = 0

            for cache_type, key, value in entries:
                if self.put(cache_type, key, value):
                    warmed_count += 1

            self.logger.info(f"Cache warm-up completed: {warmed_count} entries")

        except Exception as e:
            self.logger.error(f"Cache warm-up failed: {e}")

    def update_cache_config(self, **kwargs) -> bool:
        """
        Update cache configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.cache_config:
                    self.cache_config[key] = value
                    self.logger.info(f"Cache configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update cache configuration: {e}")
            return False

    def shutdown(self):
        """Shutdown cache manager"""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)

        self.clear()
        self.logger.info("Cache manager shutdown completed")