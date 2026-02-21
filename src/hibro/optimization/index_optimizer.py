#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index optimizer
Provides database index analysis and optimization functionality
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from ..utils.config import Config


@dataclass
class IndexInfo:
    """Index information"""
    name: str
    table: str
    columns: List[str]
    unique: bool
    partial: bool
    size_kb: float
    usage_count: int = 0
    last_used: Optional[str] = None


@dataclass
class QueryPlan:
    """Query execution plan"""
    query: str
    plan: List[Dict[str, Any]]
    cost: float
    uses_index: bool
    suggested_indexes: List[str]


class IndexOptimizer:
    """Index optimizer"""

    def __init__(self, config: Config, db_path: Path):
        """
        Initialize index optimizer

        Args:
            config: Configuration object
            db_path: Database path
        """
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger('hibro.index_optimizer')

        # Optimization configuration
        self.optimization_config = {
            'analyze_threshold_queries': 100,
            'unused_index_threshold_days': 30,
            'index_size_threshold_mb': 10,
            'query_cost_threshold': 1000,
            'auto_create_indexes': False,
            'auto_drop_unused_indexes': False
        }

        # Query statistics
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.slow_queries: List[Dict[str, Any]] = []

    def analyze_database_indexes(self) -> Dict[str, Any]:
        """
        Analyze database indexes

        Returns:
            Index analysis results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get all index information
                indexes = self._get_all_indexes(cursor)

                # Analyze index usage
                index_usage = self._analyze_index_usage(cursor, indexes)

                # Check for duplicate indexes
                duplicate_indexes = self._find_duplicate_indexes(indexes)

                # Check for unused indexes
                unused_indexes = self._find_unused_indexes(index_usage)

                # Analyze table statistics
                table_stats = self._get_table_statistics(cursor)

                return {
                    'total_indexes': len(indexes),
                    'indexes': [index.__dict__ for index in indexes],
                    'index_usage': index_usage,
                    'duplicate_indexes': duplicate_indexes,
                    'unused_indexes': unused_indexes,
                    'table_statistics': table_stats,
                    'recommendations': self._generate_index_recommendations(
                        indexes, index_usage, table_stats
                    )
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze database indexes: {e}")
            return {}

    def _get_all_indexes(self, cursor: sqlite3.Cursor) -> List[IndexInfo]:
        """Get all index information"""
        indexes = []

        try:
            # Get index list
            cursor.execute("""
                SELECT name, tbl_name, sql
                FROM sqlite_master
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """)

            for row in cursor.fetchall():
                index_name = row['name']
                table_name = row['tbl_name']
                sql = row['sql'] or ''

                # Get detailed index information
                try:
                    cursor.execute(f"PRAGMA index_info('{index_name}')")
                    columns = [col[2] for col in cursor.fetchall()]

                    # Check if unique index
                    unique = 'UNIQUE' in sql.upper()

                    # Check if partial index
                    partial = 'WHERE' in sql.upper()

                    # Estimate index size
                    size_kb = self._estimate_index_size(cursor, table_name, columns)

                    indexes.append(IndexInfo(
                        name=index_name,
                        table=table_name,
                        columns=columns,
                        unique=unique,
                        partial=partial,
                        size_kb=size_kb
                    ))

                except Exception as e:
                    self.logger.warning(f"Failed to get index {index_name} information: {e}")

        except Exception as e:
            self.logger.error(f"Failed to get index list: {e}")

        return indexes

    def _estimate_index_size(self, cursor: sqlite3.Cursor, table: str, columns: List[str]) -> float:
        """Estimate index size"""
        try:
            # Get table row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]

            # Estimate index size per row (simplified calculation)
            estimated_row_size = len(columns) * 8  # Assume 8 bytes per column

            # Total size (KB)
            total_size_bytes = row_count * estimated_row_size
            return total_size_bytes / 1024

        except Exception:
            return 0.0

    def _analyze_index_usage(self, cursor: sqlite3.Cursor, indexes: List[IndexInfo]) -> Dict[str, Dict[str, Any]]:
        """Analyze index usage"""
        usage_stats = {}

        try:
            # SQLite has no direct index usage statistics, needs analysis through query plans
            for index in indexes:
                usage_stats[index.name] = {
                    'table': index.table,
                    'columns': index.columns,
                    'estimated_usage': 0,
                    'query_types': []
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze index usage: {e}")

        return usage_stats

    def _find_duplicate_indexes(self, indexes: List[IndexInfo]) -> List[Dict[str, Any]]:
        """Find duplicate indexes"""
        duplicates = []
        index_signatures = {}

        for index in indexes:
            # Create index signature (table name + column name combination)
            signature = f"{index.table}:{':'.join(sorted(index.columns))}"

            if signature in index_signatures:
                duplicates.append({
                    'signature': signature,
                    'indexes': [index_signatures[signature], index.name],
                    'table': index.table,
                    'columns': index.columns
                })
            else:
                index_signatures[signature] = index.name

        return duplicates

    def _find_unused_indexes(self, usage_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """Find unused indexes"""
        unused = []

        for index_name, stats in usage_stats.items():
            if stats['estimated_usage'] == 0:
                unused.append(index_name)

        return unused

    def _get_table_statistics(self, cursor: sqlite3.Cursor) -> Dict[str, Dict[str, Any]]:
        """Get table statistics"""
        stats = {}

        try:
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """)

            for row in cursor.fetchall():
                table_name = row[0]

                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]

                    # Get table structure
                    cursor.execute(f"PRAGMA table_info('{table_name}')")
                    columns = cursor.fetchall()

                    stats[table_name] = {
                        'row_count': row_count,
                        'column_count': len(columns),
                        'columns': [col[1] for col in columns]
                    }

                except Exception as e:
                    self.logger.warning(f"Failed to get statistics for table {table_name}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to get table statistics: {e}")

        return stats

    def _generate_index_recommendations(self, indexes: List[IndexInfo],
                                      usage_stats: Dict[str, Dict[str, Any]],
                                      table_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate index optimization recommendations"""
        recommendations = []

        try:
            # Suggest dropping unused indexes
            for index in indexes:
                if index.name in usage_stats and usage_stats[index.name]['estimated_usage'] == 0:
                    if index.size_kb > self.optimization_config['index_size_threshold_mb'] * 1024:
                        recommendations.append({
                            'type': 'drop_unused',
                            'index': index.name,
                            'table': index.table,
                            'reason': f'Index unused and occupies {index.size_kb:.1f}KB space',
                            'priority': 'high'
                        })

            # Suggest creating indexes for large tables
            for table_name, stats in table_stats.items():
                if stats['row_count'] > 10000:  # Large table threshold
                    # Check if there are enough indexes
                    table_indexes = [idx for idx in indexes if idx.table == table_name]
                    if len(table_indexes) < 2:  # Recommend at least 2 indexes
                        recommendations.append({
                            'type': 'create_index',
                            'table': table_name,
                            'reason': f'Large table ({stats["row_count"]} rows) lacks indexes',
                            'priority': 'medium',
                            'suggested_columns': self._suggest_index_columns(table_name, stats)
                        })

        except Exception as e:
            self.logger.error(f"Failed to generate index recommendations: {e}")

        return recommendations

    def _suggest_index_columns(self, table_name: str, stats: Dict[str, Any]) -> List[str]:
        """Suggest index columns"""
        suggestions = []

        # Heuristic suggestions based on table name and column names
        columns = stats.get('columns', [])

        # Common index candidate columns
        index_candidates = [
            'id', 'created_at', 'updated_at', 'last_accessed',
            'importance', 'memory_type', 'project_id', 'user_id'
        ]

        for candidate in index_candidates:
            if candidate in columns:
                suggestions.append(candidate)

        return suggestions[:3]  # Maximum 3 column suggestions

    def analyze_query_performance(self, query: str) -> QueryPlan:
        """
        Analyze query performance

        Args:
            query: SQL query

        Returns:
            Query plan analysis
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get query plan
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                plan_rows = cursor.fetchall()

                plan = []
                uses_index = False
                cost = 0.0

                for row in plan_rows:
                    plan_item = {
                        'id': row[0],
                        'parent': row[1],
                        'detail': row[3]
                    }
                    plan.append(plan_item)

                    # Check if index is used
                    if 'USING INDEX' in row[3].upper():
                        uses_index = True

                    # Estimate cost (simplified)
                    if 'SCAN' in row[3].upper():
                        cost += 100  # Full table scan has high cost
                    elif 'SEARCH' in row[3].upper():
                        cost += 10   # Index search has low cost

                # Generate index recommendations
                suggested_indexes = self._suggest_indexes_for_query(query, plan)

                return QueryPlan(
                    query=query,
                    plan=plan,
                    cost=cost,
                    uses_index=uses_index,
                    suggested_indexes=suggested_indexes
                )

        except Exception as e:
            self.logger.error(f"Failed to analyze query performance: {e}")
            return QueryPlan(query, [], 0.0, False, [])

    def _suggest_indexes_for_query(self, query: str, plan: List[Dict[str, Any]]) -> List[str]:
        """Suggest indexes for query"""
        suggestions = []

        try:
            # Simple heuristic analysis
            query_upper = query.upper()

            # Analyze WHERE clause
            if 'WHERE' in query_upper:
                # Extract possible index columns (simplified implementation)
                common_columns = ['id', 'created_at', 'importance', 'memory_type']
                for column in common_columns:
                    if column.upper() in query_upper:
                        suggestions.append(f"CREATE INDEX idx_{column} ON table_name({column})")

            # Analyze ORDER BY clause
            if 'ORDER BY' in query_upper:
                suggestions.append("Consider creating indexes for ORDER BY columns")

        except Exception as e:
            self.logger.warning(f"Failed to generate query index recommendations: {e}")

        return suggestions

    def optimize_indexes(self, auto_apply: bool = False) -> Dict[str, Any]:
        """
        Optimize indexes

        Args:
            auto_apply: Whether to automatically apply optimizations

        Returns:
            Optimization results
        """
        try:
            # Analyze current index status
            analysis = self.analyze_database_indexes()
            recommendations = analysis.get('recommendations', [])

            applied_changes = []
            skipped_changes = []

            if auto_apply and self.optimization_config['auto_drop_unused_indexes']:
                # Automatically drop unused indexes
                for rec in recommendations:
                    if rec['type'] == 'drop_unused' and rec['priority'] == 'high':
                        if self._drop_index(rec['index']):
                            applied_changes.append(f"Dropped unused index: {rec['index']}")
                        else:
                            skipped_changes.append(f"Failed to drop index: {rec['index']}")

            if auto_apply and self.optimization_config['auto_create_indexes']:
                # Automatically create suggested indexes
                for rec in recommendations:
                    if rec['type'] == 'create_index' and rec['priority'] in ['high', 'medium']:
                        suggested_columns = rec.get('suggested_columns', [])
                        if suggested_columns:
                            index_name = f"idx_{rec['table']}_{'_'.join(suggested_columns)}"
                            if self._create_index(index_name, rec['table'], suggested_columns):
                                applied_changes.append(f"Created index: {index_name}")
                            else:
                                skipped_changes.append(f"Failed to create index: {index_name}")

            return {
                'analysis': analysis,
                'applied_changes': applied_changes,
                'skipped_changes': skipped_changes,
                'auto_apply': auto_apply
            }

        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return {'error': str(e)}

    def _drop_index(self, index_name: str) -> bool:
        """Drop index"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                conn.commit()

            self.logger.info(f"Dropped index: {index_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to drop index {index_name}: {e}")
            return False

    def _create_index(self, index_name: str, table_name: str, columns: List[str]) -> bool:
        """Create index"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                columns_str = ', '.join(columns)
                sql = f"CREATE INDEX {index_name} ON {table_name}({columns_str})"
                cursor.execute(sql)
                conn.commit()

            self.logger.info(f"Created index: {index_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")
            return False

    def vacuum_database(self) -> bool:
        """Vacuum database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()

            self.logger.info("Database vacuum completed")
            return True

        except Exception as e:
            self.logger.error(f"Database vacuum failed: {e}")
            return False

    def analyze_statistics(self) -> bool:
        """Analyze database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("ANALYZE")
                conn.commit()

            self.logger.info("Database statistics analysis completed")
            return True

        except Exception as e:
            self.logger.error(f"Database statistics analysis failed: {e}")
            return False

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status"""
        try:
            analysis = self.analyze_database_indexes()

            return {
                'total_indexes': analysis.get('total_indexes', 0),
                'unused_indexes': len(analysis.get('unused_indexes', [])),
                'duplicate_indexes': len(analysis.get('duplicate_indexes', [])),
                'recommendations': len(analysis.get('recommendations', [])),
                'config': self.optimization_config.copy()
            }

        except Exception as e:
            self.logger.error(f"Failed to get optimization status: {e}")
            return {}

    def update_optimization_config(self, **kwargs) -> bool:
        """
        Update optimization configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.optimization_config:
                    self.optimization_config[key] = value
                    self.logger.info(f"Index optimization configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update index optimization configuration: {e}")
            return False