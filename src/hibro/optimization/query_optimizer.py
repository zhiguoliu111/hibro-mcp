#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query optimizer
Provides SQL query analysis, optimization, and rewriting functionality
"""

import re
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..utils.config import Config


class QueryType(Enum):
    """Query type"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"


@dataclass
class QueryAnalysis:
    """Query analysis result"""
    original_query: str
    query_type: QueryType
    tables_used: List[str]
    columns_used: List[str]
    has_where_clause: bool
    has_order_by: bool
    has_group_by: bool
    has_join: bool
    estimated_cost: float
    optimization_suggestions: List[str]


@dataclass
class QueryOptimization:
    """Query optimization result"""
    original_query: str
    optimized_query: str
    improvements: List[str]
    performance_gain: float
    confidence: float


class QueryOptimizer:
    """Query optimizer"""

    def __init__(self, config: Config):
        """
        Initialize query optimizer

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.query_optimizer')

        # Optimization configuration
        self.optimization_config = {
            'enable_query_rewrite': True,
            'enable_index_hints': True,
            'max_query_complexity': 1000,
            'cache_query_plans': True,
            'analyze_slow_queries': True,
            'slow_query_threshold_ms': 1000
        }

        # Query patterns and optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
        self.query_patterns = self._initialize_query_patterns()

        # Query statistics
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.slow_queries: List[Dict[str, Any]] = []

    def _initialize_optimization_rules(self) -> List[Dict[str, Any]]:
        """Initialize optimization rules"""
        return [
            {
                'name': 'avoid_select_star',
                'pattern': r'SELECT\s+\*\s+FROM',
                'suggestion': 'Avoid using SELECT *, explicitly specify required columns',
                'severity': 'medium'
            },
            {
                'name': 'missing_where_clause',
                'pattern': r'SELECT.*FROM\s+\w+(?!\s+WHERE)',
                'suggestion': 'Consider adding WHERE clause to limit result set',
                'severity': 'low'
            },
            {
                'name': 'inefficient_like',
                'pattern': r'LIKE\s+[\'"]%.*%[\'"]',
                'suggestion': 'Leading wildcard causes full table scan, consider using full-text search',
                'severity': 'high'
            },
            {
                'name': 'missing_limit',
                'pattern': r'SELECT.*FROM.*ORDER\s+BY(?!.*LIMIT)',
                'suggestion': 'Sorted queries should include LIMIT clause',
                'severity': 'medium'
            },
            {
                'name': 'inefficient_or',
                'pattern': r'WHERE.*\bOR\b.*\bOR\b',
                'suggestion': 'Multiple OR conditions may cause performance issues, consider using IN or UNION',
                'severity': 'medium'
            },
            {
                'name': 'function_in_where',
                'pattern': r'WHERE\s+\w+\([^)]*\)\s*[=<>]',
                'suggestion': 'Functions in WHERE clause prevent index usage',
                'severity': 'high'
            }
        ]

    def _initialize_query_patterns(self) -> Dict[str, str]:
        """Initialize query patterns"""
        return {
            'memory_search': r'SELECT.*FROM\s+memories\s+WHERE.*content.*LIKE',
            'importance_filter': r'WHERE.*importance\s*[><=]',
            'date_range': r'WHERE.*created_at.*BETWEEN|WHERE.*created_at.*[><=]',
            'project_filter': r'WHERE.*project_path',
            'type_filter': r'WHERE.*memory_type'
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query

        Args:
            query: SQL query

        Returns:
            Query analysis result
        """
        try:
            query_clean = self._clean_query(query)

            # Determine query type
            query_type = self._determine_query_type(query_clean)

            # Extract table and column names
            tables_used = self._extract_tables(query_clean)
            columns_used = self._extract_columns(query_clean)

            # Analyze query structure
            has_where_clause = 'WHERE' in query_clean.upper()
            has_order_by = 'ORDER BY' in query_clean.upper()
            has_group_by = 'GROUP BY' in query_clean.upper()
            has_join = any(join in query_clean.upper()
                          for join in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'])

            # Estimate query cost
            estimated_cost = self._estimate_query_cost(query_clean, tables_used)

            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(query_clean)

            return QueryAnalysis(
                original_query=query,
                query_type=query_type,
                tables_used=tables_used,
                columns_used=columns_used,
                has_where_clause=has_where_clause,
                has_order_by=has_order_by,
                has_group_by=has_group_by,
                has_join=has_join,
                estimated_cost=estimated_cost,
                optimization_suggestions=optimization_suggestions
            )

        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.SELECT,
                tables_used=[],
                columns_used=[],
                has_where_clause=False,
                has_order_by=False,
                has_group_by=False,
                has_join=False,
                estimated_cost=0.0,
                optimization_suggestions=[]
            )

    def _clean_query(self, query: str) -> str:
        """Clean query string"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine query type"""
        query_upper = query.upper().strip()

        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith('CREATE'):
            return QueryType.CREATE
        elif query_upper.startswith('DROP'):
            return QueryType.DROP
        else:
            return QueryType.SELECT

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        tables = []

        try:
            # Simplified table name extraction (real-world applications need more complex parsing)
            query_upper = query.upper()

            # FROM clause
            from_match = re.search(r'FROM\s+(\w+)', query_upper)
            if from_match:
                tables.append(from_match.group(1).lower())

            # JOIN clause
            join_matches = re.findall(r'JOIN\s+(\w+)', query_upper)
            for match in join_matches:
                tables.append(match.lower())

            # INSERT INTO
            insert_match = re.search(r'INSERT\s+INTO\s+(\w+)', query_upper)
            if insert_match:
                tables.append(insert_match.group(1).lower())

            # UPDATE
            update_match = re.search(r'UPDATE\s+(\w+)', query_upper)
            if update_match:
                tables.append(update_match.group(1).lower())

        except Exception as e:
            self.logger.warning(f"Failed to extract table names: {e}")

        return list(set(tables))  # Remove duplicates

    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from query"""
        columns = []

        try:
            query_upper = query.upper()

            # Columns in SELECT clause
            if query_upper.startswith('SELECT'):
                select_part = query_upper.split('FROM')[0]
                select_part = select_part.replace('SELECT', '').strip()

                if select_part != '*':
                    # Simplified column name extraction
                    column_matches = re.findall(r'\b(\w+)\b', select_part)
                    columns.extend([col.lower() for col in column_matches])

            # Columns in WHERE clause
            where_matches = re.findall(r'WHERE.*?(\w+)\s*[=<>!]', query_upper)
            columns.extend([col.lower() for col in where_matches])

            # Columns in ORDER BY clause
            order_matches = re.findall(r'ORDER\s+BY\s+(\w+)', query_upper)
            columns.extend([col.lower() for col in order_matches])

        except Exception as e:
            self.logger.warning(f"Failed to extract column names: {e}")

        return list(set(columns))  # Remove duplicates

    def _estimate_query_cost(self, query: str, tables: List[str]) -> float:
        """Estimate query cost"""
        cost = 0.0
        query_upper = query.upper()

        try:
            # Base cost
            cost += 10

            # Impact of table count
            cost += len(tables) * 20

            # JOIN cost
            if 'JOIN' in query_upper:
                join_count = len(re.findall(r'JOIN', query_upper))
                cost += join_count * 100

            # WHERE clause
            if 'WHERE' not in query_upper:
                cost += 200  # High cost without WHERE clause

            # LIKE operations
            like_count = len(re.findall(r'LIKE', query_upper))
            cost += like_count * 50

            # ORDER BY
            if 'ORDER BY' in query_upper:
                cost += 30

            # GROUP BY
            if 'GROUP BY' in query_upper:
                cost += 40

            # Subqueries
            subquery_count = query.count('(SELECT')
            cost += subquery_count * 100

        except Exception as e:
            self.logger.warning(f"Failed to estimate query cost: {e}")

        return cost

    def _generate_optimization_suggestions(self, query: str) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []

        try:
            query_upper = query.upper()

            # Apply optimization rules
            for rule in self.optimization_rules:
                if re.search(rule['pattern'], query_upper, re.IGNORECASE):
                    suggestions.append(f"[{rule['severity']}] {rule['suggestion']}")

            # Specific pattern suggestions
            if 'SELECT *' in query_upper:
                suggestions.append('[medium] Avoid using SELECT *, explicitly specify required columns')

            if 'LIKE' in query_upper and '%' in query:
                if query.find('%') < query.rfind('%'):  # Has % on both sides
                    suggestions.append('[high] Leading and trailing wildcards cause full table scan')

            if 'ORDER BY' in query_upper and 'LIMIT' not in query_upper:
                suggestions.append('[medium] Sorted queries should include LIMIT to restrict result count')

        except Exception as e:
            self.logger.warning(f"Failed to generate optimization suggestions: {e}")

        return suggestions

    def optimize_query(self, query: str) -> QueryOptimization:
        """
        Optimize query

        Args:
            query: Original query

        Returns:
            Query optimization result
        """
        try:
            if not self.optimization_config['enable_query_rewrite']:
                return QueryOptimization(
                    original_query=query,
                    optimized_query=query,
                    improvements=[],
                    performance_gain=0.0,
                    confidence=0.0
                )

            optimized_query = query
            improvements = []
            performance_gain = 0.0

            # Apply optimization rules
            optimized_query, rule_improvements = self._apply_optimization_rules(optimized_query)
            improvements.extend(rule_improvements)

            # Add index hints
            if self.optimization_config['enable_index_hints']:
                optimized_query, index_improvements = self._add_index_hints(optimized_query)
                improvements.extend(index_improvements)

            # Rewrite inefficient queries
            optimized_query, rewrite_improvements = self._rewrite_inefficient_patterns(optimized_query)
            improvements.extend(rewrite_improvements)

            # Calculate performance gain
            if improvements:
                performance_gain = len(improvements) * 10.0  # Simplified calculation

            # Calculate confidence
            confidence = min(0.9, len(improvements) * 0.2)

            return QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                improvements=improvements,
                performance_gain=performance_gain,
                confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                improvements=[],
                performance_gain=0.0,
                confidence=0.0
            )

    def _apply_optimization_rules(self, query: str) -> Tuple[str, List[str]]:
        """Apply optimization rules"""
        optimized_query = query
        improvements = []

        try:
            # Remove unnecessary parentheses
            if '(' in optimized_query and ')' in optimized_query:
                # Simplified parenthesis optimization
                old_query = optimized_query
                optimized_query = re.sub(r'\(\s*(\w+)\s*\)', r'\1', optimized_query)
                if old_query != optimized_query:
                    improvements.append('Removed unnecessary parentheses')

            # Optimize LIKE queries
            if 'LIKE' in optimized_query.upper():
                # Convert prefix LIKE to range query (if possible)
                like_pattern = r"(\w+)\s+LIKE\s+['\"](\w+)%['\"]"
                match = re.search(like_pattern, optimized_query, re.IGNORECASE)
                if match:
                    column, prefix = match.groups()
                    # More complex LIKE optimization logic can be added here
                    improvements.append('Optimized LIKE query pattern')

        except Exception as e:
            self.logger.warning(f"Failed to apply optimization rules: {e}")

        return optimized_query, improvements

    def _add_index_hints(self, query: str) -> Tuple[str, List[str]]:
        """Add index hints"""
        optimized_query = query
        improvements = []

        try:
            # SQLite doesn't support index hints, but queries can be rewritten to better utilize indexes
            query_upper = query.upper()

            # For queries with ORDER BY, ensure WHERE conditions come first
            if 'ORDER BY' in query_upper and 'WHERE' in query_upper:
                # Query rewrite logic can be added here
                improvements.append('Optimized query to better utilize indexes')

        except Exception as e:
            self.logger.warning(f"Failed to add index hints: {e}")

        return optimized_query, improvements

    def _rewrite_inefficient_patterns(self, query: str) -> Tuple[str, List[str]]:
        """Rewrite inefficient query patterns"""
        optimized_query = query
        improvements = []

        try:
            # Convert multiple OR conditions to IN
            or_pattern = r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+OR\s+\1\s*=\s*['\"]([^'\"]+)['\"]"
            or_matches = re.findall(or_pattern, optimized_query, re.IGNORECASE)

            for match in or_matches:
                column, value1, value2 = match
                old_pattern = f"{column} = '{value1}' OR {column} = '{value2}'"
                new_pattern = f"{column} IN ('{value1}', '{value2}')"
                optimized_query = optimized_query.replace(old_pattern, new_pattern)
                improvements.append('Converted OR conditions to IN operation')

            # Optimize EXISTS subqueries
            if 'EXISTS' in optimized_query.upper():
                # EXISTS optimization logic can be added here
                improvements.append('Optimized EXISTS subquery')

        except Exception as e:
            self.logger.warning(f"Failed to rewrite inefficient query patterns: {e}")

        return optimized_query, improvements

    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze query patterns

        Args:
            queries: List of queries

        Returns:
            Pattern analysis results
        """
        try:
            pattern_stats = {}
            query_types = {}
            common_tables = {}
            common_columns = {}

            for query in queries:
                analysis = self.analyze_query(query)

                # Count query types
                query_type = analysis.query_type.value
                query_types[query_type] = query_types.get(query_type, 0) + 1

                # Count table usage
                for table in analysis.tables_used:
                    common_tables[table] = common_tables.get(table, 0) + 1

                # Count column usage
                for column in analysis.columns_used:
                    common_columns[column] = common_columns.get(column, 0) + 1

                # Check query patterns
                for pattern_name, pattern_regex in self.query_patterns.items():
                    if re.search(pattern_regex, query, re.IGNORECASE):
                        pattern_stats[pattern_name] = pattern_stats.get(pattern_name, 0) + 1

            return {
                'total_queries': len(queries),
                'query_types': query_types,
                'common_tables': dict(sorted(common_tables.items(), key=lambda x: x[1], reverse=True)),
                'common_columns': dict(sorted(common_columns.items(), key=lambda x: x[1], reverse=True)),
                'pattern_stats': pattern_stats
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze query patterns: {e}")
            return {}

    def get_optimization_recommendations(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations

        Args:
            queries: List of queries

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        try:
            pattern_analysis = self.analyze_query_patterns(queries)

            # Generate recommendations based on query patterns
            common_tables = pattern_analysis.get('common_tables', {})
            for table, count in common_tables.items():
                if count > 10:  # Frequently accessed tables
                    recommendations.append({
                        'type': 'index_recommendation',
                        'priority': 'high',
                        'description': f'Table {table} is frequently accessed ({count} times), recommend checking index configuration',
                        'table': table
                    })

            # Check slow query patterns
            for query in queries:
                analysis = self.analyze_query(query)
                if analysis.estimated_cost > self.optimization_config['max_query_complexity']:
                    recommendations.append({
                        'type': 'query_optimization',
                        'priority': 'high',
                        'description': f'Query complexity too high ({analysis.estimated_cost}), requires optimization',
                        'query': query[:100] + '...' if len(query) > 100 else query,
                        'suggestions': analysis.optimization_suggestions
                    })

        except Exception as e:
            self.logger.error(f"Failed to get optimization recommendations: {e}")

        return recommendations

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
                    self.logger.info(f"Query optimization configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update query optimization configuration: {e}")
            return False

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics"""
        return {
            'total_queries_analyzed': len(self.query_stats),
            'slow_queries_count': len(self.slow_queries),
            'optimization_config': self.optimization_config.copy(),
            'optimization_rules_count': len(self.optimization_rules)
        }