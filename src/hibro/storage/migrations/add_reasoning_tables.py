#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning functionality database migration
Adds database table structures to support causal analysis, predictive reasoning, and knowledge graphs
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

from .migration_manager import BaseMigration


class ReasoningTablesMigration(BaseMigration):
    """Reasoning functionality database migration"""

    VERSION = "2.1.0"
    DESCRIPTION = "Add reasoning functionality related table structures: causal analysis, predictive reasoning, knowledge graphs"

    def up(self) -> bool:
        """Execute migration upgrade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # 1. Extend existing memory_relations table
                self._extend_memory_relations_table(conn)

                # 2. Create causal relationship related tables
                self._create_causal_tables(conn)

                # 3. Create knowledge graph related tables
                self._create_knowledge_graph_tables(conn)

                # 4. Create prediction functionality related tables
                self._create_prediction_tables(conn)

                # 5. Create user behavior analysis tables
                self._create_behavior_analysis_tables(conn)

                # 6. Create statistical analysis tables
                self._create_statistics_tables(conn)

                # 7. Create indexes
                self._create_indexes(conn)

                conn.commit()

            self.logger.info(f"Reasoning functionality migration completed: {self.VERSION}")
            return True

        except Exception as e:
            self.logger.error(f"Reasoning functionality migration failed: {e}")
            return False

    def down(self) -> bool:
        """Rollback migration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Drop newly added tables (in reverse dependency order)
                tables_to_drop = [
                    'reasoning_statistics',
                    'user_behaviors',
                    'predictions',
                    'prediction_patterns',
                    'concept_relations',
                    'concepts',
                    'causal_chains'
                ]

                for table in tables_to_drop:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")

                # Rollback memory_relations table modifications
                self._rollback_memory_relations_table(conn)

                conn.commit()

            self.logger.info(f"Reasoning functionality migration rollback completed: {self.VERSION}")
            return True

        except Exception as e:
            self.logger.error(f"Reasoning functionality migration rollback failed: {e}")
            return False

    def _extend_memory_relations_table(self, conn: sqlite3.Connection):
        """Extend memory_relations table structure"""
        self.logger.info("Extending memory_relations table structure")

        # Check if already extended
        cursor = conn.execute("PRAGMA table_info(memory_relations)")
        columns = [row[1] for row in cursor.fetchall()]

        # Add new columns (if they don't exist)
        new_columns = [
            ("relation_strength", "REAL DEFAULT 0.5"),
            ("causal_type", "TEXT"),
            ("confidence_score", "REAL DEFAULT 0.5"),
            ("created_by", "TEXT DEFAULT 'manual'"),
            ("evidence", "TEXT"),
            ("pattern_matched", "TEXT")
        ]

        for column_name, column_def in new_columns:
            if column_name not in columns:
                conn.execute(f"ALTER TABLE memory_relations ADD COLUMN {column_name} {column_def}")
                self.logger.info(f"Added column: memory_relations.{column_name}")

        # Add check constraints (SQLite 3.37+ support)
        try:
            conn.execute("""
                ALTER TABLE memory_relations ADD CONSTRAINT chk_relation_strength
                CHECK (relation_strength IS NULL OR (relation_strength >= 0.0 AND relation_strength <= 1.0))
            """)
        except sqlite3.OperationalError:
            # If constraint already exists or SQLite version doesn't support, ignore error
            pass

        try:
            conn.execute("""
                ALTER TABLE memory_relations ADD CONSTRAINT chk_confidence_score_ext
                CHECK (confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0))
            """)
        except sqlite3.OperationalError:
            pass

    def _create_causal_tables(self, conn: sqlite3.Connection):
        """Create causal relationship related tables"""
        self.logger.info("Creating causal relationship related tables")

        # Causal relationship chain table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_id TEXT UNIQUE NOT NULL,
                root_cause_memory_id INTEGER NOT NULL,
                final_effect_memory_id INTEGER NOT NULL,
                chain_length INTEGER NOT NULL,
                total_strength REAL NOT NULL,
                relations_data TEXT NOT NULL, -- JSON format relationship chain storage
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (root_cause_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (final_effect_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                CONSTRAINT chk_chain_length CHECK (chain_length >= 2),
                CONSTRAINT chk_total_strength CHECK (total_strength >= 0.0 AND total_strength <= 1.0)
            )
        """)

    def _create_knowledge_graph_tables(self, conn: sqlite3.Connection):
        """Create knowledge graph related tables"""
        self.logger.info("Creating knowledge graph related tables")

        # Concepts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL, -- 'technology', 'methodology', 'domain', 'general'
                frequency INTEGER DEFAULT 1,
                importance REAL DEFAULT 0.5,
                aliases TEXT, -- JSON format alias list
                related_memories TEXT, -- JSON format related memory ID list
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT chk_frequency CHECK (frequency >= 1),
                CONSTRAINT chk_concept_importance CHECK (importance >= 0.0 AND importance <= 1.0)
            )
        """)

        # Concept relations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concept_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_id TEXT UNIQUE NOT NULL,
                concept1_id TEXT NOT NULL,
                concept2_id TEXT NOT NULL,
                relation_type TEXT NOT NULL, -- 'similar', 'causal', 'hierarchical', 'temporal', 'categorical'
                weight REAL NOT NULL,
                evidence_count INTEGER DEFAULT 1,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept1_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
                FOREIGN KEY (concept2_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
                CONSTRAINT chk_relation_weight CHECK (weight >= 0.0 AND weight <= 1.0),
                CONSTRAINT chk_relation_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
                CONSTRAINT chk_different_concepts CHECK (concept1_id != concept2_id)
            )
        """)

    def _create_prediction_tables(self, conn: sqlite3.Connection):
        """Create prediction functionality related tables"""
        self.logger.info("Creating prediction functionality related tables")

        # Prediction patterns table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL, -- 'decision', 'project_phase', 'tech_stack', 'causal'
                trigger_conditions TEXT NOT NULL, -- JSON format trigger conditions
                typical_sequence TEXT NOT NULL, -- JSON format typical sequence
                success_rate REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                project_path TEXT, -- Project-specific patterns, NULL indicates global patterns
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT chk_pattern_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
                CONSTRAINT chk_usage_count CHECK (usage_count >= 0)
            )
        """)

        # Prediction results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE NOT NULL,
                prediction_type TEXT NOT NULL, -- 'next_need', 'tech_choice', 'project_phase', 'importance_trend'
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                probability REAL NOT NULL,
                evidence TEXT, -- JSON format evidence list
                related_memories TEXT, -- JSON format related memory ID list
                time_horizon_days INTEGER, -- Prediction time range (days)
                project_path TEXT, -- Project-specific predictions
                status TEXT DEFAULT 'pending', -- 'pending', 'validated', 'failed', 'expired'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                validated_at TIMESTAMP,
                CONSTRAINT chk_prediction_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
                CONSTRAINT chk_prediction_probability CHECK (probability >= 0.0 AND probability <= 1.0)
            )
        """)

    def _create_behavior_analysis_tables(self, conn: sqlite3.Connection):
        """Create user behavior analysis tables"""
        self.logger.info("Creating user behavior analysis tables")

        # User behaviors table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                action_type TEXT NOT NULL, -- 'query', 'store', 'recall', 'feedback', 'prediction_request'
                target_memory_id INTEGER,
                query_text TEXT,
                response_relevance REAL,
                user_feedback TEXT, -- 'useful', 'not_useful', 'partially_useful'
                context_data TEXT, -- JSON format context data
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE SET NULL,
                CONSTRAINT chk_response_relevance CHECK (response_relevance IS NULL OR (response_relevance >= 0.0 AND response_relevance <= 1.0))
            )
        """)

    def _create_statistics_tables(self, conn: sqlite3.Connection):
        """Create statistical analysis tables"""
        self.logger.info("Creating statistical analysis tables")

        # Reasoning statistics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_type TEXT NOT NULL, -- 'causal_analysis', 'prediction_accuracy', 'concept_growth'
                project_path TEXT, -- Project-specific statistics, NULL indicates global statistics
                stat_data TEXT NOT NULL, -- JSON format statistical data
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create indexes"""
        self.logger.info("Creating reasoning functionality related indexes")

        # memory_relations table new indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_causal_type ON memory_relations(causal_type)",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_strength ON memory_relations(relation_strength)",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_confidence ON memory_relations(confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_memory_relations_created_by ON memory_relations(created_by)",

            # causal_chains table indexes
            "CREATE INDEX IF NOT EXISTS idx_causal_chains_root_cause ON causal_chains(root_cause_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_causal_chains_final_effect ON causal_chains(final_effect_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_causal_chains_strength ON causal_chains(total_strength)",
            "CREATE INDEX IF NOT EXISTS idx_causal_chains_length ON causal_chains(chain_length)",

            # concepts table indexes
            "CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)",
            "CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category)",
            "CREATE INDEX IF NOT EXISTS idx_concepts_frequency ON concepts(frequency)",
            "CREATE INDEX IF NOT EXISTS idx_concepts_importance ON concepts(importance)",

            # concept_relations table indexes
            "CREATE INDEX IF NOT EXISTS idx_concept_relations_concept1 ON concept_relations(concept1_id)",
            "CREATE INDEX IF NOT EXISTS idx_concept_relations_concept2 ON concept_relations(concept2_id)",
            "CREATE INDEX IF NOT EXISTS idx_concept_relations_type ON concept_relations(relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_concept_relations_weight ON concept_relations(weight)",

            # prediction_patterns table indexes
            "CREATE INDEX IF NOT EXISTS idx_prediction_patterns_type ON prediction_patterns(pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_patterns_project ON prediction_patterns(project_path)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_patterns_success_rate ON prediction_patterns(success_rate)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_patterns_last_used ON prediction_patterns(last_used)",

            # predictions table indexes
            "CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_project ON predictions(project_path)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_expires_at ON predictions(expires_at)",

            # user_behaviors table indexes
            "CREATE INDEX IF NOT EXISTS idx_user_behaviors_session ON user_behaviors(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_behaviors_action_type ON user_behaviors(action_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_behaviors_timestamp ON user_behaviors(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_user_behaviors_memory_id ON user_behaviors(target_memory_id)",

            # reasoning_statistics table indexes
            "CREATE INDEX IF NOT EXISTS idx_reasoning_statistics_type ON reasoning_statistics(stat_type)",
            "CREATE INDEX IF NOT EXISTS idx_reasoning_statistics_project ON reasoning_statistics(project_path)",
            "CREATE INDEX IF NOT EXISTS idx_reasoning_statistics_period ON reasoning_statistics(period_start, period_end)"
        ]

        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError as e:
                # Index may already exist, log warning but continue
                self.logger.warning(f"Index creation warning: {e}")

    def _rollback_memory_relations_table(self, conn: sqlite3.Connection):
        """Rollback memory_relations table modifications"""
        self.logger.info("Rolling back memory_relations table modifications")

        # SQLite doesn't support DROP COLUMN, need to rebuild table
        try:
            # 1. Create backup table
            conn.execute("""
                CREATE TABLE memory_relations_backup AS
                SELECT id, memory_id, related_id, relation_type, strength, created_at
                FROM memory_relations
            """)

            # 2. Drop original table
            conn.execute("DROP TABLE memory_relations")

            # 3. Recreate original table structure
            conn.execute("""
                CREATE TABLE memory_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    related_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (related_id) REFERENCES memories(id) ON DELETE CASCADE,
                    CONSTRAINT chk_strength CHECK (strength >= 0.0 AND strength <= 1.0),
                    CONSTRAINT chk_different_memories CHECK (memory_id != related_id)
                )
            """)

            # 4. Restore data
            conn.execute("""
                INSERT INTO memory_relations (id, memory_id, related_id, relation_type, strength, created_at)
                SELECT id, memory_id, related_id, relation_type, strength, created_at
                FROM memory_relations_backup
            """)

            # 5. Drop backup table
            conn.execute("DROP TABLE memory_relations_backup")

            # 6. Rebuild original indexes
            original_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_memory_relations_memory_id ON memory_relations(memory_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_relations_related_id ON memory_relations(related_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_relations_type ON memory_relations(relation_type)"
            ]

            for index_sql in original_indexes:
                conn.execute(index_sql)

            self.logger.info("memory_relations table rollback completed")

        except Exception as e:
            self.logger.error(f"memory_relations table rollback failed: {e}")
            raise

    def get_migration_info(self) -> Dict[str, Any]:
        """Get migration information"""
        return {
            'version': self.VERSION,
            'description': self.DESCRIPTION,
            'tables_created': [
                'causal_chains',
                'concepts',
                'concept_relations',
                'prediction_patterns',
                'predictions',
                'user_behaviors',
                'reasoning_statistics'
            ],
            'tables_modified': [
                'memory_relations'
            ],
            'indexes_created': 25,
            'estimated_size_increase': '10-50MB (depending on usage)'
        }