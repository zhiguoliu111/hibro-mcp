#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive learning mechanism database migration
Adds database table structures to support user behavior analysis, dynamic weight adjustment, and personalized recommendations
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

from .migration_manager import BaseMigration


class AdaptiveLearningMigration(BaseMigration):
    """Adaptive learning mechanism database migration"""

    VERSION = "2.2.0"
    DESCRIPTION = "Add adaptive learning mechanism related table structures: user behavior analysis, dynamic weight adjustment, personalized recommendations"

    def up(self) -> bool:
        """Execute migration upgrade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # 1. Create user behaviors table
                self._create_user_behaviors_table(conn)

                # 2. Create attention weights table
                self._create_attention_weights_table(conn)

                # 3. Create recommendation history table
                self._create_recommendation_history_table(conn)

                # 4. Create learning config table
                self._create_learning_config_table(conn)

                # 5. Create weight adjustment history table
                self._create_weight_adjustment_history_table(conn)

                # 6. Create user preference model table
                self._create_user_preference_model_table(conn)

                # 7. Create recommendation evaluation table
                self._create_recommendation_evaluation_table(conn)

                # 8. Create indexes
                self._create_indexes(conn)

                conn.commit()
                self.logger.info(f"Adaptive learning mechanism migration {self.VERSION} executed successfully")
                return True

        except Exception as e:
            self.logger.error(f"Adaptive learning mechanism migration failed: {e}")
            return False

    def down(self) -> bool:
        """Execute migration rollback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Drop all newly created tables
                tables_to_drop = [
                    'recommendation_evaluation',
                    'user_preference_model',
                    'weight_adjustment_history',
                    'learning_config',
                    'recommendation_history',
                    'attention_weights',
                    'user_behaviors'
                ]

                for table in tables_to_drop:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")

                conn.commit()
                self.logger.info(f"Adaptive learning mechanism migration {self.VERSION} rollback successful")
                return True

        except Exception as e:
            self.logger.error(f"Adaptive learning mechanism migration rollback failed: {e}")
            return False

    def _create_user_behaviors_table(self, conn: sqlite3.Connection):
        """Create user behaviors table"""
        conn.execute("""
            CREATE TABLE user_behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                action_type TEXT NOT NULL, -- 'query', 'store', 'recall', 'feedback', 'click', 'ignore'
                target_memory_id INTEGER,
                query_text TEXT,
                response_relevance REAL DEFAULT 0.0,
                user_feedback TEXT, -- 'useful', 'not_useful', 'partially_useful', 'very_useful'
                interaction_duration INTEGER DEFAULT 0, -- Interaction duration (seconds)
                context_data TEXT, -- JSON format context data
                project_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE SET NULL
            )
        """)

    def _create_attention_weights_table(self, conn: sqlite3.Connection):
        """Create attention weights table"""
        conn.execute("""
            CREATE TABLE attention_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL UNIQUE,
                weight REAL NOT NULL DEFAULT 1.0,
                decay_rate REAL NOT NULL DEFAULT 0.95, -- Weight decay rate
                access_count INTEGER DEFAULT 0, -- Access count
                last_access_time TIMESTAMP,
                boost_factor REAL DEFAULT 1.0, -- Weight boost factor
                category TEXT, -- 'technology', 'methodology', 'domain', 'project'
                project_path TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_recommendation_history_table(self, conn: sqlite3.Connection):
        """Create recommendation history table"""
        conn.execute("""
            CREATE TABLE recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_session TEXT NOT NULL,
                recommended_memory_id INTEGER NOT NULL,
                recommendation_type TEXT NOT NULL, -- 'collaborative', 'content_based', 'hybrid', 'causal', 'predictive'
                recommendation_source TEXT, -- Recommendation source algorithm
                confidence_score REAL NOT NULL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                user_action TEXT, -- 'clicked', 'ignored', 'saved', 'dismissed', 'rated'
                user_rating INTEGER, -- 1-5 star rating
                context_query TEXT, -- Query that triggered the recommendation
                project_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recommended_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)

    def _create_learning_config_table(self, conn: sqlite3.Connection):
        """Create learning config table"""
        conn.execute("""
            CREATE TABLE learning_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT NOT NULL UNIQUE,
                config_value TEXT NOT NULL, -- JSON format configuration value
                config_type TEXT NOT NULL, -- 'scoring_weights', 'learning_rate', 'decay_params', 'recommendation_params'
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                version TEXT DEFAULT '1.0',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_weight_adjustment_history_table(self, conn: sqlite3.Connection):
        """Create weight adjustment history table"""
        conn.execute("""
            CREATE TABLE weight_adjustment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                adjustment_type TEXT NOT NULL, -- 'scoring_factor', 'attention_weight', 'recommendation_weight'
                target_key TEXT NOT NULL, -- Target key for adjustment
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                adjustment_reason TEXT, -- Adjustment reason
                performance_metric REAL, -- Performance metric
                user_feedback_score REAL, -- User feedback score
                algorithm_used TEXT, -- Algorithm used for adjustment
                confidence REAL DEFAULT 0.0,
                project_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_user_preference_model_table(self, conn: sqlite3.Connection):
        """Create user preference model table"""
        conn.execute("""
            CREATE TABLE user_preference_model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT NOT NULL, -- 'content_type', 'technology', 'methodology', 'project_phase'
                preference_key TEXT NOT NULL,
                preference_value REAL NOT NULL DEFAULT 0.0, -- Preference strength
                confidence REAL DEFAULT 0.0, -- Preference confidence
                evidence_count INTEGER DEFAULT 0, -- Supporting evidence count
                last_reinforcement TIMESTAMP, -- Last reinforcement time
                decay_applied TIMESTAMP, -- Last decay application time
                project_path TEXT,
                category TEXT,
                metadata TEXT, -- JSON format metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(preference_type, preference_key, project_path)
            )
        """)

    def _create_recommendation_evaluation_table(self, conn: sqlite3.Connection):
        """Create recommendation evaluation table"""
        conn.execute("""
            CREATE TABLE recommendation_evaluation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_period TEXT NOT NULL, -- 'daily', 'weekly', 'monthly'
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                total_recommendations INTEGER DEFAULT 0,
                clicked_recommendations INTEGER DEFAULT 0,
                rated_recommendations INTEGER DEFAULT 0,
                average_rating REAL DEFAULT 0.0,
                click_through_rate REAL DEFAULT 0.0,
                precision_at_k REAL DEFAULT 0.0, -- P@K metric
                recall_at_k REAL DEFAULT 0.0, -- R@K metric
                f1_score REAL DEFAULT 0.0,
                diversity_score REAL DEFAULT 0.0, -- Recommendation diversity
                novelty_score REAL DEFAULT 0.0, -- Recommendation novelty
                algorithm_performance TEXT, -- JSON format algorithm performance
                project_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create indexes"""
        indexes = [
            # user_behaviors table indexes
            "CREATE INDEX idx_user_behaviors_session ON user_behaviors(session_id)",
            "CREATE INDEX idx_user_behaviors_action_type ON user_behaviors(action_type)",
            "CREATE INDEX idx_user_behaviors_timestamp ON user_behaviors(timestamp)",
            "CREATE INDEX idx_user_behaviors_project ON user_behaviors(project_path)",
            "CREATE INDEX idx_user_behaviors_memory ON user_behaviors(target_memory_id)",

            # attention_weights table indexes
            "CREATE INDEX idx_attention_weights_topic ON attention_weights(topic)",
            "CREATE INDEX idx_attention_weights_category ON attention_weights(category)",
            "CREATE INDEX idx_attention_weights_project ON attention_weights(project_path)",
            "CREATE INDEX idx_attention_weights_weight ON attention_weights(weight DESC)",

            # recommendation_history table indexes
            "CREATE INDEX idx_recommendation_history_session ON recommendation_history(user_session)",
            "CREATE INDEX idx_recommendation_history_type ON recommendation_history(recommendation_type)",
            "CREATE INDEX idx_recommendation_history_memory ON recommendation_history(recommended_memory_id)",
            "CREATE INDEX idx_recommendation_history_timestamp ON recommendation_history(timestamp)",
            "CREATE INDEX idx_recommendation_history_project ON recommendation_history(project_path)",

            # learning_config table indexes
            "CREATE INDEX idx_learning_config_key ON learning_config(config_key)",
            "CREATE INDEX idx_learning_config_type ON learning_config(config_type)",
            "CREATE INDEX idx_learning_config_active ON learning_config(is_active)",

            # weight_adjustment_history table indexes
            "CREATE INDEX idx_weight_adjustment_type ON weight_adjustment_history(adjustment_type)",
            "CREATE INDEX idx_weight_adjustment_target ON weight_adjustment_history(target_key)",
            "CREATE INDEX idx_weight_adjustment_timestamp ON weight_adjustment_history(timestamp)",

            # user_preference_model table indexes
            "CREATE INDEX idx_user_preference_type ON user_preference_model(preference_type)",
            "CREATE INDEX idx_user_preference_key ON user_preference_model(preference_key)",
            "CREATE INDEX idx_user_preference_value ON user_preference_model(preference_value DESC)",
            "CREATE INDEX idx_user_preference_project ON user_preference_model(project_path)",

            # recommendation_evaluation table indexes
            "CREATE INDEX idx_recommendation_evaluation_period ON recommendation_evaluation(evaluation_period)",
            "CREATE INDEX idx_recommendation_evaluation_date ON recommendation_evaluation(start_date, end_date)",
            "CREATE INDEX idx_recommendation_evaluation_project ON recommendation_evaluation(project_path)"
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    def get_verification_queries(self) -> List[str]:
        """Get verification queries"""
        return [
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%behavior%' OR name LIKE '%attention%' OR name LIKE '%recommendation%' OR name LIKE '%learning%' OR name LIKE '%preference%'",
            "SELECT COUNT(*) as index_count FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'",
            "PRAGMA table_info(user_behaviors)",
            "PRAGMA table_info(attention_weights)",
            "PRAGMA table_info(recommendation_history)"
        ]