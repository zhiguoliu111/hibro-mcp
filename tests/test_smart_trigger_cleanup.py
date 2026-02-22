#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Smart Trigger and Memory Cleanup System
智能触发与记忆清理系统测试
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hibro.utils.constants import (
    # Cleanup constants
    CLEANUP_INTERVAL_HOURS,
    CLEANUP_TIME_OF_DAY,
    MEMORY_THRESHOLD_WARNING,
    MEMORY_THRESHOLD_CLEANUP,
    MEMORY_THRESHOLD_CRITICAL,
    LFU_CLEANUP_BOTTOM_PERCENT,
    TIME_EXPIRY_DAYS_DEFAULT,
    TIME_EXPIRY_DAYS_LOW_IMPORTANCE,
    IMPORTANCE_THRESHOLD_CLEANUP,
    NEW_MEMORY_PROTECTION_DAYS,
    # Trigger constants
    TRIGGER_HIGH_CONFIDENCE,
    TRIGGER_MEDIUM_CONFIDENCE,
    TRIGGER_LOW_CONFIDENCE,
    SEMANTIC_SIMILARITY_THRESHOLD,
    # Query keywords
    QUERY_KEYWORDS_PROJECT_META,
    QUERY_KEYWORDS_PROJECT_SCAN,
    QUERY_KEYWORDS_MEMORY_STORE,
    QUERY_KEYWORDS_MEMORY_QUERY,
    QUERY_KEYWORDS_TECH_STACK,
)
from hibro.intelligence.query_analyzer import QueryAnalyzer, QueryKeywords
from hibro.intelligence.trigger_executor import TriggerExecutor


class TestConstants:
    """Test constants are defined correctly / 测试常量定义正确"""

    def test_cleanup_constants_exist(self):
        """Cleanup constants should be defined / 清理常量应已定义"""
        assert CLEANUP_INTERVAL_HOURS == 24
        assert CLEANUP_TIME_OF_DAY == "03:00"
        assert MEMORY_THRESHOLD_WARNING == 0.7
        assert MEMORY_THRESHOLD_CLEANUP == 0.85
        assert MEMORY_THRESHOLD_CRITICAL == 0.95
        assert LFU_CLEANUP_BOTTOM_PERCENT == 0.2
        assert TIME_EXPIRY_DAYS_DEFAULT == 365
        assert TIME_EXPIRY_DAYS_LOW_IMPORTANCE == 90
        assert IMPORTANCE_THRESHOLD_CLEANUP == 0.2
        assert NEW_MEMORY_PROTECTION_DAYS == 30

    def test_trigger_constants_exist(self):
        """Trigger constants should be defined / 触发常量应已定义"""
        assert TRIGGER_HIGH_CONFIDENCE == 0.8
        assert TRIGGER_MEDIUM_CONFIDENCE == 0.5
        assert TRIGGER_LOW_CONFIDENCE == 0.3
        assert SEMANTIC_SIMILARITY_THRESHOLD == 0.5

    def test_query_keywords_exist(self):
        """Query keywords should be defined / 查询关键词应已定义"""
        assert len(QUERY_KEYWORDS_PROJECT_META) > 0
        assert len(QUERY_KEYWORDS_PROJECT_SCAN) > 0
        assert len(QUERY_KEYWORDS_MEMORY_STORE) > 0
        assert len(QUERY_KEYWORDS_MEMORY_QUERY) > 0
        assert len(QUERY_KEYWORDS_TECH_STACK) > 0

        # Check some specific keywords
        assert "project" in QUERY_KEYWORDS_PROJECT_META
        assert "项目" in QUERY_KEYWORDS_PROJECT_META
        assert "remember" in QUERY_KEYWORDS_MEMORY_STORE
        assert "记住" in QUERY_KEYWORDS_MEMORY_STORE


class TestQueryAnalyzer:
    """Test QueryAnalyzer functionality / 测试查询分析器功能"""

    def setup_method(self):
        """Setup test fixtures / 设置测试固件"""
        self.analyzer = QueryAnalyzer(memory_engine=None, project_scanner=None)

    def test_analyze_project_meta_query(self):
        """Should recognize project meta queries / 应识别项目元信息查询"""
        result = self.analyzer.analyze("项目进度怎么样？", "/test/project")
        assert result["is_project_related"] is True
        assert result["confidence"] >= 0.5
        assert "project_meta" in result["matched_keywords"]

    def test_analyze_project_scan_query(self):
        """Should recognize project scan queries / 应识别项目扫描查询"""
        result = self.analyzer.analyze("扫描项目结构", "/test/project")
        assert result["is_project_related"] is True
        assert result["confidence"] >= 0.5
        assert "project_scan" in result["matched_keywords"]

    def test_analyze_memory_store_query(self):
        """Should recognize memory store queries / 应识别记忆存储查询"""
        result = self.analyzer.analyze("记住这个配置", "/test/project")
        assert result["is_project_related"] is True
        assert "memory_store" in result["matched_keywords"]

    def test_analyze_memory_query_query(self):
        """Should recognize memory query queries / 应识别记忆查询"""
        result = self.analyzer.analyze("之前的项目决策是什么", "/test/project")
        assert result["is_project_related"] is True
        assert "memory_query" in result["matched_keywords"]

    def test_analyze_tech_stack_query(self):
        """Should recognize tech stack queries / 应识别技术栈查询"""
        result = self.analyzer.analyze("这个react项目的架构是什么", "/test/project")
        assert result["is_project_related"] is True
        assert "tech_stack" in result["matched_keywords"]

    def test_analyze_unrelated_query(self):
        """Should not trigger on unrelated queries / 不应触发无关查询"""
        result = self.analyzer.analyze("今天天气怎么样", "/test/project")
        # Without semantic matching (no memory engine), this should be unrelated
        assert result["confidence"] < 0.5

    def test_analyze_empty_query(self):
        """Should handle empty queries / 应处理空查询"""
        result = self.analyzer.analyze("", "/test/project")
        assert result["is_project_related"] is False
        assert result["confidence"] == 0.0

    def test_suggested_tools(self):
        """Should suggest correct tools / 应建议正确的工具"""
        result = self.analyzer.analyze("项目进度", "/test/project")
        assert "get_project_progress" in result["suggested_tools"]


class TestTriggerExecutor:
    """Test TriggerExecutor functionality / 测试触发执行器功能"""

    def test_trigger_thresholds(self):
        """Trigger thresholds should be correct / 触发阈值应正确"""
        assert TriggerExecutor.HIGH_CONFIDENCE_THRESHOLD == 0.8
        assert TriggerExecutor.MEDIUM_CONFIDENCE_THRESHOLD == 0.5
        assert TriggerExecutor.LOW_CONFIDENCE_THRESHOLD == 0.3

    def test_get_trigger_status(self):
        """Should return trigger status / 应返回触发器状态"""
        # Create executor without actual server
        executor = TriggerExecutor(mcp_server=None)
        status = executor.get_trigger_status()

        assert status["enabled"] is True
        assert status["high_confidence_threshold"] == 0.8
        assert status["medium_confidence_threshold"] == 0.5


class TestThresholdChecker:
    """Test ThresholdChecker functionality / 测试阈值检查器功能"""

    def test_threshold_levels(self):
        """Threshold levels should be correct / 阈值级别应正确"""
        assert MEMORY_THRESHOLD_WARNING == 0.7
        assert MEMORY_THRESHOLD_CLEANUP == 0.85
        assert MEMORY_THRESHOLD_CRITICAL == 0.95

        # Verify order: warning < cleanup < critical
        assert MEMORY_THRESHOLD_WARNING < MEMORY_THRESHOLD_CLEANUP
        assert MEMORY_THRESHOLD_CLEANUP < MEMORY_THRESHOLD_CRITICAL


class TestCleanupScheduler:
    """Test CleanupScheduler functionality / 测试清理调度器功能"""

    def test_cleanup_time_parsing(self):
        """Should parse cleanup time correctly / 应正确解析清理时间"""
        from hibro.core.cleanup_scheduler import CleanupScheduler

        # Mock cleaner
        class MockCleaner:
            def execute_cleanup(self, force=False):
                return {"success": True, "deleted_count": 0}

        scheduler = CleanupScheduler(
            cleaner=MockCleaner(),
            config={'cleanup_time_of_day': '03:00', 'cleanup_enabled': True}
        )

        next_time = scheduler._get_next_cleanup_time()
        assert next_time.hour == 3
        assert next_time.minute == 0

    def test_cleanup_disabled(self):
        """Should not start when disabled / 禁用时应不启动"""
        from hibro.core.cleanup_scheduler import CleanupScheduler

        class MockCleaner:
            pass

        scheduler = CleanupScheduler(
            cleaner=MockCleaner(),
            config={'cleanup_time_of_day': '03:00', 'cleanup_enabled': False}
        )

        # Should not be running
        scheduler.start()
        assert scheduler._running is False


class TestMemoryCleaner:
    """Test MemoryCleaner functionality / 测试记忆清理器功能"""

    def test_protected_categories(self):
        """Should define protected categories / 应定义保护类别"""
        from hibro.core.memory_cleaner import MemoryCleaner

        protected = MemoryCleaner.PROTECTED_CATEGORIES
        assert "active_task" in protected
        assert "project_init" in protected
        assert "preference" in protected


class TestConfigIntegration:
    """Test config integration / 测试配置集成"""

    def test_forgetting_config_defaults(self):
        """ForgettingConfig should have new fields / ForgettingConfig应有新字段"""
        from hibro.utils.config import ForgettingConfig

        config = ForgettingConfig()
        assert config.cleanup_time == "03:00"
        assert config.cleanup_enabled is True
        assert config.threshold_warning == 0.7
        assert config.threshold_cleanup == 0.85
        assert config.threshold_critical == 0.95


class TestModuleImports:
    """Test module imports / 测试模块导入"""

    def test_core_imports(self):
        """Should import core modules / 应导入核心模块"""
        from hibro.core import (
            MemoryEngine,
            LFUCalculator,
            MemoryPartition,
            CleanupScheduler,
            ThresholdChecker,
            MemoryCleaner
        )
        assert MemoryEngine is not None
        assert LFUCalculator is not None
        assert CleanupScheduler is not None
        assert ThresholdChecker is not None
        assert MemoryCleaner is not None

    def test_intelligence_imports(self):
        """Should import intelligence modules / 应导入智能模块"""
        from hibro.intelligence import (
            QueryAnalyzer,
            QueryKeywords,
            TriggerExecutor
        )
        assert QueryAnalyzer is not None
        assert QueryKeywords is not None
        assert TriggerExecutor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
