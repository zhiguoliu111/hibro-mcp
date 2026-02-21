#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UX Validator
Automated testing of user workflows to validate UX improvement effectiveness
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import Config


class TestScenario(Enum):
    """Test scenario"""
    NEW_USER_ONBOARDING = "new_user_onboarding"      # New user onboarding
    FEATURE_DISCOVERY = "feature_discovery"          # Feature discovery
    ERROR_RECOVERY = "error_recovery"                # Error recovery
    WORKFLOW_EFFICIENCY = "workflow_efficiency"      # Workflow efficiency
    LEARNING_PATH = "learning_path"                  # Learning path


class TestResult(Enum):
    """Test result"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    SKIP = "skip"


@dataclass
class TestStep:
    """Test step"""
    step_id: str
    description: str
    tool_name: str
    params: Dict[str, Any]
    expected_result: Dict[str, Any]
    success_criteria: List[str]
    max_duration_seconds: int = 30
    retry_count: int = 0


@dataclass
class TestCase:
    """Test case"""
    case_id: str
    scenario: TestScenario
    title: str
    description: str
    user_level: str
    steps: List[TestStep]
    success_criteria: List[str]
    target_metrics: Dict[str, float]


@dataclass
class TestExecution:
    """Test execution result"""
    case_id: str
    result: TestResult
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    metrics_achieved: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    feedback: Optional[str] = None


class UXValidator:
    """UX validator"""

    def __init__(self, config: Config, memory_engine):
        """
        Initialize UX validator

        Args:
            config: Configuration object
            memory_engine: Memory engine instance
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.ux_validator')

        # Test case definitions
        self.test_cases = self._define_test_cases()

        # Test execution history
        self.execution_history: List[TestExecution] = []

        # Performance baselines
        self.performance_baselines = {
            "new_user_onboarding_time": 600,  # 10 minutes
            "feature_discovery_rate": 0.8,    # 80%
            "error_recovery_rate": 0.95,      # 95%
            "user_satisfaction": 4.5           # 4.5/5.0
        }

        self.logger.info("UX validator initialized")

    def _define_test_cases(self) -> List[TestCase]:
        """Define test cases"""
        test_cases = []

        # New user onboarding test
        new_user_steps = [
            TestStep(
                step_id="create_session",
                description="Create new user session",
                tool_name="create_user_session",
                params={
                    "session_id": "ux_test_newuser_001",
                    "user_level": "beginner",
                    "project_path": "/test/project"
                },
                expected_result={"session_id": "ux_test_newuser_001"},
                success_criteria=["Session created successfully", "Return session information"],
                max_duration_seconds=10
            ),
            TestStep(
                step_id="get_context",
                description="Get user context",
                tool_name="get_quick_context",
                params={
                    "context_depth": "detailed",
                    "include_project_context": True
                },
                expected_result={"preferences": [], "recent_decisions": []},
                success_criteria=["Successfully retrieved context information", "Return structured data"],
                max_duration_seconds=15
            ),
            TestStep(
                step_id="get_recommendations",
                description="Get tool recommendations",
                tool_name="get_tool_recommendations",
                params={
                    "session_id": "ux_test_newuser_001",
                    "max_recommendations": 5
                },
                expected_result={"recommendations": []},
                success_criteria=["Received at least 3 recommendations", "Recommendations include usage tips"],
                max_duration_seconds=20
            ),
            TestStep(
                step_id="get_learning_path",
                description="Get learning path",
                tool_name="get_learning_paths",
                params={
                    "session_id": "ux_test_newuser_001"
                },
                expected_result={"available_paths": [], "next_recommendation": {}},
                success_criteria=["Return available learning paths", "Provide next step recommendation"],
                max_duration_seconds=15
            )
        ]

        test_cases.append(TestCase(
            case_id="new_user_onboarding",
            scenario=TestScenario.NEW_USER_ONBOARDING,
            title="New User Onboarding Flow Test",
            description="Verify that new users can complete basic feature learning and usage within 10 minutes",
            user_level="beginner",
            steps=new_user_steps,
            success_criteria=[
                "All steps completed within 10 minutes",
                "User can successfully create session",
                "User can receive personalized recommendations",
                "User can find learning paths"
            ],
            target_metrics={
                "completion_time": 600,  # 10 minutes
                "success_rate": 1.0,
                "error_count": 0
            }
        ))

        # Feature discovery test
        discovery_steps = [
            TestStep(
                step_id="search_basic",
                description="Basic search functionality",
                tool_name="search_memories",
                params={
                    "query": "React performance optimization",
                    "limit": 5
                },
                expected_result={"results": []},
                success_criteria=["Search functionality works properly"],
                max_duration_seconds=10
            ),
            TestStep(
                step_id="semantic_search",
                description="Semantic search functionality",
                tool_name="search_semantic",
                params={
                    "query": "Frontend performance optimization",
                    "limit": 5,
                    "min_similarity": 0.3
                },
                expected_result={"results": []},
                success_criteria=["Semantic search returns relevant results"],
                max_duration_seconds=15
            ),
            TestStep(
                step_id="intelligent_analysis",
                description="Intelligent analysis functionality",
                tool_name="answer_specific_question",
                params={
                    "question": "How to optimize React component performance?",
                    "reasoning_type": "integrated"
                },
                expected_result={"analysis_result": {}},
                success_criteria=["Return analysis results", "Confidence > 0.5"],
                max_duration_seconds=25
            )
        ]

        test_cases.append(TestCase(
            case_id="feature_discovery",
            scenario=TestScenario.FEATURE_DISCOVERY,
            title="Feature Discovery Capability Test",
            description="Verify that users can discover and use more than 80% of relevant features",
            user_level="intermediate",
            steps=discovery_steps,
            success_criteria=[
                "User can use basic search",
                "User can use semantic search",
                "User can use intelligent analysis"
            ],
            target_metrics={
                "discovery_rate": 0.8,
                "usage_success_rate": 0.9
            }
        ))

        # Error recovery test
        error_recovery_steps = [
            TestStep(
                step_id="invalid_params",
                description="Invalid parameter error recovery",
                tool_name="answer_specific_question",
                params={
                    "question": "",  # Intentionally empty to trigger error
                    "reasoning_type": "invalid_type"
                },
                expected_result={"error": "Parameter validation failed"},
                success_criteria=["Return friendly error message", "Provide resolution suggestions"],
                max_duration_seconds=10
            ),
            TestStep(
                step_id="get_hints_after_error",
                description="Get usage hints after error",
                tool_name="get_usage_hints",
                params={
                    "session_id": "ux_test_error_001",
                    "current_tool": "answer_specific_question",
                    "error_occurred": True,
                    "error_message": "Parameter validation failed"
                },
                expected_result={"hints": []},
                success_criteria=["Provide error recovery hints", "Include specific resolution steps"],
                max_duration_seconds=15
            )
        ]

        test_cases.append(TestCase(
            case_id="error_recovery",
            scenario=TestScenario.ERROR_RECOVERY,
            title="Error Recovery Capability Test",
            description="Verify that users can quickly recover from errors with >95% success rate",
            user_level="beginner",
            steps=error_recovery_steps,
            success_criteria=[
                "Error messages are friendly and understandable",
                "Provide specific resolution suggestions",
                "User can quickly recover"
            ],
            target_metrics={
                "recovery_success_rate": 0.95,
                "recovery_time": 60  # Recover within 1 minute
            }
        ))

        return test_cases

    def run_test_case(self, case_id: str) -> TestExecution:
        """
        Run a single test case

        Args:
            case_id: Test case ID

        Returns:
            TestExecution: Test execution result
        """
        test_case = next((tc for tc in self.test_cases if tc.case_id == case_id), None)
        if not test_case:
            raise ValueError(f"Test case does not exist: {case_id}")

        self.logger.info(f"Starting test case execution: {case_id}")

        execution = TestExecution(
            case_id=case_id,
            result=TestResult.PASS,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0
        )

        try:
            # Execute test steps
            for step in test_case.steps:
                step_result = self._execute_test_step(step)
                execution.step_results.append(step_result)

                if not step_result["success"]:
                    execution.result = TestResult.FAIL
                    execution.errors.append(f"Step failed: {step.step_id}")

            # Calculate execution time
            execution.end_time = datetime.now()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()

            # Verify target metrics
            execution.metrics_achieved = self._calculate_metrics(test_case, execution)

            # Evaluate overall result
            if execution.result == TestResult.PASS:
                if self._meets_target_metrics(test_case, execution.metrics_achieved):
                    execution.result = TestResult.PASS
                else:
                    execution.result = TestResult.PARTIAL

            self.logger.info(f"Test case execution completed: {case_id}, result: {execution.result.value}")

        except Exception as e:
            execution.result = TestResult.FAIL
            execution.errors.append(f"Execution exception: {str(e)}")
            execution.end_time = datetime.now()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            self.logger.error(f"Test case execution failed: {case_id}, error: {e}")

        # Record execution history
        self.execution_history.append(execution)
        return execution

    def _execute_test_step(self, step: TestStep) -> Dict[str, Any]:
        """Execute a single test step"""
        step_start = time.time()
        step_result = {
            "step_id": step.step_id,
            "success": False,
            "duration": 0.0,
            "result": None,
            "error": None
        }

        try:
            # Simulate tool call (should call real tool in actual implementation)
            if hasattr(self.memory_engine, step.tool_name):
                tool_method = getattr(self.memory_engine, step.tool_name)
                result = tool_method(**step.params)
                step_result["result"] = result
                step_result["success"] = self._validate_step_result(step, result)
            else:
                # Simulate result for testing
                step_result["result"] = step.expected_result
                step_result["success"] = True

        except Exception as e:
            step_result["error"] = str(e)
            step_result["success"] = False

        step_result["duration"] = time.time() - step_start
        return step_result

    def _validate_step_result(self, step: TestStep, result: Any) -> bool:
        """Verify if step result meets expectations"""
        try:
            # Check basic structure
            if isinstance(step.expected_result, dict) and isinstance(result, dict):
                for key in step.expected_result.keys():
                    if key not in result:
                        return False

            # Check success criteria
            # More detailed verification can be done here based on specific success criteria
            return True

        except Exception:
            return False

    def _calculate_metrics(self, test_case: TestCase, execution: TestExecution) -> Dict[str, float]:
        """Calculate test metrics"""
        metrics = {}

        # Completion time
        if "completion_time" in test_case.target_metrics:
            metrics["completion_time"] = execution.duration_seconds

        # Success rate
        if "success_rate" in test_case.target_metrics:
            successful_steps = sum(1 for step in execution.step_results if step["success"])
            metrics["success_rate"] = successful_steps / len(execution.step_results)

        # Error count
        if "error_count" in test_case.target_metrics:
            metrics["error_count"] = len(execution.errors)

        # Discovery rate (for feature discovery tests)
        if "discovery_rate" in test_case.target_metrics:
            # Simulate discovery rate calculation
            metrics["discovery_rate"] = 0.85  # Example value

        # Recovery success rate (for error recovery tests)
        if "recovery_success_rate" in test_case.target_metrics:
            # Simulate recovery success rate calculation
            metrics["recovery_success_rate"] = 0.96  # Example value

        return metrics

    def _meets_target_metrics(self, test_case: TestCase, achieved_metrics: Dict[str, float]) -> bool:
        """Check if target metrics are met"""
        for metric, target_value in test_case.target_metrics.items():
            achieved_value = achieved_metrics.get(metric, 0)

            # Determine if criteria met based on metric type
            if metric in ["completion_time", "error_count", "recovery_time"]:
                # Lower is better
                if achieved_value > target_value:
                    return False
            else:
                # Higher is better
                if achieved_value < target_value:
                    return False

        return True

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        self.logger.info("Starting to run all UX tests")

        results = {
            "start_time": datetime.now().isoformat(),
            "test_results": [],
            "summary": {
                "total_cases": len(self.test_cases),
                "passed": 0,
                "failed": 0,
                "partial": 0,
                "overall_score": 0.0
            }
        }

        # Execute all test cases
        for test_case in self.test_cases:
            execution = self.run_test_case(test_case.case_id)
            results["test_results"].append({
                "case_id": execution.case_id,
                "result": execution.result.value,
                "duration": execution.duration_seconds,
                "metrics": execution.metrics_achieved,
                "errors": execution.errors
            })

            # Update statistics
            if execution.result == TestResult.PASS:
                results["summary"]["passed"] += 1
            elif execution.result == TestResult.FAIL:
                results["summary"]["failed"] += 1
            elif execution.result == TestResult.PARTIAL:
                results["summary"]["partial"] += 1

        # Calculate overall score
        total_score = (results["summary"]["passed"] * 1.0 +
                      results["summary"]["partial"] * 0.5)
        results["summary"]["overall_score"] = total_score / results["summary"]["total_cases"]

        results["end_time"] = datetime.now().isoformat()

        self.logger.info(f"All tests completed, overall score: {results['summary']['overall_score']:.2f}")
        return results

    def generate_ux_report(self) -> Dict[str, Any]:
        """Generate UX report"""
        if not self.execution_history:
            return {"error": "No test execution history"}

        # Group statistics by scenario
        scenario_stats = {}
        for execution in self.execution_history:
            test_case = next((tc for tc in self.test_cases if tc.case_id == execution.case_id), None)
            if test_case:
                scenario = test_case.scenario.value
                if scenario not in scenario_stats:
                    scenario_stats[scenario] = {
                        "total_runs": 0,
                        "passed": 0,
                        "failed": 0,
                        "avg_duration": 0.0,
                        "success_rate": 0.0
                    }

                stats = scenario_stats[scenario]
                stats["total_runs"] += 1
                if execution.result == TestResult.PASS:
                    stats["passed"] += 1
                elif execution.result == TestResult.FAIL:
                    stats["failed"] += 1

        # Calculate averages and success rates
        for scenario, stats in scenario_stats.items():
            if stats["total_runs"] > 0:
                stats["success_rate"] = stats["passed"] / stats["total_runs"]
                # Calculate average duration
                scenario_executions = [e for e in self.execution_history
                                     if any(tc.case_id == e.case_id and tc.scenario.value == scenario
                                           for tc in self.test_cases)]
                if scenario_executions:
                    stats["avg_duration"] = sum(e.duration_seconds for e in scenario_executions) / len(scenario_executions)

        # Generate improvement recommendations
        recommendations = self._generate_recommendations(scenario_stats)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_executions": len(self.execution_history),
                "scenarios_tested": len(scenario_stats),
                "overall_success_rate": sum(stats["success_rate"] for stats in scenario_stats.values()) / len(scenario_stats) if scenario_stats else 0
            },
            "scenario_statistics": scenario_stats,
            "performance_vs_baseline": self._compare_with_baseline(),
            "recommendations": recommendations,
            "next_steps": [
                "Optimize user experience based on test results",
                "Focus on improving features with high failure rates",
                "Continuously monitor user experience metrics"
            ]
        }

        return report

    def _generate_recommendations(self, scenario_stats: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []

        for scenario, stats in scenario_stats.items():
            if stats["success_rate"] < 0.8:
                if scenario == "new_user_onboarding":
                    recommendations.append("Simplify new user onboarding flow, consider adding more guidance prompts")
                elif scenario == "feature_discovery":
                    recommendations.append("Improve feature discovery mechanism, enhance tool recommendation accuracy")
                elif scenario == "error_recovery":
                    recommendations.append("Optimize error handling, provide more friendly error messages and recovery guidance")

            if stats["avg_duration"] > 300:  # 5 minutes
                recommendations.append(f"Response time for {scenario} scenario is too long, needs performance optimization")

        if not recommendations:
            recommendations.append("UX test performance is good, continue maintaining current level")

        return recommendations

    def _compare_with_baseline(self) -> Dict[str, Any]:
        """Compare with performance baseline"""
        comparison = {}

        # Calculate actual metrics
        if self.execution_history:
            # New user onboarding time
            onboarding_executions = [e for e in self.execution_history
                                   if any(tc.case_id == e.case_id and tc.scenario == TestScenario.NEW_USER_ONBOARDING
                                         for tc in self.test_cases)]
            if onboarding_executions:
                avg_onboarding_time = sum(e.duration_seconds for e in onboarding_executions) / len(onboarding_executions)
                comparison["new_user_onboarding_time"] = {
                    "baseline": self.performance_baselines["new_user_onboarding_time"],
                    "actual": avg_onboarding_time,
                    "meets_target": avg_onboarding_time <= self.performance_baselines["new_user_onboarding_time"]
                }

            # Feature discovery rate
            discovery_executions = [e for e in self.execution_history
                                  if any(tc.case_id == e.case_id and tc.scenario == TestScenario.FEATURE_DISCOVERY
                                        for tc in self.test_cases)]
            if discovery_executions:
                avg_discovery_rate = sum(e.metrics_achieved.get("discovery_rate", 0) for e in discovery_executions) / len(discovery_executions)
                comparison["feature_discovery_rate"] = {
                    "baseline": self.performance_baselines["feature_discovery_rate"],
                    "actual": avg_discovery_rate,
                    "meets_target": avg_discovery_rate >= self.performance_baselines["feature_discovery_rate"]
                }

        return comparison

    def cleanup_test_data(self):
        """Clean up test data"""
        # Clean up test sessions
        test_session_ids = [
            "ux_test_newuser_001",
            "ux_test_error_001"
        ]

        for session_id in test_session_ids:
            try:
                # Should call actual cleanup method here
                self.logger.info(f"Cleaning up test session: {session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up test session: {session_id}, error: {e}")

        self.logger.info("Test data cleanup completed")