#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UX Test Runner
Execute automated UX tests and collect test results and performance data
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .ux_validator import UXValidator, TestScenario, TestResult
from .feedback_collector import FeedbackCollector, UserFeedback, FeedbackType, FeedbackPriority
from ..utils.config import Config


@dataclass
class TestSession:
    """Test session"""
    session_id: str
    user_level: str
    test_scenarios: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[TestResult] = field(default_factory=list)
    feedback: List[UserFeedback] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class UXTestReport:
    """UX test report"""
    report_id: str
    test_sessions: List[TestSession]
    overall_metrics: Dict[str, float]
    success_criteria_met: Dict[str, bool]
    improvement_recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)


class UXTestRunner:
    """UX test runner"""

    def __init__(self, config: Config, memory_engine=None):
        """
        Initialize test runner

        Args:
            config: Configuration object
            memory_engine: Memory engine object
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.ux_test_runner')

        # Initialize components
        self.validator = UXValidator(config, memory_engine) if memory_engine else None
        self.feedback_collector = FeedbackCollector(config)

        # Test session storage
        self.test_sessions: List[TestSession] = []
        self.test_reports: List[UXTestReport] = []

        # Success criteria definition
        self.success_criteria = {
            "new_user_onboarding_time": 600,  # 10 minutes (seconds)
            "feature_discovery_rate": 0.8,      # 80%
            "user_satisfaction": 4.5,      # 4.5/5.0
            "error_recovery_rate": 0.95   # 95%
        }

        self.logger.info("UX test runner initialized")

    def run_comprehensive_test(self, user_levels: List[str] = None) -> UXTestReport:
        """
        Run comprehensive UX test

        Args:
            user_levels: List of user levels to test

        Returns:
            Test report
        """
        if user_levels is None:
            user_levels = ["beginner", "intermediate", "advanced", "expert"]

        report_id = f"ux_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_sessions = []

        self.logger.info(f"Starting comprehensive UX test: {report_id}")

        for user_level in user_levels:
            # Run test for each user level
            session = self._run_user_level_test(user_level)
            test_sessions.append(session)

        # Generate comprehensive report
        report = self._generate_comprehensive_report(report_id, test_sessions)
        self.test_reports.append(report)

        self.logger.info(f"Comprehensive UX test completed: {report_id}")
        return report

    def _run_user_level_test(self, user_level: str) -> TestSession:
        """Run test for specific user level"""
        session_id = f"test_{user_level}_{datetime.now().strftime('%H%M%S')}"

        # Define test scenarios
        test_scenarios = self._get_test_scenarios_for_level(user_level)

        session = TestSession(
            session_id=session_id,
            user_level=user_level,
            test_scenarios=test_scenarios,
            start_time=datetime.now()
        )

        self.logger.info(f"Starting {user_level} level user test: {session_id}")

        # Execute each test scenario
        for scenario_name in test_scenarios:
            result = self._execute_test_scenario(session_id, scenario_name, user_level)
            session.results.append(result)

            # Generate feedback based on test results
            feedback = self._generate_test_feedback(result, user_level)
            if feedback:
                session.feedback.append(feedback)

        # Calculate performance metrics
        session.performance_metrics = self._calculate_session_metrics(session)
        session.end_time = datetime.now()

        self.test_sessions.append(session)
        return session

    def _get_test_scenarios_for_level(self, user_level: str) -> List[str]:
        """Get test scenarios for user level"""
        base_scenarios = ["onboarding", "basic_usage", "error_recovery"]

        if user_level in ["intermediate", "advanced", "expert"]:
            base_scenarios.extend(["feature_discovery", "workflow_efficiency"])

        if user_level in ["advanced", "expert"]:
            base_scenarios.extend(["advanced_features", "system_management"])

        return base_scenarios

    def _execute_test_scenario(self, session_id: str, scenario_name: str, user_level: str) -> TestResult:
        """Execute single test scenario"""
        start_time = time.time()

        try:
            # Get test scenario
            scenario = self._create_test_scenario(scenario_name, user_level)

            # Execute validation
            result = self.validator.validate_scenario(scenario)

            # Record execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            self.logger.info(f"Test scenario {scenario_name} completed: {result.success}, duration: {execution_time:.2f}s")

        except Exception as e:
            # Test execution failed
            result = TestResult(
                scenario_id=scenario_name,
                success=False,
                score=0.0,
                issues=[f"Test execution exception: {str(e)}"],
                recommendations=[f"Check {scenario_name} test scenario configuration"],
                execution_time=time.time() - start_time
            )
            self.logger.error(f"Test scenario {scenario_name} execution failed: {e}")

        return result

    def _create_test_scenario(self, scenario_name: str, user_level: str) -> TestScenario:
        """Create test scenario"""
        scenario_configs = {
            "onboarding": {
                "name": "New User Onboarding Test",
                "description": f"Test onboarding flow for {user_level} level users",
                "steps": [
                    "Create user session",
                    "Get quick context",
                    "View tool recommendations",
                    "Execute first tool call",
                    "Get usage hints"
                ],
                "expected_duration": 600,  # 10 minutes
                "success_criteria": ["Session created successfully", "Received relevant recommendations", "Tool call successful"]
            },
            "basic_usage": {
                "name": "Basic Usage Test",
                "description": f"Test basic feature usage for {user_level} level users",
                "steps": [
                    "Memory storage and retrieval",
                    "Simple Q&A",
                    "Preference settings and retrieval",
                    "Search functionality usage"
                ],
                "expected_duration": 300,  # 5 minutes
                "success_criteria": ["Features executed normally", "Results meet expectations", "Response time reasonable"]
            },
            "feature_discovery": {
                "name": "Feature Discovery Test",
                "description": f"Test feature discovery capability for {user_level} level users",
                "steps": [
                    "Get intelligent suggestions",
                    "Explore tool recommendations",
                    "View learning paths",
                    "Discover advanced features"
                ],
                "expected_duration": 450,  # 7.5 minutes
                "success_criteria": ["Relevant features discovered", "High recommendation accuracy", "Clear learning path"]
            },
            "error_recovery": {
                "name": "Error Recovery Test",
                "description": f"Test error handling and recovery for {user_level} level users",
                "steps": [
                    "Trigger parameter error",
                    "Handle tool call failure",
                    "Recover session state",
                    "Get error help"
                ],
                "expected_duration": 240,  # 4 minutes
                "success_criteria": ["Clear error messages", "Clear recovery steps", "Successful recovery"]
            },
            "workflow_efficiency": {
                "name": "Workflow Efficiency Test",
                "description": f"Test workflow efficiency for {user_level} level users",
                "steps": [
                    "Execute composite tasks",
                    "Use tool chains",
                    "Batch operations",
                    "Performance optimization"
                ],
                "expected_duration": 600,  # 10 minutes
                "success_criteria": ["Task completion efficiency", "Smooth tool chain", "Good performance"]
            },
            "advanced_features": {
                "name": "Advanced Features Test",
                "description": f"Test advanced feature usage for {user_level} level users",
                "steps": [
                    "Deep project analysis",
                    "Knowledge graph construction",
                    "Complex reasoning tasks",
                    "System integration features"
                ],
                "expected_duration": 900,  # 15 minutes
                "success_criteria": ["Advanced features available", "Accurate analysis results", "Integration working"]
            },
            "system_management": {
                "name": "System Management Test",
                "description": f"Test system management capability for {user_level} level users",
                "steps": [
                    "Security status check",
                    "Backup and recovery",
                    "Performance monitoring",
                    "System maintenance"
                ],
                "expected_duration": 720,  # 12 minutes
                "success_criteria": ["Management features available", "Security checks passed", "Backup and recovery working"]
            }
        }

        config = scenario_configs.get(scenario_name, scenario_configs["basic_usage"])

        return TestScenario(
            scenario_id=scenario_name,
            name=config["name"],
            description=config["description"],
            steps=config["steps"],
            expected_duration=config["expected_duration"],
            success_criteria=config["success_criteria"],
            user_level=user_level
        )

    def _generate_test_feedback(self, result: TestResult, user_level: str) -> Optional[UserFeedback]:
        """Generate feedback based on test results"""
        if not result.issues:
            return None

        # Determine feedback type
        feedback_type = FeedbackType.USABILITY_ISSUE
        if any("error" in issue or "exception" in issue for issue in result.issues):
            feedback_type = FeedbackType.BUG_REPORT
        elif any("performance" in issue or "slow" in issue for issue in result.issues):
            feedback_type = FeedbackType.PERFORMANCE_ISSUE

        # Generate feedback content
        title = f"{result.scenario_id} test discovered issues"
        description = f"User level: {user_level}\n"
        description += f"Test score: {result.score:.2f}\n"
        description += f"Execution time: {result.execution_time:.2f}s\n"
        description += f"Issues found:\n" + "\n".join(f"- {issue}" for issue in result.issues)

        if result.recommendations:
            description += f"\nRecommended improvements:\n" + "\n".join(f"- {rec}" for rec in result.recommendations)

        return self.feedback_collector.collect_feedback(
            user_session=f"test_{user_level}",
            feedback_type=feedback_type.value,
            title=title,
            description=description,
            tool_name=result.scenario_id,
            user_context={
                "test_scenario": result.scenario_id,
                "user_level": user_level,
                "test_score": result.score,
                "execution_time": result.execution_time
            }
        )

    def _calculate_session_metrics(self, session: TestSession) -> Dict[str, float]:
        """Calculate session performance metrics"""
        if not session.results:
            return {}

        # Basic metrics
        total_tests = len(session.results)
        successful_tests = sum(1 for r in session.results if r.success)
        average_score = sum(r.score for r in session.results) / total_tests
        total_time = sum(r.execution_time for r in session.results)
        average_time = total_time / total_tests

        # Specific metric calculations
        onboarding_time = next(
            (r.execution_time for r in session.results if r.scenario_id == "onboarding"),
            0
        )

        # Feature discovery rate (based on success rate of recommendation and discovery tests)
        discovery_tests = [r for r in session.results if "discovery" in r.scenario_id or "recommendation" in r.scenario_id]
        feature_discovery_rate = (
            sum(r.score for r in discovery_tests) / len(discovery_tests)
            if discovery_tests else 0.8  # Default value
        )

        # Error recovery success rate
        error_recovery_tests = [r for r in session.results if "error" in r.scenario_id or "recovery" in r.scenario_id]
        error_recovery_rate = (
            sum(1 for r in error_recovery_tests if r.success) / len(error_recovery_tests)
            if error_recovery_tests else 0.95  # Default value
        )

        # User satisfaction (based on average score)
        user_satisfaction = min(average_score * 5, 5.0)  # Convert to 5-point scale

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests,
            "average_score": average_score,
            "total_execution_time": total_time,
            "average_execution_time": average_time,
            "new_user_onboarding_time": onboarding_time,
            "feature_discovery_rate": feature_discovery_rate,
            "user_satisfaction": user_satisfaction,
            "error_recovery_rate": error_recovery_rate
        }

    def _generate_comprehensive_report(self, report_id: str, test_sessions: List[TestSession]) -> UXTestReport:
        """Generate comprehensive test report"""
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(test_sessions)

        # Check success criteria
        success_criteria_met = self._check_success_criteria(overall_metrics)

        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            test_sessions, overall_metrics, success_criteria_met
        )

        return UXTestReport(
            report_id=report_id,
            test_sessions=test_sessions,
            overall_metrics=overall_metrics,
            success_criteria_met=success_criteria_met,
            improvement_recommendations=improvement_recommendations
        )

    def _calculate_overall_metrics(self, test_sessions: List[TestSession]) -> Dict[str, float]:
        """Calculate overall metrics"""
        if not test_sessions:
            return {}

        # Aggregate metrics from all sessions
        all_metrics = {}
        metric_counts = {}

        for session in test_sessions:
            for metric, value in session.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = 0
                    metric_counts[metric] = 0
                all_metrics[metric] += value
                metric_counts[metric] += 1

        # Calculate averages
        overall_metrics = {}
        for metric, total in all_metrics.items():
            overall_metrics[metric] = total / metric_counts[metric]

        return overall_metrics

    def _check_success_criteria(self, overall_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check success criteria"""
        criteria_met = {}

        # New user onboarding time < 10 minutes
        onboarding_time = overall_metrics.get("new_user_onboarding_time", float('inf'))
        criteria_met["new_user_onboarding_time"] = onboarding_time <= self.success_criteria["new_user_onboarding_time"]

        # Feature discovery rate > 80%
        discovery_rate = overall_metrics.get("feature_discovery_rate", 0)
        criteria_met["feature_discovery_rate"] = discovery_rate >= self.success_criteria["feature_discovery_rate"]

        # User satisfaction > 4.5/5.0
        satisfaction = overall_metrics.get("user_satisfaction", 0)
        criteria_met["user_satisfaction"] = satisfaction >= self.success_criteria["user_satisfaction"]

        # Error recovery rate > 95%
        recovery_rate = overall_metrics.get("error_recovery_rate", 0)
        criteria_met["error_recovery_rate"] = recovery_rate >= self.success_criteria["error_recovery_rate"]

        return criteria_met

    def _generate_improvement_recommendations(self, test_sessions: List[TestSession],
                                           overall_metrics: Dict[str, float],
                                           success_criteria_met: Dict[str, bool]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Generate recommendations based on unmet success criteria
        for criterion, met in success_criteria_met.items():
            if not met:
                if criterion == "new_user_onboarding_time":
                    recommendations.append("Optimize new user onboarding flow, simplify initial setup steps")
                    recommendations.append("Add quick start wizard with more intuitive operation guidance")
                elif criterion == "feature_discovery_rate":
                    recommendations.append("Improve tool recommendation algorithm, increase recommendation accuracy")
                    recommendations.append("Enhance feature discovery interface, make related features easier to discover")
                elif criterion == "user_satisfaction":
                    recommendations.append("Analyze user feedback, focus on improving low-scoring functional modules")
                    recommendations.append("Optimize user interface design, improve overall user experience")
                elif criterion == "error_recovery_rate":
                    recommendations.append("Improve error handling mechanism, provide clearer error messages")
                    recommendations.append("Add auto-recovery functionality, reduce manual user intervention")

        # Generate recommendations based on test result analysis
        all_feedback = []
        for session in test_sessions:
            all_feedback.extend(session.feedback)

        if all_feedback:
            # Analyze feedback patterns
            feedback_analysis = self.feedback_collector.analyze_feedback_patterns()
            recommendations.extend(feedback_analysis.improvement_suggestions[:5])

        # Generate recommendations based on performance metrics
        avg_response_time = overall_metrics.get("average_execution_time", 0)
        if avg_response_time > 3.0:  # If average response time exceeds 3 seconds
            recommendations.append("Optimize system performance, reduce tool call response time")

        return recommendations[:10]  # Return top 10 recommendations

    def export_test_report(self, report_id: str, format: str = "json") -> Dict[str, Any]:
        """
        Export test report

        Args:
            report_id: Report ID
            format: Export format

        Returns:
            Exported report data
        """
        report = next((r for r in self.test_reports if r.report_id == report_id), None)
        if not report:
            return {}

        export_data = {
            "report_id": report.report_id,
            "created_at": report.created_at.isoformat(),
            "overall_metrics": report.overall_metrics,
            "success_criteria_met": report.success_criteria_met,
            "improvement_recommendations": report.improvement_recommendations,
            "test_sessions": []
        }

        # Export test session data
        for session in report.test_sessions:
            session_data = {
                "session_id": session.session_id,
                "user_level": session.user_level,
                "test_scenarios": session.test_scenarios,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "performance_metrics": session.performance_metrics,
                "results": [],
                "feedback": []
            }

            # Export test results
            for result in session.results:
                session_data["results"].append({
                    "scenario_id": result.scenario_id,
                    "success": result.success,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "issues": result.issues,
                    "recommendations": result.recommendations
                })

            # Export feedback data
            for feedback in session.feedback:
                session_data["feedback"].append({
                    "feedback_id": feedback.feedback_id,
                    "feedback_type": feedback.feedback_type.value,
                    "priority": feedback.priority.value,
                    "title": feedback.title,
                    "description": feedback.description,
                    "created_at": feedback.created_at.isoformat()
                })

            export_data["test_sessions"].append(session_data)

        return export_data

    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        if not self.test_reports:
            return {"message": "No test reports available"}

        latest_report = self.test_reports[-1]

        return {
            "latest_report_id": latest_report.report_id,
            "test_time": latest_report.created_at.isoformat(),
            "test_sessions_count": len(latest_report.test_sessions),
            "overall_metrics": latest_report.overall_metrics,
            "success_criteria_met": latest_report.success_criteria_met,
            "improvement_recommendations_count": len(latest_report.improvement_recommendations),
            "historical_reports_count": len(self.test_reports)
        }