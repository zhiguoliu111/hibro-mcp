#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Automator
Identify user repetitive work patterns and provide automation solutions
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from pathlib import Path

from ..storage import Memory, MemoryRepository


class PatternType(Enum):
    """Pattern type"""
    COMMAND_SEQUENCE = "command_sequence"       # Command sequence pattern
    PROJECT_INIT = "project_init"              # Project initialization pattern
    CONFIG_GENERATION = "config_generation"     # Configuration generation pattern
    FILE_OPERATION = "file_operation"          # File operation pattern
    DEVELOPMENT_WORKFLOW = "development_workflow"  # Development workflow pattern


class AutomationLevel(Enum):
    """Automation level"""
    MANUAL = "manual"           # Manual execution
    SEMI_AUTO = "semi_auto"     # Semi-automatic (requires confirmation)
    FULL_AUTO = "full_auto"     # Fully automatic execution


class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"         # Pending execution
    RUNNING = "running"         # Running
    SUCCESS = "success"         # Execution successful
    FAILED = "failed"          # Execution failed
    CANCELLED = "cancelled"     # Cancelled


@dataclass
class Activity:
    """User activity record"""
    timestamp: datetime
    action_type: str            # 'command', 'query', 'file_edit', 'project_create'
    content: str               # Activity content
    context: Dict[str, Any] = field(default_factory=dict)
    project_path: Optional[str] = None
    success: bool = True


@dataclass
class Pattern:
    """Repeated pattern"""
    pattern_type: PatternType
    name: str
    description: str
    activities: List[Activity]
    frequency: int              # Occurrence frequency
    confidence: float           # Pattern confidence
    last_occurrence: datetime
    context_similarity: float   # Context similarity
    time_pattern: Optional[str] = None  # Time pattern (e.g., daily, weekly)


@dataclass
class WorkflowStep:
    """Workflow step"""
    step_id: str
    name: str
    description: str
    action_type: str           # 'command', 'file_create', 'file_edit', 'api_call'
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)  # Execution conditions
    retry_count: int = 0
    timeout_seconds: int = 30


@dataclass
class WorkflowTemplate:
    """Workflow template"""
    template_id: str
    name: str
    description: str
    category: str              # 'project_init', 'development', 'deployment'
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)  # Template variables
    automation_level: AutomationLevel = AutomationLevel.SEMI_AUTO
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0


@dataclass
class ExecutionResult:
    """Execution result"""
    execution_id: str
    template_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)


class WorkflowAutomator:
    """Workflow automator"""

    def __init__(self, memory_repo: MemoryRepository):
        """
        Initialize workflow automator

        Args:
            memory_repo: Memory data repository
        """
        self.memory_repo = memory_repo
        self.logger = logging.getLogger('hibro.workflow_automator')

        # Activity history cache
        self.activity_history = []
        self.detected_patterns = []
        self.workflow_templates = {}
        self.execution_history = []

        # Initialize built-in templates
        self._initialize_builtin_templates()

        # Pattern detection configuration
        self.pattern_config = {
            'min_frequency': 3,          # Minimum repeat frequency
            'min_confidence': 0.7,       # Minimum confidence
            'time_window_days': 30,      # Time window
            'similarity_threshold': 0.8   # Similarity threshold
        }

    def detect_repetitive_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """
        Detect repetitive patterns

        Args:
            activities: List of user activities

        Returns:
            List of detected patterns
        """
        try:
            patterns = []

            # Detect command sequence patterns
            command_patterns = self._detect_command_sequence_patterns(activities)
            patterns.extend(command_patterns)

            # Detect project initialization patterns
            project_init_patterns = self._detect_project_init_patterns(activities)
            patterns.extend(project_init_patterns)

            # Detect configuration generation patterns
            config_patterns = self._detect_config_generation_patterns(activities)
            patterns.extend(config_patterns)

            # Detect file operation patterns
            file_operation_patterns = self._detect_file_operation_patterns(activities)
            patterns.extend(file_operation_patterns)

            # Detect development workflow patterns
            workflow_patterns = self._detect_development_workflow_patterns(activities)
            patterns.extend(workflow_patterns)

            # Filter and rank patterns
            filtered_patterns = self._filter_and_rank_patterns(patterns)

            self.detected_patterns = filtered_patterns
            self.logger.info(f"Pattern detection completed: found {len(filtered_patterns)} repetitive patterns")

            return filtered_patterns

        except Exception as e:
            self.logger.error(f"Repetitive pattern detection failed: {e}")
            return []

    def create_workflow_templates(self, patterns: List[Pattern]) -> List[WorkflowTemplate]:
        """
        Create workflow templates based on patterns

        Args:
            patterns: List of detected patterns

        Returns:
            List of created workflow templates
        """
        try:
            templates = []

            for pattern in patterns:
                template = self._pattern_to_template(pattern)
                if template:
                    templates.append(template)
                    self.workflow_templates[template.template_id] = template

            self.logger.info(f"Workflow template creation completed: generated {len(templates)} templates")
            return templates

        except Exception as e:
            self.logger.error(f"Workflow template creation failed: {e}")
            return []

    def execute_automated_workflow(self, template: WorkflowTemplate,
                                 variables: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute automated workflow

        Args:
            template: Workflow template
            variables: Execution variables

        Returns:
            Execution result
        """
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{template.template_id}"

        result = ExecutionResult(
            execution_id=execution_id,
            template_id=template.template_id,
            status=ExecutionStatus.PENDING,
            started_at=datetime.now()
        )

        try:
            self.logger.info(f"Starting workflow execution: {template.name}")
            result.status = ExecutionStatus.RUNNING

            # Merge variables
            execution_vars = {**template.variables}
            if variables:
                execution_vars.update(variables)

            # Execute workflow step by step
            for i, step in enumerate(template.steps):
                step_result = self._execute_workflow_step(step, execution_vars)
                result.step_results.append(step_result)

                if not step_result.get('success', False):
                    result.status = ExecutionStatus.FAILED
                    result.error_message = step_result.get('error', 'Unknown error')
                    break

                # Update execution variables
                if 'output_vars' in step_result:
                    execution_vars.update(step_result['output_vars'])

            # Set final status
            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.SUCCESS

            result.completed_at = datetime.now()
            result.output_data = execution_vars

            # Update template usage statistics
            template.usage_count += 1

            # Record execution history
            self.execution_history.append(result)

            self.logger.info(f"Workflow execution completed: {template.name}, status: {result.status.value}")
            return result

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()

            self.logger.error(f"Workflow execution failed: {e}")
            return result

    def get_workflow_recommendations(self, context: Dict[str, Any]) -> List[WorkflowTemplate]:
        """
        Get workflow recommendations

        Args:
            context: Current context

        Returns:
            List of recommended workflow templates
        """
        try:
            recommendations = []

            # Match templates based on context
            for template in self.workflow_templates.values():
                score = self._calculate_template_relevance(template, context)
                if score > 0.5:  # Relevance threshold
                    recommendations.append((template, score))

            # Sort by relevance
            recommendations.sort(key=lambda x: x[1], reverse=True)

            # Return template list
            recommended_templates = [t[0] for t in recommendations[:5]]

            self.logger.info(f"Workflow recommendation completed: recommended {len(recommended_templates)} templates")
            return recommended_templates

        except Exception as e:
            self.logger.error(f"Workflow recommendation failed: {e}")
            return []

    def get_automation_statistics(self) -> Dict[str, Any]:
        """
        Get automation statistics

        Returns:
            Statistics dictionary
        """
        try:
            total_executions = len(self.execution_history)
            successful_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)

            # Calculate success rate
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0

            # Calculate template usage
            template_usage = {}
            for template in self.workflow_templates.values():
                template_usage[template.name] = template.usage_count

            # Calculate pattern type distribution
            pattern_types = {}
            for pattern in self.detected_patterns:
                ptype = pattern.pattern_type.value
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

            # Estimate time saved
            estimated_time_saved = self._estimate_time_saved()

            return {
                'total_templates': len(self.workflow_templates),
                'total_patterns': len(self.detected_patterns),
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': round(success_rate, 3),
                'template_usage': template_usage,
                'pattern_type_distribution': pattern_types,
                'estimated_time_saved_minutes': estimated_time_saved,
                'automation_level_distribution': self._get_automation_level_stats()
            }

        except Exception as e:
            self.logger.error(f"Failed to get automation statistics: {e}")
            return {}

    def _initialize_builtin_templates(self):
        """Initialize built-in workflow templates"""

        # Python project initialization template
        python_init_template = WorkflowTemplate(
            template_id="python_project_init",
            name="Python Project Initialization",
            description="Create standard Python project structure",
            category="project_init",
            steps=[
                WorkflowStep(
                    step_id="create_structure",
                    name="Create Project Structure",
                    description="Create standard directory structure",
                    action_type="file_create",
                    parameters={
                        "directories": ["src", "tests", "docs", ".github/workflows"],
                        "files": ["README.md", "requirements.txt", ".gitignore", "setup.py"]
                    }
                ),
                WorkflowStep(
                    step_id="init_git",
                    name="Initialize Git Repository",
                    description="Initialize Git version control",
                    action_type="command",
                    parameters={"command": "git init"}
                ),
                WorkflowStep(
                    step_id="create_venv",
                    name="Create Virtual Environment",
                    description="Create Python virtual environment",
                    action_type="command",
                    parameters={"command": "python -m venv venv"}
                )
            ],
            variables={"project_name": "", "python_version": "3.8"},
            automation_level=AutomationLevel.SEMI_AUTO,
            tags=["python", "initialization", "project"]
        )

        # FastAPI project template
        fastapi_template = WorkflowTemplate(
            template_id="fastapi_project_init",
            name="FastAPI Project Initialization",
            description="Create FastAPI project structure and configuration",
            category="project_init",
            steps=[
                WorkflowStep(
                    step_id="create_fastapi_structure",
                    name="Create FastAPI Structure",
                    description="Create FastAPI project directory",
                    action_type="file_create",
                    parameters={
                        "directories": ["app", "app/api", "app/core", "app/models", "tests"],
                        "files": ["app/main.py", "app/__init__.py", "requirements.txt"]
                    }
                ),
                WorkflowStep(
                    step_id="create_main_file",
                    name="Create Main File",
                    description="Create FastAPI main application file",
                    action_type="file_edit",
                    parameters={
                        "file": "app/main.py",
                        "content": """from fastapi import FastAPI

app = FastAPI(title="{{project_name}}", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}
"""
                    }
                )
            ],
            variables={"project_name": "My API"},
            automation_level=AutomationLevel.SEMI_AUTO,
            tags=["fastapi", "api", "web"]
        )

        # Test setup template
        test_setup_template = WorkflowTemplate(
            template_id="test_setup",
            name="Test Environment Setup",
            description="Setup pytest test environment",
            category="development",
            steps=[
                WorkflowStep(
                    step_id="install_pytest",
                    name="Install Pytest",
                    description="Install test dependencies",
                    action_type="command",
                    parameters={"command": "pip install pytest pytest-cov"}
                ),
                WorkflowStep(
                    step_id="create_test_config",
                    name="Create Test Configuration",
                    description="Create pytest configuration file",
                    action_type="file_create",
                    parameters={
                        "file": "pytest.ini",
                        "content": """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=src --cov-report=html
"""
                    }
                )
            ],
            automation_level=AutomationLevel.FULL_AUTO,
            tags=["testing", "pytest", "setup"]
        )

        # Register built-in templates
        for template in [python_init_template, fastapi_template, test_setup_template]:
            self.workflow_templates[template.template_id] = template

    def _detect_command_sequence_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """Detect command sequence patterns"""
        patterns = []

        # Extract command activities
        command_activities = [a for a in activities if a.action_type == 'command']

        if len(command_activities) < 3:
            return patterns

        # Use sliding window to detect repeated sequences
        window_size = 3
        sequence_counter = Counter()

        for i in range(len(command_activities) - window_size + 1):
            sequence = tuple(a.content for a in command_activities[i:i + window_size])
            sequence_counter[sequence] += 1

        # Identify repeated sequences
        for sequence, frequency in sequence_counter.items():
            if frequency >= self.pattern_config['min_frequency']:
                pattern = Pattern(
                    pattern_type=PatternType.COMMAND_SEQUENCE,
                    name=f"Command Sequence: {' -> '.join(sequence[:2])}...",
                    description=f"Repeatedly executed command sequence, appeared {frequency} times",
                    activities=[a for a in command_activities if a.content in sequence],
                    frequency=frequency,
                    confidence=min(frequency / 10.0, 1.0),
                    last_occurrence=max(a.timestamp for a in command_activities if a.content in sequence),
                    context_similarity=0.8
                )
                patterns.append(pattern)

        return patterns

    def _detect_project_init_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """Detect project initialization patterns"""
        patterns = []

        # Find project creation activities
        project_activities = [a for a in activities if 'init' in a.content.lower() or 'create' in a.content.lower()]

        if len(project_activities) < 2:
            return patterns

        # Analyze project initialization sequences
        init_sequences = defaultdict(list)

        for activity in project_activities:
            if activity.project_path:
                init_sequences[activity.project_path].append(activity)

        # Detect repeated initialization patterns
        for project_path, activities_list in init_sequences.items():
            if len(activities_list) >= 3:
                pattern = Pattern(
                    pattern_type=PatternType.PROJECT_INIT,
                    name="Project Initialization Pattern",
                    description=f"Repeated operation sequence for project initialization",
                    activities=activities_list,
                    frequency=len(activities_list),
                    confidence=0.9,
                    last_occurrence=max(a.timestamp for a in activities_list),
                    context_similarity=0.9
                )
                patterns.append(pattern)

        return patterns

    def _detect_config_generation_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """Detect configuration generation patterns"""
        patterns = []

        # Find configuration file related activities
        config_keywords = ['config', 'settings', '.json', '.yaml', '.ini', '.env']
        config_activities = []

        for activity in activities:
            if any(keyword in activity.content.lower() for keyword in config_keywords):
                config_activities.append(activity)

        if len(config_activities) >= self.pattern_config['min_frequency']:
            pattern = Pattern(
                pattern_type=PatternType.CONFIG_GENERATION,
                name="Configuration File Generation Pattern",
                description="Repeated configuration file creation and modification operations",
                activities=config_activities,
                frequency=len(config_activities),
                confidence=0.7,
                last_occurrence=max(a.timestamp for a in config_activities),
                context_similarity=0.7
            )
            patterns.append(pattern)

        return patterns

    def _detect_file_operation_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """Detect file operation patterns"""
        patterns = []

        # Analyze file operation sequences
        file_operations = [a for a in activities if a.action_type in ['file_edit', 'file_create']]

        if len(file_operations) < 3:
            return patterns

        # Detect repeated file operation patterns
        operation_patterns = defaultdict(list)

        for activity in file_operations:
            # Extract file type
            if '.' in activity.content:
                file_ext = activity.content.split('.')[-1]
                operation_patterns[file_ext].append(activity)

        for file_type, ops in operation_patterns.items():
            if len(ops) >= self.pattern_config['min_frequency']:
                pattern = Pattern(
                    pattern_type=PatternType.FILE_OPERATION,
                    name=f"{file_type} File Operation Pattern",
                    description=f"Repeated {file_type} file operations",
                    activities=ops,
                    frequency=len(ops),
                    confidence=0.6,
                    last_occurrence=max(a.timestamp for a in ops),
                    context_similarity=0.6
                )
                patterns.append(pattern)

        return patterns

    def _detect_development_workflow_patterns(self, activities: List[Activity]) -> List[Pattern]:
        """Detect development workflow patterns"""
        patterns = []

        # Find development related activities
        dev_keywords = ['test', 'build', 'deploy', 'commit', 'push', 'pull']
        dev_activities = []

        for activity in activities:
            if any(keyword in activity.content.lower() for keyword in dev_keywords):
                dev_activities.append(activity)

        if len(dev_activities) >= self.pattern_config['min_frequency']:
            # Sort by time to analyze workflow
            dev_activities.sort(key=lambda x: x.timestamp)

            pattern = Pattern(
                pattern_type=PatternType.DEVELOPMENT_WORKFLOW,
                name="Development Workflow Pattern",
                description="Repeated development workflow process",
                activities=dev_activities,
                frequency=len(dev_activities),
                confidence=0.8,
                last_occurrence=max(a.timestamp for a in dev_activities),
                context_similarity=0.8
            )
            patterns.append(pattern)

        return patterns

    def _filter_and_rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Filter and rank patterns"""
        # Filter low confidence patterns
        filtered = [p for p in patterns if p.confidence >= self.pattern_config['min_confidence']]

        # Sort by frequency and confidence
        filtered.sort(key=lambda p: (p.frequency * p.confidence), reverse=True)

        return filtered[:10]  # Return top 10 patterns

    def _pattern_to_template(self, pattern: Pattern) -> Optional[WorkflowTemplate]:
        """Convert pattern to workflow template"""
        try:
            template_id = f"auto_{pattern.pattern_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            steps = []
            for i, activity in enumerate(pattern.activities[:5]):  # Limit step count
                step = WorkflowStep(
                    step_id=f"step_{i+1}",
                    name=f"Step {i+1}",
                    description=activity.content,
                    action_type=activity.action_type,
                    parameters={"content": activity.content}
                )
                steps.append(step)

            template = WorkflowTemplate(
                template_id=template_id,
                name=pattern.name,
                description=pattern.description,
                category="auto_generated",
                steps=steps,
                automation_level=AutomationLevel.SEMI_AUTO,
                tags=["auto_generated", pattern.pattern_type.value]
            )

            return template

        except Exception as e:
            self.logger.error(f"Failed to convert pattern to template: {e}")
            return None

    def _execute_workflow_step(self, step: WorkflowStep, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step"""
        try:
            result = {
                'step_id': step.step_id,
                'success': False,
                'output': '',
                'error': None,
                'output_vars': {}
            }

            # Substitute variables
            processed_params = self._substitute_variables(step.parameters, variables)

            if step.action_type == 'command':
                # Execute command (simplified here, actual implementation should call system commands)
                command = processed_params.get('command', '')
                result['output'] = f"Simulated command execution: {command}"
                result['success'] = True

            elif step.action_type == 'file_create':
                # Create file (simplified)
                files = processed_params.get('files', [])
                directories = processed_params.get('directories', [])
                result['output'] = f"Simulated file creation: {files}, directories: {directories}"
                result['success'] = True

            elif step.action_type == 'file_edit':
                # Edit file (simplified)
                file_path = processed_params.get('file', '')
                content = processed_params.get('content', '')
                result['output'] = f"Simulated file edit: {file_path}"
                result['success'] = True

            else:
                result['error'] = f"Unknown action type: {step.action_type}"

            return result

        except Exception as e:
            return {
                'step_id': step.step_id,
                'success': False,
                'output': '',
                'error': str(e),
                'output_vars': {}
            }

    def _substitute_variables(self, parameters: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in parameters"""
        result = {}

        for key, value in parameters.items():
            if isinstance(value, str):
                # Simple variable substitution
                for var_name, var_value in variables.items():
                    value = value.replace(f"{{{{{var_name}}}}}", str(var_value))
                result[key] = value
            else:
                result[key] = value

        return result

    def _calculate_template_relevance(self, template: WorkflowTemplate, context: Dict[str, Any]) -> float:
        """Calculate template relevance"""
        score = 0.0

        # Based on tag matching
        context_tags = context.get('tags', [])
        matching_tags = set(template.tags) & set(context_tags)
        score += len(matching_tags) * 0.3

        # Based on category matching
        if template.category == context.get('category'):
            score += 0.4

        # Based on usage frequency
        score += min(template.usage_count / 10.0, 0.3)

        return min(score, 1.0)

    def _estimate_time_saved(self) -> int:
        """Estimate time saved (minutes)"""
        # Simplified calculation: each successful execution saves 5 minutes
        successful_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        return successful_executions * 5

    def _get_automation_level_stats(self) -> Dict[str, int]:
        """Get automation level statistics"""
        stats = defaultdict(int)
        for template in self.workflow_templates.values():
            stats[template.automation_level.value] += 1
        return dict(stats)