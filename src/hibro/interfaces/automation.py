#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automation Rule Engine
Configuration-driven intelligent memory automation processing
"""

import re
import yaml
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from ..core.memory_engine import MemoryEngine
from ..intelligence import MemoryExtractor, ImportanceScorer
from ..utils.config import Config


@dataclass
class AutomationRule:
    """Automation Rule"""
    name: str
    description: str
    trigger_type: str  # 'content', 'pattern', 'keyword', 'context'
    trigger_value: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AutomationEngine:
    """Automation Engine"""

    def __init__(self, config: Config, memory_engine: MemoryEngine):
        """
        Initialize automation engine

        Args:
            config: Configuration object
            memory_engine: Memory engine
        """
        self.config = config
        self.memory_engine = memory_engine
        self.logger = logging.getLogger('hibro.automation_engine')

        # Rule storage
        self.rules: List[AutomationRule] = []
        self.rule_stats: Dict[str, Dict[str, Any]] = {}

        # Action handlers
        self.action_handlers: Dict[str, Callable] = {
            'store_memory': self._handle_store_memory,
            'update_importance': self._handle_update_importance,
            'add_category': self._handle_add_category,
            'set_project': self._handle_set_project,
            'trigger_extraction': self._handle_trigger_extraction,
            'send_notification': self._handle_send_notification,
            'create_snapshot': self._handle_create_snapshot
        }

        # Load rules
        self._load_rules()

    def _load_rules(self):
        """Load automation rules"""
        rules_file = self.config.config_dir / 'automation_rules.yaml'

        try:
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules_data = yaml.safe_load(f) or {}

                self.rules = []
                for rule_data in rules_data.get('rules', []):
                    rule = AutomationRule(**rule_data)
                    self.rules.append(rule)

                self.logger.info(f"Loaded {len(self.rules)} automation rules")
            else:
                # Create default rules
                self._create_default_rules()
                self._save_rules()

        except Exception as e:
            self.logger.error(f"Failed to load automation rules: {e}")
            self._create_default_rules()

    def _create_default_rules(self):
        """Create default automation rules"""
        default_rules = [
            AutomationRule(
                name="Auto-mark important information",
                description="Automatically identify and mark important information",
                trigger_type="keyword",
                trigger_value="important|key|core|must|attention",
                conditions={
                    "min_length": 10,
                    "max_length": 500
                },
                actions=[
                    {
                        "type": "update_importance",
                        "params": {"boost": 0.3}
                    },
                    {
                        "type": "add_category",
                        "params": {"category": "important"}
                    }
                ]
            ),
            AutomationRule(
                name="Auto-classify technical decisions",
                description="Automatically identify and classify technical decisions",
                trigger_type="pattern",
                trigger_value="(decide|choose|adopt).*(technology|framework|library|solution)",
                conditions={
                    "memory_type": ["conversation", "decision"]
                },
                actions=[
                    {
                        "type": "add_category",
                        "params": {"category": "technical_decision"}
                    },
                    {
                        "type": "update_importance",
                        "params": {"boost": 0.2}
                    }
                ]
            ),
            AutomationRule(
                name="Auto-process code snippets",
                description="Automatically identify and process code snippets",
                trigger_type="pattern",
                trigger_value="```|`[^`]+`|function|class|def |import ",
                conditions={
                    "min_length": 20
                },
                actions=[
                    {
                        "type": "add_category",
                        "params": {"category": "code"}
                    },
                    {
                        "type": "trigger_extraction",
                        "params": {"extract_code": True}
                    }
                ]
            ),
            AutomationRule(
                name="Auto-associate project-related",
                description="Automatically associate project-related memories",
                trigger_type="context",
                trigger_value="project_path",
                conditions={},
                actions=[
                    {
                        "type": "set_project",
                        "params": {"from_context": True}
                    },
                    {
                        "type": "create_snapshot",
                        "params": {"if_significant": True}
                    }
                ]
            ),
            AutomationRule(
                name="Reinforce learning content",
                description="Reinforce learning-related memories",
                trigger_type="keyword",
                trigger_value="learned|understand|master|discover|solve",
                conditions={
                    "memory_type": ["learning", "conversation"]
                },
                actions=[
                    {
                        "type": "add_category",
                        "params": {"category": "learning"}
                    },
                    {
                        "type": "update_importance",
                        "params": {"boost": 0.15}
                    }
                ]
            )
        ]

        self.rules = default_rules
        self.logger.info(f"Created {len(default_rules)} default automation rules")

    def _save_rules(self):
        """Save automation rules"""
        rules_file = self.config.config_dir / 'automation_rules.yaml'

        try:
            rules_data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'rules': []
            }

            for rule in self.rules:
                rule_dict = {
                    'name': rule.name,
                    'description': rule.description,
                    'trigger_type': rule.trigger_type,
                    'trigger_value': rule.trigger_value,
                    'conditions': rule.conditions,
                    'actions': rule.actions,
                    'enabled': rule.enabled,
                    'priority': rule.priority,
                    'created_at': rule.created_at.isoformat() if rule.created_at else None
                }
                rules_data['rules'].append(rule_dict)

            with open(rules_file, 'w', encoding='utf-8') as f:
                yaml.dump(rules_data, f, default_flow_style=False, allow_unicode=True, indent=2)

            self.logger.info(f"Saved {len(self.rules)} automation rules")

        except Exception as e:
            self.logger.error(f"Failed to save automation rules: {e}")

    def process_memory(self, memory_content: str, memory_type: str = 'conversation',
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process memory content, apply automation rules

        Args:
            memory_content: Memory content
            memory_type: Memory type
            context: Context information

        Returns:
            Processing results and suggested modifications
        """
        if not memory_content or not memory_content.strip():
            return {}

        # Initialize processing result
        result = {
            'original_content': memory_content,
            'modified_content': memory_content,
            'importance_boost': 0.0,
            'categories': [],
            'project_path': None,
            'actions_taken': [],
            'matched_rules': []
        }

        # Sort rules by priority
        active_rules = [rule for rule in self.rules if rule.enabled]
        active_rules.sort(key=lambda x: x.priority, reverse=True)

        # Apply rules
        for rule in active_rules:
            if self._match_rule(rule, memory_content, memory_type, context):
                self.logger.debug(f"Matched rule: {rule.name}")

                # Check conditions
                if self._check_conditions(rule, memory_content, memory_type, context):
                    # Execute actions
                    actions_result = self._execute_actions(rule, memory_content, context, result)
                    result.update(actions_result)
                    result['matched_rules'].append(rule.name)

                    # Update statistics
                    self._update_rule_stats(rule.name)

        return result

    def _match_rule(self, rule: AutomationRule, content: str, memory_type: str,
                   context: Optional[Dict[str, Any]]) -> bool:
        """
        Check if rule matches

        Args:
            rule: Automation rule
            content: Content
            memory_type: Memory type
            context: Context

        Returns:
            Whether it matches
        """
        try:
            if rule.trigger_type == 'keyword':
                # Keyword matching
                keywords = rule.trigger_value.split('|')
                return any(keyword in content.lower() for keyword in keywords)

            elif rule.trigger_type == 'pattern':
                # Regular expression matching
                return bool(re.search(rule.trigger_value, content, re.IGNORECASE))

            elif rule.trigger_type == 'content':
                # Content contains matching
                return rule.trigger_value.lower() in content.lower()

            elif rule.trigger_type == 'context':
                # Context matching
                if not context:
                    return False
                return rule.trigger_value in context

            return False

        except Exception as e:
            self.logger.warning(f"Rule matching failed {rule.name}: {e}")
            return False

    def _check_conditions(self, rule: AutomationRule, content: str, memory_type: str,
                         context: Optional[Dict[str, Any]]) -> bool:
        """
        Check rule conditions

        Args:
            rule: Automation rule
            content: Content
            memory_type: Memory type
            context: Context

        Returns:
            Whether conditions are met
        """
        try:
            conditions = rule.conditions

            # Check length conditions
            if 'min_length' in conditions:
                if len(content) < conditions['min_length']:
                    return False

            if 'max_length' in conditions:
                if len(content) > conditions['max_length']:
                    return False

            # Check memory type conditions
            if 'memory_type' in conditions:
                allowed_types = conditions['memory_type']
                if isinstance(allowed_types, str):
                    allowed_types = [allowed_types]
                if memory_type not in allowed_types:
                    return False

            # Check context conditions
            if 'context_required' in conditions:
                required_keys = conditions['context_required']
                if isinstance(required_keys, str):
                    required_keys = [required_keys]
                if not context or not all(key in context for key in required_keys):
                    return False

            # Check time conditions
            if 'time_range' in conditions:
                time_range = conditions['time_range']
                current_hour = datetime.now().hour
                if not (time_range.get('start', 0) <= current_hour <= time_range.get('end', 23)):
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Condition check failed {rule.name}: {e}")
            return False

    def _execute_actions(self, rule: AutomationRule, content: str,
                        context: Optional[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rule actions

        Args:
            rule: Automation rule
            content: Content
            context: Context
            result: Current result

        Returns:
            Action execution result
        """
        action_result = {}

        for action in rule.actions:
            action_type = action.get('type')
            action_params = action.get('params', {})

            if action_type in self.action_handlers:
                try:
                    handler_result = self.action_handlers[action_type](
                        content, context, action_params, result
                    )
                    if handler_result:
                        action_result.update(handler_result)

                    result['actions_taken'].append({
                        'rule': rule.name,
                        'action': action_type,
                        'params': action_params,
                        'result': handler_result
                    })

                except Exception as e:
                    self.logger.error(f"Action execution failed {rule.name}.{action_type}: {e}")

        return action_result

    def _handle_store_memory(self, content: str, context: Optional[Dict[str, Any]],
                           params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle store memory action"""
        # This action is usually handled by external caller
        return {'store_memory': True}

    def _handle_update_importance(self, content: str, context: Optional[Dict[str, Any]],
                                params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update importance action"""
        boost = params.get('boost', 0.1)
        result['importance_boost'] += boost
        return {'importance_boost': boost}

    def _handle_add_category(self, content: str, context: Optional[Dict[str, Any]],
                           params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add category action"""
        category = params.get('category')
        if category and category not in result['categories']:
            result['categories'].append(category)
        return {'category_added': category}

    def _handle_set_project(self, content: str, context: Optional[Dict[str, Any]],
                          params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set project action"""
        if params.get('from_context') and context:
            project_path = context.get('project_path')
            if project_path:
                result['project_path'] = project_path
                return {'project_set': project_path}

        project_path = params.get('project_path')
        if project_path:
            result['project_path'] = project_path
            return {'project_set': project_path}

        return {}

    def _handle_trigger_extraction(self, content: str, context: Optional[Dict[str, Any]],
                                 params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trigger extraction action"""
        try:
            extractor = MemoryExtractor()
            extracted_memories = extractor.extract_memories(content)

            if extracted_memories:
                # Update extraction info in result
                result['extracted_memories'] = [
                    {
                        'content': mem.content,
                        'type': mem.memory_type,
                        'importance': mem.importance,
                        'category': mem.category
                    }
                    for mem in extracted_memories
                ]

            return {'extraction_triggered': True, 'extracted_count': len(extracted_memories)}

        except Exception as e:
            self.logger.error(f"Trigger extraction failed: {e}")
            return {'extraction_triggered': False, 'error': str(e)}

    def _handle_send_notification(self, content: str, context: Optional[Dict[str, Any]],
                                params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle send notification action"""
        # Simple log notification
        message = params.get('message', 'Automation rule triggered')
        self.logger.info(f"Notification: {message} - {content[:50]}...")
        return {'notification_sent': True}

    def _handle_create_snapshot(self, content: str, context: Optional[Dict[str, Any]],
                              params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create snapshot action"""
        try:
            if params.get('if_significant') and result.get('importance_boost', 0) < 0.2:
                return {'snapshot_created': False, 'reason': 'not_significant'}

            project_path = result.get('project_path') or (context and context.get('project_path'))
            if project_path:
                # Can trigger snapshot creation here
                return {'snapshot_triggered': True, 'project_path': project_path}

            return {'snapshot_created': False, 'reason': 'no_project'}

        except Exception as e:
            self.logger.error(f"Create snapshot failed: {e}")
            return {'snapshot_created': False, 'error': str(e)}

    def _update_rule_stats(self, rule_name: str):
        """Update rule statistics"""
        if rule_name not in self.rule_stats:
            self.rule_stats[rule_name] = {
                'match_count': 0,
                'last_matched': None,
                'created_at': datetime.now()
            }

        self.rule_stats[rule_name]['match_count'] += 1
        self.rule_stats[rule_name]['last_matched'] = datetime.now()

    def add_rule(self, rule: AutomationRule) -> bool:
        """
        Add new rule

        Args:
            rule: Automation rule

        Returns:
            Whether addition was successful
        """
        try:
            # Check if rule name already exists
            if any(r.name == rule.name for r in self.rules):
                self.logger.warning(f"Rule name already exists: {rule.name}")
                return False

            self.rules.append(rule)
            self._save_rules()

            self.logger.info(f"Added new rule: {rule.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add rule: {e}")
            return False

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove rule

        Args:
            rule_name: Rule name

        Returns:
            Whether removal was successful
        """
        try:
            original_count = len(self.rules)
            self.rules = [rule for rule in self.rules if rule.name != rule_name]

            if len(self.rules) < original_count:
                self._save_rules()
                self.logger.info(f"Removed rule: {rule_name}")
                return True
            else:
                self.logger.warning(f"Rule not found: {rule_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remove rule: {e}")
            return False

    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """
        Enable/disable rule

        Args:
            rule_name: Rule name
            enabled: Whether to enable

        Returns:
            Whether operation was successful
        """
        try:
            for rule in self.rules:
                if rule.name == rule_name:
                    rule.enabled = enabled
                    self._save_rules()
                    status = "enabled" if enabled else "disabled"
                    self.logger.info(f"Rule {status}: {rule_name}")
                    return True

            self.logger.warning(f"Rule not found: {rule_name}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to operate rule: {e}")
            return False

    def get_rule_stats(self) -> Dict[str, Any]:
        """
        Get rule statistics

        Returns:
            Statistics information
        """
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'rule_stats': self.rule_stats.copy(),
            'rules_summary': [
                {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'priority': rule.priority,
                    'trigger_type': rule.trigger_type,
                    'match_count': self.rule_stats.get(rule.name, {}).get('match_count', 0)
                }
                for rule in self.rules
            ]
        }

    def create_rule_report(self) -> str:
        """
        Create rule report

        Returns:
            Rule report text
        """
        stats = self.get_rule_stats()

        report = []
        report.append("# Automation Rules Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Overall Statistics")
        report.append(f"- Total rules: {stats['total_rules']}")
        report.append(f"- Enabled rules: {stats['enabled_rules']}")
        report.append(f"- Disabled rules: {stats['total_rules'] - stats['enabled_rules']}")
        report.append("")

        report.append("## Rule Details")
        for rule_info in stats['rules_summary']:
            status = "✅" if rule_info['enabled'] else "❌"
            report.append(f"### {status} {rule_info['name']}")
            report.append(f"- Trigger type: {rule_info['trigger_type']}")
            report.append(f"- Priority: {rule_info['priority']}")
            report.append(f"- Match count: {rule_info['match_count']}")
            report.append("")

        return "\n".join(report)