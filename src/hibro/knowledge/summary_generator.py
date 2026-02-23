#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary generator

Generates layered summaries of knowledge graph to optimize token usage
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from .graph_storage import GraphStorage, GraphNode, GraphNodeType, RelationType


class SummaryGenerator:
    """
    Summary generator
    """

    def __init__(self, storage: GraphStorage):
        """
        Initialize summary generator

        Args:
            storage: Graph storage layer
        """
        self.storage = storage
        self.logger = logging.getLogger('hibro.summary_generator')

    # ==================== Ultra-lightweight Summary ====================

    def generate_lightweight_summary(
        self,
        project_path: str,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate ultra-lightweight summary (~500 tokens)

        Used for get_quick_context, provides basic project overview

        Args:
            project_path: Project path
            max_tokens: Maximum tokens

        Returns:
            Summary dict
        """
        try:
            self.logger.info(f"Generating lightweight summary: {project_path}")

            # Get all nodes
            all_nodes = self.storage.search_nodes(project_path=project_path, limit=1000)

            # Count node types
            node_counts = defaultdict(int)
            for node in all_nodes:
                node_counts[node.node_type.value] += 1

            # Get file nodes
            file_nodes = [n for n in all_nodes if n.node_type == GraphNodeType.FILE]

            # Get class nodes (sorted by importance)
            class_nodes = sorted(
                [n for n in all_nodes if n.node_type == GraphNodeType.CLASS],
                key=lambda x: x.importance,
                reverse=True
            )

            # Get function nodes
            function_nodes = [n for n in all_nodes if n.node_type == GraphNodeType.FUNCTION]

            # Get API endpoints
            api_nodes = [n for n in all_nodes if n.node_type == GraphNodeType.API_ENDPOINT]

            # Identify core modules (high importance files)
            core_modules = sorted(
                file_nodes,
                key=lambda x: x.importance,
                reverse=True
            )[:5]

            # Get recently modified files
            recent_threshold = datetime.now() - timedelta(days=7)
            recent_files = []
            for node in file_nodes:
                if node.created_at and node.created_at > recent_threshold:
                    recent_files.append(node)

            # Build summary
            summary = {
                "project_path": project_path,
                "generated_at": datetime.now().isoformat(),
                "summary_type": "lightweight",
                "statistics": {
                    "total_files": node_counts.get('file', 0),
                    "total_classes": node_counts.get('class', 0),
                    "total_functions": node_counts.get('function', 0),
                    "total_api_endpoints": node_counts.get('api_endpoint', 0),
                },
                "core_modules": [
                    {
                        "name": node.name,
                        "path": node.file_path,
                        "importance": round(node.importance, 2)
                    }
                    for node in core_modules[:3]
                ],
                "key_classes": [
                    node.name for node in class_nodes[:5]
                ],
                "recent_changes": [
                    {
                        "file": node.file_path,
                        "modified": node.created_at.strftime("%Y-%m-%d") if node.created_at else None
                    }
                    for node in recent_files[:3]
                ]
            }

            self.logger.info(f"Lightweight summary generated: {len(json.dumps(summary))} chars")

            return summary

        except Exception as e:
            self.logger.error(f"Generate lightweight summary failed: {e}")
            return {}

    # ==================== Medium Summary ====================

    def generate_medium_summary(
        self,
        project_path: str,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate medium summary (~2000 tokens)

        Used for specific questions, includes key class and function signatures

        Args:
            project_path: Project path
            max_tokens: Maximum tokens

        Returns:
            Summary dict
        """
        try:
            self.logger.info(f"Generating medium summary: {project_path}")

            # Get all nodes
            all_nodes = self.storage.search_nodes(project_path=project_path, limit=1000)

            # Get class nodes (sorted by importance)
            class_nodes = sorted(
                [n for n in all_nodes if n.node_type == GraphNodeType.CLASS],
                key=lambda x: x.importance,
                reverse=True
            )

            # Get function nodes (sorted by importance)
            function_nodes = sorted(
                [n for n in all_nodes if n.node_type == GraphNodeType.FUNCTION],
                key=lambda x: x.importance,
                reverse=True
            )

            # Get API endpoints
            api_nodes = [n for n in all_nodes if n.node_type == GraphNodeType.API_ENDPOINT]

            # Build class info (including methods)
            classes_info = []
            for class_node in class_nodes[:10]:  # Limit to top 10 classes
                class_info = {
                    "name": class_node.name,
                    "file": class_node.file_path,
                    "line": class_node.line_number,
                    "importance": round(class_node.importance, 2),
                }

                # Add class metadata
                if class_node.metadata:
                    if "methods" in class_node.metadata:
                        class_info["methods"] = class_node.metadata["methods"]
                    if "inherits_from" in class_node.metadata:
                        class_info["inherits_from"] = class_node.metadata["inherits_from"]
                    if "docstring" in class_node.metadata:
                        class_info["docstring"] = class_node.metadata["docstring"][:100]  # Limit length

                classes_info.append(class_info)

            # Build function info
            functions_info = []
            for func_node in function_nodes[:20]:  # Limit to top 20 functions
                func_info = {
                    "name": func_node.name,
                    "file": func_node.file_path,
                    "line": func_node.line_number,
                    "importance": round(func_node.importance, 2),
                }

                # Add function metadata
                if func_node.metadata:
                    if "signature" in func_node.metadata:
                        func_info["signature"] = func_node.metadata["signature"]
                    if "parameters" in func_node.metadata:
                        func_info["parameters"] = func_node.metadata["parameters"]
                    if "return_type" in func_node.metadata:
                        func_info["return_type"] = func_node.metadata["return_type"]

                functions_info.append(func_info)

            # Build API endpoint info
            api_info = []
            for api_node in api_nodes[:15]:  # Limit to top 15 endpoints
                endpoint_info = {
                    "path": api_node.name,
                    "file": api_node.file_path,
                    "line": api_node.line_number,
                }

                if api_node.metadata:
                    if "method" in api_node.metadata:
                        endpoint_info["method"] = api_node.metadata["method"]
                    if "handler_function" in api_node.metadata:
                        endpoint_info["handler"] = api_node.metadata["handler_function"]

                api_info.append(endpoint_info)

            # Analyze module dependencies
            module_deps = self._analyze_module_dependencies(all_nodes)

            # Build summary
            summary = {
                "project_path": project_path,
                "generated_at": datetime.now().isoformat(),
                "summary_type": "medium",
                "classes": classes_info,
                "functions": functions_info,
                "api_endpoints": api_info,
                "module_dependencies": module_deps,
            }

            self.logger.info(f"Medium summary generated: {len(json.dumps(summary))} chars")

            return summary

        except Exception as e:
            self.logger.error(f"Generate medium summary failed: {e}")
            return {}

    # ==================== Detailed Information ====================

    def get_node_details(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed node information (on-demand loading)

        Args:
            node_id: Node ID

        Returns:
            Detailed info dict
        """
        try:
            node = self.storage.get_node(node_id)
            if not node:
                return None

            # Get all node relations
            relations = self.storage.get_node_relations(node_id, direction='both')

            # Build detailed info
            details = {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "name": node.name,
                "file_path": node.file_path,
                "line_number": node.line_number,
                "importance": node.importance,
                "metadata": node.metadata,
                "file_hash": node.file_hash,
                "created_at": node.created_at.isoformat() if node.created_at else None,
                "relations": {
                    "outgoing": [],
                    "incoming": []
                }
            }

            # Add relation info
            for relation, related_node in relations:
                relation_info = {
                    "type": relation.relation_type.value,
                    "target": related_node.name,
                    "target_type": related_node.node_type.value,
                    "weight": relation.weight
                }

                if relation.source_node_id == node_id:
                    details["relations"]["outgoing"].append(relation_info)
                else:
                    details["relations"]["incoming"].append(relation_info)

            return details

        except Exception as e:
            self.logger.error(f"Get node details failed: {e}")
            return None

    # ==================== Helper Methods ====================

    def _analyze_module_dependencies(self, nodes: List[GraphNode]) -> Dict[str, List[str]]:
        """
        Analyze module dependencies

        Args:
            nodes: Node list

        Returns:
            Dependencies dict
        """
        dependencies = defaultdict(set)

        # Get all file nodes
        file_nodes = {n.node_id: n for n in nodes if n.node_type == GraphNodeType.FILE}

        # Analyze import relations for each file
        for file_id, file_node in file_nodes.items():
            relations = self.storage.get_node_relations(file_id, relation_type=RelationType.IMPORTS, direction='outgoing')

            for relation, target_node in relations:
                if target_node.node_type == GraphNodeType.FILE:
                    dependencies[file_node.name].add(target_node.name)

        # Convert to regular dict
        return {k: list(v) for k, v in dependencies.items() if v}

    def generate_text_summary(self, summary: Dict[str, Any]) -> str:
        """
        Convert summary dict to text format

        Args:
            summary: Summary dict

        Returns:
            Text summary
        """
        if summary.get("summary_type") == "lightweight":
            return self._format_lightweight_summary(summary)
        elif summary.get("summary_type") == "medium":
            return self._format_medium_summary(summary)
        else:
            return json.dumps(summary, indent=2, ensure_ascii=False)

    def _format_lightweight_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format lightweight summary to text

        Args:
            summary: Summary dict

        Returns:
            Text summary
        """
        lines = []
        lines.append(f"Project: {summary['project_path']}")
        lines.append("")

        stats = summary.get("statistics", {})
        lines.append("Statistics:")
        lines.append(f"  - Files: {stats.get('total_files', 0)}")
        lines.append(f"  - Classes: {stats.get('total_classes', 0)}")
        lines.append(f"  - Functions: {stats.get('total_functions', 0)}")
        lines.append(f"  - API Endpoints: {stats.get('total_api_endpoints', 0)}")
        lines.append("")

        core_modules = summary.get("core_modules", [])
        if core_modules:
            lines.append("Core Modules:")
            for module in core_modules:
                lines.append(f"  - {module['name']} (importance: {module['importance']})")
            lines.append("")

        key_classes = summary.get("key_classes", [])
        if key_classes:
            lines.append("Key Classes:")
            lines.append(f"  {', '.join(key_classes)}")
            lines.append("")

        recent_changes = summary.get("recent_changes", [])
        if recent_changes:
            lines.append("Recent Changes:")
            for change in recent_changes:
                lines.append(f"  - {change['file']} ({change['modified']})")

        return "\n".join(lines)

    def _format_medium_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format medium summary to text

        Args:
            summary: Summary dict

        Returns:
            Text summary
        """
        lines = []
        lines.append(f"Project: {summary['project_path']}")
        lines.append("")

        # Class info
        classes = summary.get("classes", [])
        if classes:
            lines.append("Key Classes:")
            for cls in classes[:5]:  # Show only top 5
                lines.append(f"\n  {cls['name']} ({cls['file']}:{cls['line']})")
                if "methods" in cls:
                    lines.append(f"    Methods: {', '.join(cls['methods'][:5])}")
                if "inherits_from" in cls and cls['inherits_from']:
                    lines.append(f"    Inherits: {', '.join(cls['inherits_from'])}")
            lines.append("")

        # Function info
        functions = summary.get("functions", [])
        if functions:
            lines.append("Key Functions:")
            for func in functions[:10]:  # Show only top 10
                signature = func.get("signature", func['name'])
                lines.append(f"  - {signature} ({func['file']}:{func['line']})")
            lines.append("")

        # API endpoints
        api_endpoints = summary.get("api_endpoints", [])
        if api_endpoints:
            lines.append("API Endpoints:")
            for api in api_endpoints[:10]:  # Show only top 10
                method = api.get("method", "")
                handler = api.get("handler", "")
                lines.append(f"  - {method} {api['path']} -> {handler}")
            lines.append("")

        # Module dependencies
        module_deps = summary.get("module_dependencies", {})
        if module_deps:
            lines.append("Module Dependencies:")
            for module, deps in list(module_deps.items())[:5]:  # Show only top 5
                lines.append(f"  {module} -> {', '.join(deps[:3])}")

        return "\n".join(lines)
