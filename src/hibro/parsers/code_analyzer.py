#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code analyzer

Provides high-level analysis combining parsers to build:
- Function call graphs
- Class inheritance trees
- API endpoint mappings
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .python_parser import PythonParser, ParsedFile, FunctionInfo, ClassInfo
from .js_parser import JSParser, ParsedJSFile, JSFunctionInfo, JSClassInfo, JSAPIEndpoint
from .vue_parser import VueParser, ParsedVueFile, VueComponentInfo


@dataclass
class CallRelation:
    """Function call relation"""
    caller: str
    caller_file: str
    callee: str
    callee_file: Optional[str] = None
    line_number: int = 0


@dataclass
class InheritanceRelation:
    """Class inheritance relation"""
    child: str
    child_file: str
    parent: str
    parent_file: Optional[str] = None


@dataclass
class APIEndpoint:
    """API endpoint information"""
    path: str
    method: str
    handler: Optional[str] = None
    file_path: str = ""
    line_number: int = 0


@dataclass
class ProjectAnalysis:
    """Complete project analysis result"""
    project_path: str
    call_graph: List[CallRelation] = field(default_factory=list)
    inheritance_tree: List[InheritanceRelation] = field(default_factory=list)
    api_endpoints: List[APIEndpoint] = field(default_factory=list)
    all_classes: Dict[str, List[str]] = field(default_factory=dict)  # class -> [files]
    all_functions: Dict[str, List[str]] = field(default_factory=dict)  # function -> [files]


class CodeAnalyzer:
    """
    High-level code analyzer

    Combines parsers to build comprehensive code structure analysis.
    """

    # File extensions to analyze
    PYTHON_EXTENSIONS = {'.py'}
    JS_EXTENSIONS = {'.js', '.jsx', '.ts', '.tsx'}
    VUE_EXTENSIONS = {'.vue'}

    def __init__(self):
        self.python_parser = PythonParser()
        self.js_parser = JSParser()
        self.vue_parser = VueParser()
        self.logger = logging.getLogger('hibro.code_analyzer')

    def analyze_project(self, project_path: str, files: Optional[List[str]] = None) -> ProjectAnalysis:
        """
        Analyze a project's code structure

        Args:
            project_path: Root path of the project
            files: Optional list of specific files to analyze

        Returns:
            ProjectAnalysis with complete structure information
        """
        result = ProjectAnalysis(project_path=project_path)

        if files is None:
            files = self._discover_files(project_path)

        self.logger.info(f"Analyzing {len(files)} files in {project_path}")

        # Class and function registry for cross-file analysis
        class_registry: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # name -> [(file, line)]
        function_registry: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for file_path in files:
            ext = Path(file_path).suffix.lower()

            if ext in self.PYTHON_EXTENSIONS:
                self._analyze_python_file(file_path, result, class_registry, function_registry)
            elif ext in self.JS_EXTENSIONS:
                self._analyze_js_file(file_path, result, class_registry, function_registry)
            elif ext in self.VUE_EXTENSIONS:
                self._analyze_vue_file(file_path, result, class_registry, function_registry)

        # Resolve inheritance parent locations
        self._resolve_inheritance(result, class_registry)

        self.logger.info(
            f"Analysis complete: {len(result.call_graph)} calls, "
            f"{len(result.inheritance_tree)} inheritance, "
            f"{len(result.api_endpoints)} endpoints"
        )

        return result

    def _discover_files(self, project_path: str) -> List[str]:
        """Discover source files in project"""
        files = []
        project = Path(project_path)

        # Excluded directories
        excluded = {'node_modules', '__pycache__', '.git', '.venv', 'venv',
                   'dist', 'build', '.next', 'coverage', '.pytest_cache'}

        for ext in self.PYTHON_EXTENSIONS | self.JS_EXTENSIONS | self.VUE_EXTENSIONS:
            for file in project.rglob(f'*{ext}'):
                # Skip excluded directories
                if any(part in excluded for part in file.parts):
                    continue
                files.append(str(file))

        return files

    def _analyze_python_file(
        self,
        file_path: str,
        result: ProjectAnalysis,
        class_registry: Dict[str, List[Tuple[str, int]]],
        function_registry: Dict[str, List[Tuple[str, int]]]
    ):
        """Analyze a Python file"""
        parsed = self.python_parser.parse_file(file_path)

        # Register classes
        for cls in parsed.classes:
            class_registry[cls.name].append((file_path, cls.line_number))
            result.all_classes[cls.name] = result.all_classes.get(cls.name, [])
            if file_path not in result.all_classes[cls.name]:
                result.all_classes[cls.name].append(file_path)

            # Record inheritance
            for base in cls.bases:
                result.inheritance_tree.append(InheritanceRelation(
                    child=cls.name,
                    child_file=file_path,
                    parent=base
                ))

        # Register functions
        for func in parsed.functions:
            function_registry[func.name].append((file_path, func.line_number))
            result.all_functions[func.name] = result.all_functions.get(func.name, [])
            if file_path not in result.all_functions[func.name]:
                result.all_functions[func.name].append(file_path)

        # Extract function calls
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            calls = self.python_parser.get_function_calls(source)

            for call in calls:
                result.call_graph.append(CallRelation(
                    caller="",  # Would need more complex analysis to determine caller
                    caller_file=file_path,
                    callee=call["function"],
                    line_number=call["line"]
                ))
        except:
            pass

    def _analyze_js_file(
        self,
        file_path: str,
        result: ProjectAnalysis,
        class_registry: Dict[str, List[Tuple[str, int]]],
        function_registry: Dict[str, List[Tuple[str, int]]]
    ):
        """Analyze a JavaScript/TypeScript file"""
        parsed = self.js_parser.parse_file(file_path)

        # Register classes
        for cls in parsed.classes:
            class_registry[cls.name].append((file_path, cls.line_number))
            result.all_classes[cls.name] = result.all_classes.get(cls.name, [])
            if file_path not in result.all_classes[cls.name]:
                result.all_classes[cls.name].append(file_path)

            # Record inheritance
            if cls.extends:
                result.inheritance_tree.append(InheritanceRelation(
                    child=cls.name,
                    child_file=file_path,
                    parent=cls.extends
                ))

        # Register functions
        for func in parsed.functions:
            function_registry[func.name].append((file_path, func.line_number))
            result.all_functions[func.name] = result.all_functions.get(func.name, [])
            if file_path not in result.all_functions[func.name]:
                result.all_functions[func.name].append(file_path)

        # Extract API endpoints
        for endpoint in parsed.api_endpoints:
            result.api_endpoints.append(APIEndpoint(
                path=endpoint.path,
                method=endpoint.method,
                handler=endpoint.handler,
                file_path=file_path,
                line_number=endpoint.line_number
            ))

    def _analyze_vue_file(
        self,
        file_path: str,
        result: ProjectAnalysis,
        class_registry: Dict[str, List[Tuple[str, int]]],
        function_registry: Dict[str, List[Tuple[str, int]]]
    ):
        """Analyze a Vue Single File Component"""
        parsed = self.vue_parser.parse_file(file_path)

        # Register component
        if parsed.component:
            class_registry[parsed.component.name].append((file_path, parsed.component.line_number))
            result.all_classes[parsed.component.name] = result.all_classes.get(parsed.component.name, [])
            if file_path not in result.all_classes[parsed.component.name]:
                result.all_classes[parsed.component.name].append(file_path)

            # Record component hierarchy (extends)
            for cls in parsed.classes:
                if cls.extends:
                    result.inheritance_tree.append(InheritanceRelation(
                        child=cls.name,
                        child_file=file_path,
                        parent=cls.extends
                    ))

        # Register functions from script
        for func in parsed.functions:
            function_registry[func.name].append((file_path, func.line_number))
            result.all_functions[func.name] = result.all_functions.get(func.name, [])
            if file_path not in result.all_functions[func.name]:
                result.all_functions[func.name].append(file_path)

        # Register classes from script (Options API)
        for cls in parsed.classes:
            class_registry[cls.name].append((file_path, cls.line_number))
            result.all_classes[cls.name] = result.all_classes.get(cls.name, [])
            if file_path not in result.all_classes[cls.name]:
                result.all_classes[cls.name].append(file_path)

            # Record inheritance
            if cls.extends:
                result.inheritance_tree.append(InheritanceRelation(
                    child=cls.name,
                    child_file=file_path,
                    parent=cls.extends
                ))

    def _resolve_inheritance(
        self,
        result: ProjectAnalysis,
        class_registry: Dict[str, List[Tuple[str, int]]]
    ):
        """Resolve parent class locations in inheritance relations"""
        for rel in result.inheritance_tree:
            if rel.parent in class_registry:
                # Use first occurrence as parent location
                rel.parent_file = class_registry[rel.parent][0][0]

    def get_class_hierarchy(self, analysis: ProjectAnalysis, class_name: str) -> Dict[str, Any]:
        """
        Get full inheritance hierarchy for a class

        Args:
            analysis: Project analysis result
            class_name: Class name to get hierarchy for

        Returns:
            Dict with 'ancestors', 'descendants', 'siblings'
        """
        hierarchy = {
            "class": class_name,
            "ancestors": [],
            "descendants": [],
            "siblings": []
        }

        # Find ancestors (parents, grandparents, etc.)
        visited = {class_name}
        to_visit = [class_name]

        while to_visit:
            current = to_visit.pop(0)
            for rel in analysis.inheritance_tree:
                if rel.child == current and rel.parent not in visited:
                    visited.add(rel.parent)
                    to_visit.append(rel.parent)
                    if current == class_name:
                        hierarchy["ancestors"].append(rel.parent)
                    else:
                        hierarchy["ancestors"].append(rel.parent)

        # Find descendants (children, grandchildren, etc.)
        visited = {class_name}
        to_visit = [class_name]

        while to_visit:
            current = to_visit.pop(0)
            for rel in analysis.inheritance_tree:
                if rel.parent == current and rel.child not in visited:
                    visited.add(rel.child)
                    to_visit.append(rel.child)
                    hierarchy["descendants"].append(rel.child)

        # Find siblings (same parent)
        for rel in analysis.inheritance_tree:
            if rel.child == class_name:
                parent = rel.parent
                for sibling_rel in analysis.inheritance_tree:
                    if (sibling_rel.parent == parent and
                        sibling_rel.child != class_name and
                        sibling_rel.child not in hierarchy["siblings"]):
                        hierarchy["siblings"].append(sibling_rel.child)

        return hierarchy

    def find_api_endpoints(
        self,
        analysis: ProjectAnalysis,
        method: Optional[str] = None,
        path_pattern: Optional[str] = None
    ) -> List[APIEndpoint]:
        """
        Find API endpoints matching criteria

        Args:
            analysis: Project analysis result
            method: HTTP method filter (GET, POST, etc.)
            path_pattern: Path pattern to match

        Returns:
            List of matching endpoints
        """
        endpoints = analysis.api_endpoints

        if method:
            endpoints = [e for e in endpoints if e.method == method.upper()]

        if path_pattern:
            endpoints = [e for e in endpoints if path_pattern in e.path]

        return endpoints

    def get_function_callers(
        self,
        analysis: ProjectAnalysis,
        function_name: str
    ) -> List[CallRelation]:
        """
        Find all callers of a function

        Args:
            analysis: Project analysis result
            function_name: Function name to find callers for

        Returns:
            List of call relations
        """
        return [
            call for call in analysis.call_graph
            if call.callee == function_name
        ]
