#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python code parser

Parses Python source files using the ast module to extract:
- Class definitions and inheritance
- Function/method definitions with signatures
- Import statements and dependencies
- Decorators and type annotations
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ClassInfo:
    """Extracted class information"""
    name: str
    line_number: int
    end_line: int
    docstring: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class FunctionInfo:
    """Extracted function/method information"""
    name: str
    line_number: int
    end_line: int
    docstring: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class ImportInfo:
    """Extracted import information"""
    module: str
    names: List[str]  # For 'from x import a, b'
    aliases: Dict[str, str]  # name -> alias
    line_number: int
    is_from_import: bool = False


@dataclass
class ParsedFile:
    """Complete parsed file information"""
    file_path: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    module_docstring: Optional[str] = None
    syntax_errors: List[str] = field(default_factory=list)


class PythonParser:
    """
    Python AST parser for extracting code structure

    Uses Python's built-in ast module for zero-dependency parsing.
    """

    def __init__(self):
        self.logger = logging.getLogger('hibro.python_parser')

    def parse_file(self, file_path: str) -> ParsedFile:
        """
        Parse a Python file and extract structure

        Args:
            file_path: Path to Python file

        Returns:
            ParsedFile with extracted information
        """
        result = ParsedFile(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            return self.parse_source(source, file_path)

        except FileNotFoundError:
            result.syntax_errors.append(f"File not found: {file_path}")
            return result
        except UnicodeDecodeError as e:
            result.syntax_errors.append(f"Encoding error: {e}")
            return result
        except Exception as e:
            result.syntax_errors.append(f"Read error: {e}")
            return result

    def parse_source(self, source: str, file_path: str = "<string>") -> ParsedFile:
        """
        Parse Python source code string

        Args:
            source: Python source code
            file_path: Optional file path for error messages

        Returns:
            ParsedFile with extracted information
        """
        result = ParsedFile(file_path=file_path)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            result.syntax_errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return result

        # Extract module docstring
        result.module_docstring = ast.get_docstring(tree)

        # Walk the AST
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                result.classes.append(self._extract_class(node))
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                result.functions.append(self._extract_function(node))
            elif isinstance(node, ast.Import):
                result.imports.append(self._extract_import(node))
            elif isinstance(node, ast.ImportFrom):
                result.imports.append(self._extract_import_from(node))

        self.logger.debug(
            f"Parsed {file_path}: {len(result.classes)} classes, "
            f"{len(result.functions)} functions, {len(result.imports)} imports"
        )

        return result

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """
        Extract class information from AST node

        Args:
            node: ClassDef AST node

        Returns:
            ClassInfo object
        """
        # Extract base classes
        bases = []
        for base in node.bases:
            bases.append(self._get_annotation_string(base))

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(self._get_annotation_string(decorator))

        # Extract methods and attributes
        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.append(item.target.id)

        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            bases=bases,
            methods=methods,
            attributes=attributes,
            decorators=decorators
        )

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        """
        Extract function information from AST node

        Args:
            node: FunctionDef or AsyncFunctionDef AST node

        Returns:
            FunctionInfo object
        """
        # Extract parameters
        parameters = []

        # Handle positional arguments
        for arg in node.args.args:
            param = {
                "name": arg.arg,
                "annotation": self._get_annotation_string(arg.annotation) if arg.annotation else None,
                "kind": "positional"
            }
            parameters.append(param)

        # Handle *args
        if node.args.vararg:
            parameters.append({
                "name": f"*{node.args.vararg.arg}",
                "annotation": self._get_annotation_string(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                "kind": "vararg"
            })

        # Handle keyword-only arguments
        for arg in node.args.kwonlyargs:
            param = {
                "name": arg.arg,
                "annotation": self._get_annotation_string(arg.annotation) if arg.annotation else None,
                "kind": "keyword_only"
            }
            parameters.append(param)

        # Handle **kwargs
        if node.args.kwarg:
            parameters.append({
                "name": f"**{node.args.kwarg.arg}",
                "annotation": self._get_annotation_string(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                "kind": "kwarg"
            })

        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_string(node.returns)

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(self._get_annotation_string(decorator))

        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )

    def _extract_import(self, node: ast.Import) -> ImportInfo:
        """
        Extract import information from AST Import node

        Args:
            node: Import AST node

        Returns:
            ImportInfo object
        """
        names = []
        aliases = {}

        for alias in node.names:
            names.append(alias.name)
            if alias.asname:
                aliases[alias.name] = alias.asname

        return ImportInfo(
            module="",
            names=names,
            aliases=aliases,
            line_number=node.lineno,
            is_from_import=False
        )

    def _extract_import_from(self, node: ast.ImportFrom) -> ImportInfo:
        """
        Extract from-import information from AST ImportFrom node

        Args:
            node: ImportFrom AST node

        Returns:
            ImportInfo object
        """
        names = []
        aliases = {}

        for alias in node.names:
            names.append(alias.name)
            if alias.asname:
                aliases[alias.name] = alias.asname

        return ImportInfo(
            module=node.module or "",
            names=names,
            aliases=aliases,
            line_number=node.lineno,
            is_from_import=True
        )

    def _get_annotation_string(self, node: Optional[ast.AST]) -> str:
        """
        Convert annotation AST node to string representation

        Args:
            node: AST node (could be Name, Subscript, Attribute, etc.)

        Returns:
            String representation of the annotation
        """
        if node is None:
            return ""

        try:
            return ast.unparse(node)
        except (AttributeError, ValueError):
            # Fallback for older Python versions or complex nodes
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_annotation_string(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value = self._get_annotation_string(node.value)
                slice_str = self._get_annotation_string(node.slice)
                return f"{value}[{slice_str}]"
            elif isinstance(node, ast.Tuple):
                elements = [self._get_annotation_string(el) for el in node.elts]
                return ", ".join(elements)
            elif isinstance(node, ast.List):
                elements = [self._get_annotation_string(el) for el in node.elts]
                return f"[{', '.join(elements)}]"
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            else:
                return str(type(node).__name__)

    def get_function_calls(self, source: str) -> List[Dict[str, Any]]:
        """
        Extract function call information from source

        Args:
            source: Python source code

        Returns:
            List of function call information
        """
        calls = []

        try:
            tree = ast.parse(source)

            class CallVisitor(ast.NodeVisitor):
                def visit_Call(self, node):
                    call_info = {
                        "line": node.lineno,
                        "col": node.col_offset,
                    }

                    # Get function name
                    if isinstance(node.func, ast.Name):
                        call_info["function"] = node.func.id
                        call_info["type"] = "direct"
                    elif isinstance(node.func, ast.Attribute):
                        call_info["function"] = node.func.attr
                        call_info["object"] = ast.unparse(node.func.value) if hasattr(ast, 'unparse') else ""
                        call_info["type"] = "method"
                    else:
                        call_info["function"] = "<unknown>"
                        call_info["type"] = "other"

                    calls.append(call_info)
                    self.generic_visit(node)

            visitor = CallVisitor()
            visitor.visit(tree)

        except SyntaxError:
            pass

        return calls

    def analyze_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """
        Analyze import dependencies of a Python file

        Args:
            file_path: Path to Python file

        Returns:
            Dict with 'stdlib', 'third_party', 'local' import lists
        """
        result = {
            "stdlib": [],
            "third_party": [],
            "local": [],
            "all": []
        }

        parsed = self.parse_file(file_path)

        # Common stdlib modules (subset)
        STDLIB_MODULES = {
            'os', 'sys', 're', 'json', 'logging', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'datetime', 'time',
            'subprocess', 'threading', 'multiprocessing', 'asyncio',
            'io', 'pickle', 'shutil', 'tempfile', 'hashlib', 'hmac',
            'socket', 'http', 'urllib', 'email', 'html', 'xml',
            'sqlite3', 'csv', 'configparser', 'argparse', 'unittest',
            'dataclasses', 'enum', 'abc', 'copy', 'pprint', 'warnings',
            'contextlib', 'traceback', 'inspect', 'ast', 'dis', 'code',
            'codeop', 'codecs', 'unicodedata', 'string', 'textwrap',
            'difflib', 'heapq', 'bisect', 'array', 'weakref', 'types',
            'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random',
            'statistics', 'operator', 'struct', 'codecs', 'gettext',
            'locale', 'calendar', 'zoneinfo', 'graphlib', 'tomllib'
        }

        for imp in parsed.imports:
            if imp.is_from_import:
                module = imp.module.split('.')[0] if imp.module else ""
            else:
                module = imp.names[0].split('.')[0] if imp.names else ""

            if module:
                result["all"].append(module)

                if module in STDLIB_MODULES:
                    result["stdlib"].append(module)
                elif module.startswith('.'):
                    result["local"].append(module)
                else:
                    # Check if it's a local import (same directory)
                    try:
                        module_path = Path(file_path).parent / f"{module}.py"
                        if module_path.exists():
                            result["local"].append(module)
                        else:
                            result["third_party"].append(module)
                    except:
                        result["third_party"].append(module)

        # Remove duplicates while preserving order
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x in seen or seen.add(x))]

        return result
