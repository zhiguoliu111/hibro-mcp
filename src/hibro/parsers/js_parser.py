#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JavaScript/TypeScript code parser

Parses JS/TS source files using regex patterns to extract:
- Class definitions and inheritance
- Function definitions with signatures
- Import/export statements
- React components
- API routes (Express/Fastify)

Note: This is a lightweight parser using regex patterns.
For more accurate parsing, consider using @babel/parser with Node.js.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class JSClassInfo:
    """Extracted class information"""
    name: str
    line_number: int
    extends: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    is_component: bool = False
    is_exported: bool = False


@dataclass
class JSFunctionInfo:
    """Extracted function information"""
    name: str
    line_number: int
    parameters: List[str] = field(default_factory=list)
    is_async: bool = False
    is_arrow: bool = False
    is_exported: bool = False
    is_component: bool = False


@dataclass
class JSImportInfo:
    """Extracted import information"""
    source: str
    names: List[str] = field(default_factory=list)
    is_default: bool = False
    line_number: int = 0


@dataclass
class JSExportInfo:
    """Extracted export information"""
    name: str
    export_type: str  # 'default', 'named', 'const', 'function', 'class'
    line_number: int = 0


@dataclass
class JSAPIEndpoint:
    """Extracted API endpoint information"""
    method: str
    path: str
    handler: Optional[str] = None
    line_number: int = 0


@dataclass
class ParsedJSFile:
    """Complete parsed JS/TS file information"""
    file_path: str
    is_typescript: bool = False
    classes: List[JSClassInfo] = field(default_factory=list)
    functions: List[JSFunctionInfo] = field(default_factory=list)
    imports: List[JSImportInfo] = field(default_factory=list)
    exports: List[JSExportInfo] = field(default_factory=list)
    api_endpoints: List[JSAPIEndpoint] = field(default_factory=list)
    react_components: List[str] = field(default_factory=list)


class JSParser:
    """
    JavaScript/TypeScript parser using regex patterns

    Provides lightweight parsing without Node.js dependency.
    For complex projects, consider using @babel/parser instead.
    """

    # Regex patterns for parsing
    PATTERNS = {
        # Import patterns
        'import_default': r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
        'import_named': r'import\s+\{\s*([^}]+)\s*\}\s+from\s+[\'"]([^\'"]+)[\'"]',
        'import_all': r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
        'import_side_effect': r'import\s+[\'"]([^\'"]+)[\'"]',

        # Export patterns
        'export_default': r'export\s+default\s+(\w+)',
        'export_named': r'export\s+\{\s*([^}]+)\s*\}',
        'export_const': r'export\s+(?:const|let|var)\s+(\w+)',
        'export_function': r'export\s+(?:async\s+)?function\s+(\w+)',
        'export_class': r'export\s+class\s+(\w+)',

        # Class patterns
        'class': r'class\s+(\w+)(?:\s+extends\s+([\w.]+))?',
        'class_method': r'(\w+)\s*\([^)]*\)\s*\{',

        # Function patterns
        'function': r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        'arrow_function': r'(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>',
        'method_shorthand': r'(\w+)\s*\(([^)]*)\)\s*\{',

        # React component patterns
        'react_function_component': r'(?:export\s+)?(?:default\s+)?function\s+(\w+)(?:\s*\([^)]*\))?\s*(?::\s*React\.(?:FC|FunctionComponent))?',
        'react_arrow_component': r'(?:export\s+)?(?:const|let)\s+(\w+)\s*(?::\s*React\.(?:FC|FunctionComponent))?\s*=\s*\([^)]*\)\s*=>',

        # API endpoint patterns (Express/Fastify)
        'express_route': r'(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
        'fastify_route': r'fastify\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
    }

    def __init__(self):
        self.logger = logging.getLogger('hibro.js_parser')

    def parse_file(self, file_path: str) -> ParsedJSFile:
        """
        Parse a JavaScript/TypeScript file

        Args:
            file_path: Path to JS/TS file

        Returns:
            ParsedJSFile with extracted information
        """
        result = ParsedJSFile(
            file_path=file_path,
            is_typescript=file_path.endswith(('.ts', '.tsx'))
        )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            return self.parse_source(source, file_path)

        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return result
        except UnicodeDecodeError as e:
            self.logger.warning(f"Encoding error in {file_path}: {e}")
            return result
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return result

    def parse_source(self, source: str, file_path: str = "<string>") -> ParsedJSFile:
        """
        Parse JavaScript/TypeScript source code string

        Args:
            source: JS/TS source code
            file_path: Optional file path for reference

        Returns:
            ParsedJSFile with extracted information
        """
        result = ParsedJSFile(
            file_path=file_path,
            is_typescript='.ts' in file_path
        )

        lines = source.split('\n')

        # Extract imports
        result.imports = self._extract_imports(source, lines)

        # Extract exports
        result.exports = self._extract_exports(source, lines)

        # Extract classes
        result.classes = self._extract_classes(source, lines)

        # Extract functions
        result.functions = self._extract_functions(source, lines)

        # Extract API endpoints
        result.api_endpoints = self._extract_api_endpoints(source, lines)

        # Identify React components
        result.react_components = self._identify_react_components(result, source)

        self.logger.debug(
            f"Parsed {file_path}: {len(result.classes)} classes, "
            f"{len(result.functions)} functions, {len(result.imports)} imports, "
            f"{len(result.react_components)} components"
        )

        return result

    def _get_line_number(self, source: str, match_start: int) -> int:
        """Get line number for a match position"""
        return source[:match_start].count('\n') + 1

    def _extract_imports(self, source: str, lines: List[str]) -> List[JSImportInfo]:
        """Extract import statements"""
        imports = []

        # Default imports: import X from 'module'
        for match in re.finditer(self.PATTERNS['import_default'], source):
            name = match.group(1)
            module = match.group(2)
            line = self._get_line_number(source, match.start())
            imports.append(JSImportInfo(
                source=module,
                names=[name],
                is_default=True,
                line_number=line
            ))

        # Named imports: import { X, Y } from 'module'
        for match in re.finditer(self.PATTERNS['import_named'], source):
            names_str = match.group(1)
            module = match.group(2)
            names = [n.strip().split(' as ')[0] for n in names_str.split(',')]
            line = self._get_line_number(source, match.start())
            imports.append(JSImportInfo(
                source=module,
                names=names,
                is_default=False,
                line_number=line
            ))

        # Import all: import * as X from 'module'
        for match in re.finditer(self.PATTERNS['import_all'], source):
            name = match.group(1)
            module = match.group(2)
            line = self._get_line_number(source, match.start())
            imports.append(JSImportInfo(
                source=module,
                names=[f"* as {name}"],
                is_default=False,
                line_number=line
            ))

        return imports

    def _extract_exports(self, source: str, lines: List[str]) -> List[JSExportInfo]:
        """Extract export statements"""
        exports = []

        # Export default
        for match in re.finditer(self.PATTERNS['export_default'], source):
            name = match.group(1)
            line = self._get_line_number(source, match.start())
            exports.append(JSExportInfo(name=name, export_type='default', line_number=line))

        # Export named
        for match in re.finditer(self.PATTERNS['export_named'], source):
            names_str = match.group(1)
            line = self._get_line_number(source, match.start())
            for name in names_str.split(','):
                name = name.strip().split(' as ')[0]
                exports.append(JSExportInfo(name=name, export_type='named', line_number=line))

        # Export const/let/var
        for match in re.finditer(self.PATTERNS['export_const'], source):
            name = match.group(1)
            line = self._get_line_number(source, match.start())
            exports.append(JSExportInfo(name=name, export_type='const', line_number=line))

        return exports

    def _extract_classes(self, source: str, lines: List[str]) -> List[JSClassInfo]:
        """Extract class definitions"""
        classes = []

        for match in re.finditer(self.PATTERNS['class'], source):
            name = match.group(1)
            extends = match.group(2)
            line = self._get_line_number(source, match.start())

            # Check if exported
            line_text = lines[line - 1] if line <= len(lines) else ""
            is_exported = 'export' in line_text

            # Check if React component
            is_component = (
                extends in ['React.Component', 'Component', 'PureComponent'] or
                (extends and extends.endswith('.Component'))
            )

            # Extract methods (simplified - look for patterns after class)
            methods = self._extract_class_methods(source, match.start())

            classes.append(JSClassInfo(
                name=name,
                line_number=line,
                extends=extends,
                methods=methods,
                is_component=is_component,
                is_exported=is_exported
            ))

        return classes

    def _extract_class_methods(self, source: str, class_start: int) -> List[str]:
        """Extract method names from a class"""
        methods = []

        # Find class body (simplified approach)
        # Look for method patterns within reasonable range
        class_body_start = source.find('{', class_start)
        if class_body_start == -1:
            return methods

        # Count braces to find class end
        brace_count = 1
        pos = class_body_start + 1
        max_pos = min(pos + 5000, len(source))  # Limit search range

        while pos < max_pos and brace_count > 0:
            if source[pos] == '{':
                brace_count += 1
            elif source[pos] == '}':
                brace_count -= 1
            pos += 1

        class_body = source[class_body_start:pos]

        # Find method patterns
        for match in re.finditer(self.PATTERNS['class_method'], class_body):
            method_name = match.group(1)
            if method_name not in ['if', 'for', 'while', 'switch', 'catch']:
                methods.append(method_name)

        return methods

    def _extract_functions(self, source: str, lines: List[str]) -> List[JSFunctionInfo]:
        """Extract function definitions"""
        functions = []

        # Regular functions
        for match in re.finditer(self.PATTERNS['function'], source):
            name = match.group(1)
            params_str = match.group(2)
            line = self._get_line_number(source, match.start())

            # Check for async and export
            full_match = match.group(0)
            is_async = 'async' in full_match
            is_exported = 'export' in full_match

            params = [p.strip() for p in params_str.split(',') if p.strip()]

            functions.append(JSFunctionInfo(
                name=name,
                line_number=line,
                parameters=params,
                is_async=is_async,
                is_arrow=False,
                is_exported=is_exported
            ))

        # Arrow functions
        for match in re.finditer(self.PATTERNS['arrow_function'], source):
            name = match.group(1)
            line = self._get_line_number(source, match.start())

            full_match = match.group(0)
            is_async = 'async' in full_match
            is_exported = 'export' in full_match

            functions.append(JSFunctionInfo(
                name=name,
                line_number=line,
                parameters=[],  # Complex to extract from arrow function
                is_async=is_async,
                is_arrow=True,
                is_exported=is_exported
            ))

        return functions

    def _extract_api_endpoints(self, source: str, lines: List[str]) -> List[JSAPIEndpoint]:
        """Extract API route definitions"""
        endpoints = []

        # Express-style routes
        for match in re.finditer(self.PATTERNS['express_route'], source):
            method = match.group(1).upper()
            path = match.group(2)
            line = self._get_line_number(source, match.start())

            endpoints.append(JSAPIEndpoint(
                method=method,
                path=path,
                line_number=line
            ))

        # Fastify-style routes
        for match in re.finditer(self.PATTERNS['fastify_route'], source):
            method = match.group(1).upper()
            path = match.group(2)
            line = self._get_line_number(source, match.start())

            endpoints.append(JSAPIEndpoint(
                method=method,
                path=path,
                line_number=line
            ))

        return endpoints

    def _identify_react_components(self, result: ParsedJSFile, source: str) -> List[str]:
        """Identify React components"""
        components = []

        # Check classes extending React.Component
        for cls in result.classes:
            if cls.is_component:
                components.append(cls.name)

        # Check functions that might be components (start with uppercase, use JSX)
        react_hooks = ['useState', 'useEffect', 'useContext', 'useRef', 'useMemo', 'useCallback']

        for func in result.functions:
            # React component names start with uppercase
            if func.name[0].isupper():
                components.append(func.name)

        # Check for JSX usage
        if '<' in source and '/>' in source or '</' in source:
            # Has JSX, more likely to have components
            pass

        return list(set(components))

    def analyze_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """
        Analyze import dependencies of a JS/TS file

        Args:
            file_path: Path to JS/TS file

        Returns:
            Dict with 'npm', 'local' import lists
        """
        result = {
            "npm": [],
            "local": [],
            "all": []
        }

        parsed = self.parse_file(file_path)

        for imp in parsed.imports:
            module = imp.source

            result["all"].append(module)

            if module.startswith('.'):
                result["local"].append(module)
            else:
                result["npm"].append(module)

        # Remove duplicates
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x in seen or seen.add(x))]

        return result
