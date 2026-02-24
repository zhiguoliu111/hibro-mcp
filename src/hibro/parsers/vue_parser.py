#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vue Single File Component parser

Parses Vue .vue files (SFC - Single File Components) to extract:
- Component name
- Script setup / script content
- Template structure
- Props, emits, composables
- Methods, computed properties
- Lifecycle hooks
- Components (script setup and Options API)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .js_parser import JSParser, JSClassInfo, JSFunctionInfo, ParsedJSFile


@dataclass
class VueComponentInfo:
    """Extracted Vue component information"""
    name: str
    line_number: int
    is_export_default: bool = False
    is_script_setup: bool = False
    props: List[str] = field(default_factory=list)
    emits: List[str] = field(default_factory=list)
    composables: List[str] = field(default_factory=list)
    lifecycle_hooks: List[str] = field(default_factory=list)
    template_refs: List[str] = field(default_factory=list)


@dataclass
class VueTemplateInfo:
    """Extracted template information"""
    has_template: bool
    line_number: int = 0
    element_count: int = 0
    custom_components: List[str] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)  # v-model, v-bind, etc.


@dataclass
class VueStyleInfo:
    """Extracted style information"""
    has_style: bool = False
    line_number: int = 0
    is_scoped: bool = False
    lang: Optional[str] = None


@dataclass
class ParsedVueFile:
    """Complete parsed Vue file information"""
    file_path: str
    component: Optional[VueComponentInfo] = None
    template: Optional[VueTemplateInfo] = None
    style: Optional[VueStyleInfo] = None
    # Script content parsing results
    classes: List[JSClassInfo] = field(default_factory=list)
    functions: List[JSFunctionInfo] = field(default_factory=list)
    imports: List[Any] = field(default_factory=list)
    exports: List[Any] = field(default_factory=list)


class VueParser:
    """
    Vue Single File Component (SFC) parser

    Parses .vue files by:
    1. Extracting <script>, <template>, and <style> blocks
    2. Parsing script content using JSParser
    3. Extracting Vue-specific features (props, emits, etc.)
    4. Analyzing template structure
    """

    # Patterns for extracting SFC blocks
    BLOCK_PATTERNS = {
        'script': r'<script\s*(?:setup)?(?:\s+lang="?(ts|javascript)"?)?\s*>(.*?)</script>',
        'script_setup': r'<script\s+setup(?:\s+lang="?(ts|javascript)"?)?\s*>(.*?)</script>',
        'template': r'<template\s*(?:lang="?(pug|html)"?)?\s*>(.*?)</template>',
        'style': r'<style\s*(?:scoped)?(?:\s+lang="?(css|scss|less|sass)"?)?\s*>(.*?)</style>',
    }

    # Vue-specific patterns
    VUE_PATTERNS = {
        # Script setup patterns
        'define_props': r'(?:const|let)\s+(\w+)\s*=\s*defineProps(?:<[^>]+>)?\s*\(',
        'define_emits': r'(?:const|let)\s+(\w+)\s*=\s*defineEmits(?:<[^>]+>)?\s*\(\[',
        'define_expose': r'defineExpose\s*\(',
        'use_composable': r'(?:const|let)\s+(\w+)\s*=\s*(use\w+)\s*\(',

        # Lifecycle hooks
        'on_mounted': r'onMounted\s*\(',
        'on_created': r'onCreated?\s*\(',
        'on_before_mount': r'onBeforeMount\s*\(',
        'on_before_unmount': r'onBeforeUnmount\s*\(',
        'on_unmounted': r'onUnmounted\s*\(',

        # Template refs
        'template_ref': r'(?:const|let)\s+(\w+)\s*=\s*(?:ref|shallowRef|computed)\s*\(',

        # Template patterns
        'v_model': r'v-model\s*=?\s*["\']?(\w+)["\']?',
        'v_bind': r'v-bind\s*:?\.?\s*(\w+)',
        'v_on': r'@?\s*(\w+)\s*=',
        'v_for': r'v-for\s*=\s*["\']?\w+\s+(?:in|of)\s+(\w+)["\']?',
        'v_if': r'v-if\s*=\s*["\']?(\w+)["\']?',

        # Custom component (PascalCase or kebab-case tags)
        'custom_element': r'<([A-Z][a-zA-Z0-9]*)[\s>]',
    }

    def __init__(self):
        self.logger = logging.getLogger('hibro.vue_parser')
        self.js_parser = JSParser()

    def parse_file(self, file_path: str) -> ParsedVueFile:
        """
        Parse a Vue Single File Component

        Args:
            file_path: Path to .vue file

        Returns:
            ParsedVueFile with extracted information
        """
        result = ParsedVueFile(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self._parse_content(content, file_path)

        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return result
        except UnicodeDecodeError as e:
            self.logger.warning(f"Encoding error in {file_path}: {e}")
            return result
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return result

    def _parse_content(self, content: str, file_path: str) -> ParsedVueFile:
        """Parse Vue file content"""
        result = ParsedVueFile(file_path=file_path)

        # Extract script block
        script_content, script_line, is_setup, script_lang = self._extract_script(content)
        if script_content:
            # Parse script content with JS parser
            js_result = self.js_parser.parse_source(script_content, file_path)
            result.classes = js_result.classes
            result.functions = js_result.functions
            result.imports = js_result.imports
            result.exports = js_result.exports

            # Extract Vue-specific info
            result.component = self._extract_component_info(
                script_content, script_line, is_setup
            )

        # Extract template
        template_content, template_line = self._extract_template(content)
        if template_content:
            result.template = self._extract_template_info(template_content, template_line)

        # Extract style
        style_content, style_line, style_attrs = self._extract_style(content)
        if style_content:
            result.style = VueStyleInfo(
                has_style=True,
                line_number=style_line,
                is_scoped='scoped' in style_attrs,
                lang=style_attrs.get('lang')
            )

        self.logger.debug(
            f"Parsed Vue file {file_path}: "
            f"script={'setup' if is_setup else 'options'}, "
            f"{len(result.functions)} functions, "
            f"{len(result.classes)} classes"
        )

        return result

    def _extract_script(
        self, content: str
    ) -> Tuple[Optional[str], int, bool, Optional[str]]:
        """
        Extract <script> block content

        Returns:
            (content, line_number, is_setup, lang)
        """
        # Try script setup first
        for match in re.finditer(self.BLOCK_PATTERNS['script_setup'], content, re.DOTALL):
            script_content = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            lang = match.group(1) if match.group(1) else None
            return script_content, line_number, True, lang

        # Fallback to regular script
        for match in re.finditer(self.BLOCK_PATTERNS['script'], content, re.DOTALL):
            script_content = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            lang = match.group(1) if match.group(1) else None
            return script_content, line_number, False, lang

        return None, 0, False, None

    def _extract_template(self, content: str) -> Tuple[Optional[str], int]:
        """Extract <template> block content"""
        for match in re.finditer(self.BLOCK_PATTERNS['template'], content, re.DOTALL):
            template_content = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            return template_content, line_number
        return None, 0

    def _extract_style(self, content: str) -> Tuple[Optional[str], int, Dict[str, str]]:
        """Extract <style> block content and attributes"""
        pattern = r'<style\s*(.*?)>(.*?)</style>'
        for match in re.finditer(pattern, content, re.DOTALL):
            style_content = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1

            # Parse attributes
            attrs = {}
            attrs_str = match.group(1)
            if 'scoped' in attrs_str:
                attrs['scoped'] = 'true'
            lang_match = re.search(r'lang\s*=\s*["\']?(\w+)["\']?', attrs_str)
            if lang_match:
                attrs['lang'] = lang_match.group(1)

            return style_content, line_number, attrs
        return None, 0, {}

    def _extract_component_info(
        self, script_content: str, line_number: int, is_setup: bool
    ) -> Optional[VueComponentInfo]:
        """Extract Vue component information from script content"""
        component = VueComponentInfo(
            name="Component",  # Default name
            line_number=line_number,
            is_script_setup=is_setup
        )

        # Extract props (defineProps)
        for match in re.finditer(self.VUE_PATTERNS['define_props'], script_content):
            prop_name = match.group(1)
            if prop_name not in component.props:
                component.props.append(prop_name)

        # Extract emits (defineEmits)
        for match in re.finditer(self.VUE_PATTERNS['define_emits'], script_content):
            emit_name = match.group(1)
            if emit_name not in component.emits:
                component.emits.append(emit_name)

        # Extract composables (use*)
        for match in re.finditer(self.VUE_PATTERNS['use_composable'], script_content):
            composable_name = match.group(2)
            if composable_name not in component.composables:
                component.composables.append(composable_name)

        # Extract lifecycle hooks
        for hook_name, pattern in self.VUE_PATTERNS.items():
            if 'on_' in hook_name:
                if re.search(pattern, script_content):
                    component.lifecycle_hooks.append(hook_name.replace('on_', 'on'))

        # Extract template refs
        for match in re.finditer(self.VUE_PATTERNS['template_ref'], script_content):
            ref_name = match.group(1)
            if ref_name not in component.template_refs:
                component.template_refs.append(ref_name)

        # Try to extract component name from export default
        export_match = re.search(r'export\s+default\s*\{[^}]*name:\s*["\'](\w+)["\']', script_content)
        if export_match:
            component.name = export_match.group(1)
            component.is_export_default = True

        # For script setup, try to infer component name from file name or context
        if is_setup and not component.is_export_default:
            component.name = "VueComponent (script setup)"

        return component

    def _extract_template_info(
        self, template_content: str, line_number: int
    ) -> VueTemplateInfo:
        """Extract information from template block"""
        info = VueTemplateInfo(
            has_template=True,
            line_number=line_number
        )

        # Count elements
        info.element_count = len(re.findall(r'<\w+', template_content))

        # Find custom components (PascalCase tags)
        for match in re.finditer(self.VUE_PATTERNS['custom_element'], template_content):
            component = match.group(1)
            if component not in info.custom_components:
                info.custom_components.append(component)

        # Find bindings
        for pattern_name, pattern in self.VUE_PATTERNS.items():
            if 'v_' in pattern_name or '@' in pattern or ':' in pattern_name:
                for match in re.finditer(pattern, template_content):
                    binding = match.group(1) if match.groups() else pattern_name
                    if binding not in info.bindings:
                        info.bindings.append(f"{pattern_name}: {binding}")

        return info
