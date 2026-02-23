#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test code analyzer
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.parsers.code_analyzer import (
    CodeAnalyzer, ProjectAnalysis, CallRelation, InheritanceRelation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_code_analyzer():
    """Test code analyzer"""

    logger.info("=" * 60)
    logger.info("Starting code analyzer test")
    logger.info("=" * 60)

    try:
        analyzer = CodeAnalyzer()

        # 1. Test analyzing current project
        logger.info("\n1. Testing project analysis")

        analysis = analyzer.analyze_project(str(project_root))

        logger.info(f"Classes found: {len(analysis.all_classes)}")
        logger.info(f"Functions found: {len(analysis.all_functions)}")
        logger.info(f"Call relations: {len(analysis.call_graph)}")
        logger.info(f"Inheritance relations: {len(analysis.inheritance_tree)}")

        assert len(analysis.all_classes) > 0, "Should find classes"
        assert len(analysis.all_functions) > 0, "Should find functions"

        # 2. Test class registry
        logger.info("\n2. Testing class registry")

        # Should find PythonParser, JSParser, CodeAnalyzer etc.
        expected_classes = ['PythonParser', 'JSParser', 'CodeAnalyzer']
        for cls in expected_classes:
            assert cls in analysis.all_classes, f"Should find class {cls}"
            logger.info(f"  {cls}: {analysis.all_classes[cls]}")

        # 3. Test function registry
        logger.info("\n3. Testing function registry")

        # Show some functions
        sample_functions = list(analysis.all_functions.keys())[:5]
        for func in sample_functions:
            logger.info(f"  {func}: {analysis.all_functions[func]}")

        # 4. Test inheritance tree
        logger.info("\n4. Testing inheritance tree")

        if analysis.inheritance_tree:
            logger.info(f"Found {len(analysis.inheritance_tree)} inheritance relations:")
            for rel in analysis.inheritance_tree[:5]:
                logger.info(f"  {rel.child} extends {rel.parent}")
        else:
            logger.info("No inheritance relations found (dataclasses don't show extends)")

        # 5. Test class hierarchy
        logger.info("\n5. Testing class hierarchy")

        # Find a class with inheritance
        if analysis.inheritance_tree:
            first_rel = analysis.inheritance_tree[0]
            hierarchy = analyzer.get_class_hierarchy(analysis, first_rel.child)
            logger.info(f"Hierarchy for {first_rel.child}:")
            logger.info(f"  Ancestors: {hierarchy['ancestors']}")
            logger.info(f"  Descendants: {hierarchy['descendants']}")
            logger.info(f"  Siblings: {hierarchy['siblings']}")

        # 6. Test API endpoint finding
        logger.info("\n6. Testing API endpoint search")

        # Check if there are any endpoints (project may not have them)
        all_endpoints = analyzer.find_api_endpoints(analysis)
        logger.info(f"Total API endpoints found: {len(all_endpoints)}")

        # 7. Test function callers
        logger.info("\n7. Testing function caller search")

        # Find callers of a common function
        if analysis.all_functions:
            sample_func = list(analysis.all_functions.keys())[0]
            callers = analyzer.get_function_callers(analysis, sample_func)
            logger.info(f"Callers of '{sample_func}': {len(callers)}")
            for caller in callers[:3]:
                logger.info(f"  {caller.caller_file}:{caller.line_number}")

        # 8. Test file discovery
        logger.info("\n8. Testing file discovery")

        files = analyzer._discover_files(str(project_root))
        py_files = [f for f in files if f.endswith('.py')]
        js_files = [f for f in files if f.endswith(('.js', '.ts', '.jsx', '.tsx'))]

        logger.info(f"Python files: {len(py_files)}")
        logger.info(f"JS/TS files: {len(js_files)}")
        assert len(py_files) > 0, "Should find Python files"

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

        return True

    except AssertionError as e:
        logger.error(f"\nAssertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_code_analyzer()
    sys.exit(0 if success else 1)
