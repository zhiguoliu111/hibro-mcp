#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Python code parser
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.parsers.python_parser import (
    PythonParser, ParsedFile, ClassInfo, FunctionInfo, ImportInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Sample Python code for testing
SAMPLE_CODE = '''
"""Module docstring for testing."""

import os
import sys
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

@dataclass
class Person:
    """A person class."""
    name: str
    age: int

    def greet(self) -> str:
        """Greet the person."""
        return f"Hello, {self.name}!"

class Employee(Person):
    """An employee class inheriting from Person."""

    def __init__(self, name: str, age: int, department: str):
        super().__init__(name, age)
        self.department = department

    @property
    def info(self) -> Dict[str, Any]:
        """Get employee info."""
        return {"name": self.name, "age": self.age, "department": self.department}

async def fetch_data(url: str, timeout: int = 30) -> Optional[Dict]:
    """Fetch data from URL asynchronously."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            return await response.json()

def process_items(items: List[str], *, reverse: bool = False) -> List[str]:
    """Process a list of items."""
    result = sorted(items)
    if reverse:
        result = list(reversed(result))
    return result

def complex_function(
    a: int,
    b: str = "default",
    *args: Any,
    c: float = 1.0,
    **kwargs: Dict
) -> Tuple[int, str]:
    """A function with complex signature."""
    return (a, b)
'''


def test_python_parser():
    """Test Python code parser"""

    logger.info("=" * 60)
    logger.info("Starting Python parser test")
    logger.info("=" * 60)

    try:
        parser = PythonParser()

        # 1. Test parsing source code
        logger.info("\n1. Testing source code parsing")

        result = parser.parse_source(SAMPLE_CODE, "test_module.py")

        logger.info(f"Module docstring: {result.module_docstring[:50]}..." if result.module_docstring else "No module docstring")
        logger.info(f"Classes found: {len(result.classes)}")
        logger.info(f"Functions found: {len(result.functions)}")
        logger.info(f"Imports found: {len(result.imports)}")
        logger.info(f"Syntax errors: {len(result.syntax_errors)}")

        assert len(result.syntax_errors) == 0, "Should have no syntax errors"
        assert result.module_docstring is not None, "Should have module docstring"

        # 2. Test class extraction
        logger.info("\n2. Testing class extraction")

        assert len(result.classes) == 2, f"Should find 2 classes, found {len(result.classes)}"

        # Check Person class
        person_class = next((c for c in result.classes if c.name == "Person"), None)
        assert person_class is not None, "Should find Person class"
        logger.info(f"Person class: line {person_class.line_number}, methods: {person_class.methods}")
        assert "greet" in person_class.methods, "Person should have greet method"

        # Check Employee class
        employee_class = next((c for c in result.classes if c.name == "Employee"), None)
        assert employee_class is not None, "Should find Employee class"
        logger.info(f"Employee class: bases={employee_class.bases}, methods={employee_class.methods}")
        assert "Person" in employee_class.bases, "Employee should inherit from Person"
        assert "__init__" in employee_class.methods, "Employee should have __init__ method"
        assert "info" in employee_class.methods, "Employee should have info property"

        # 3. Test function extraction
        logger.info("\n3. Testing function extraction")

        assert len(result.functions) == 3, f"Should find 3 functions, found {len(result.functions)}"

        # Check fetch_data function (async)
        fetch_func = next((f for f in result.functions if f.name == "fetch_data"), None)
        assert fetch_func is not None, "Should find fetch_data function"
        logger.info(f"fetch_data: async={fetch_func.is_async}, return={fetch_func.return_type}")
        assert fetch_func.is_async, "fetch_data should be async"
        assert "Optional[Dict]" in fetch_func.return_type, "Should have correct return type"

        # Check process_items function
        process_func = next((f for f in result.functions if f.name == "process_items"), None)
        assert process_func is not None, "Should find process_items function"
        logger.info(f"process_items: params={len(process_func.parameters)}")
        assert len(process_func.parameters) >= 2, "Should have parameters"

        # Check complex_function
        complex_func = next((f for f in result.functions if f.name == "complex_function"), None)
        assert complex_func is not None, "Should find complex_function"
        logger.info(f"complex_function: params={len(complex_func.parameters)}")
        # Check for *args and **kwargs
        param_names = [p["name"] for p in complex_func.parameters]
        assert any("*args" in name for name in param_names), "Should have *args"
        assert any("**kwargs" in name for name in param_names), "Should have **kwargs"

        # 4. Test import extraction
        logger.info("\n4. Testing import extraction")

        assert len(result.imports) >= 4, f"Should find at least 4 imports, found {len(result.imports)}"

        for imp in result.imports:
            logger.info(f"  Import: module={imp.module}, names={imp.names}, is_from={imp.is_from_import}")

        # Check from import
        typing_import = next((i for i in result.imports if i.module == "typing"), None)
        assert typing_import is not None, "Should find typing import"
        assert "List" in typing_import.names, "Should import List from typing"
        assert "Dict" in typing_import.names, "Should import Dict from typing"

        # 5. Test function call extraction
        logger.info("\n5. Testing function call extraction")

        calls = parser.get_function_calls(SAMPLE_CODE)
        logger.info(f"Found {len(calls)} function calls")

        # Should have calls like super().__init__, sorted(), reversed(), list()
        call_names = [c["function"] for c in calls]
        logger.info(f"Call functions: {call_names}")
        assert "__init__" in call_names, "Should call __init__"
        assert "sorted" in call_names, "Should call sorted"

        # 6. Test parsing a real file
        logger.info("\n6. Testing real file parsing")

        # Parse the parser itself
        parser_path = str(project_root / "src" / "hibro" / "parsers" / "python_parser.py")
        real_result = parser.parse_file(parser_path)

        logger.info(f"Parsed python_parser.py:")
        logger.info(f"  - Classes: {len(real_result.classes)}")
        logger.info(f"  - Functions: {len(real_result.functions)}")
        logger.info(f"  - Imports: {len(real_result.imports)}")
        logger.info(f"  - Syntax errors: {len(real_result.syntax_errors)}")

        assert len(real_result.syntax_errors) == 0, "Should parse without errors"
        assert len(real_result.classes) >= 4, f"Should find at least 4 classes (ParsedFile, ClassInfo, etc.), found {len(real_result.classes)}"
        # Note: python_parser.py only has class methods, not standalone functions
        # Check total methods across all classes instead
        total_methods = sum(len(c.methods) for c in real_result.classes)
        logger.info(f"  - Total methods across classes: {total_methods}")
        assert total_methods >= 10, f"Should find at least 10 methods total, found {total_methods}"

        # 7. Test dependency analysis
        logger.info("\n7. Testing dependency analysis")

        deps = parser.analyze_dependencies(parser_path)
        logger.info(f"Dependencies:")
        logger.info(f"  - stdlib: {deps['stdlib'][:5]}...")
        logger.info(f"  - third_party: {deps['third_party']}")
        logger.info(f"  - local: {deps['local']}")
        logger.info(f"  - total: {len(deps['all'])}")

        assert len(deps['stdlib']) > 0, "Should have stdlib imports"
        assert 'ast' in deps['stdlib'], "Should import ast"
        assert 'logging' in deps['stdlib'], "Should import logging"

        # 8. Test syntax error handling
        logger.info("\n8. Testing syntax error handling")

        bad_code = "def broken(\n  # missing closing paren"
        bad_result = parser.parse_source(bad_code, "broken.py")

        assert len(bad_result.syntax_errors) > 0, "Should detect syntax errors"
        logger.info(f"Detected syntax error: {bad_result.syntax_errors[0]}")

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
    success = test_python_parser()
    sys.exit(0 if success else 1)
