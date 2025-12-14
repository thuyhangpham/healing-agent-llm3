"""
Test Suite for Healing Components

Unit tests for healing functionality including
error handling, code patching, and LLM integration.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healing.error_handler import ErrorHandler
from healing.code_patcher import CodePatcher
from healing.llm_client import LLMClient
from healing.validator import CodeValidator
from healing.metrics import HealingMetrics


class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_capture_error_not_implemented(self):
        """Test that capture_error raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.error_handler.capture_error(Exception("test"), "test_agent")


class TestCodePatcher(unittest.TestCase):
    """Test cases for CodePatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.code_patcher = CodePatcher()
    
    def test_apply_fix_not_implemented(self):
        """Test that apply_fix raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.code_patcher.apply_fix("test.py", "test code")


class TestLLMClient(unittest.TestCase):
    """Test cases for LLMClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_client = LLMClient()
    
    def test_generate_code_fix_not_implemented(self):
        """Test that generate_code_fix raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.llm_client.generate_code_fix({"error": "test"})


class TestCodeValidator(unittest.TestCase):
    """Test cases for CodeValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CodeValidator()
    
    def test_validate_syntax_not_implemented(self):
        """Test that validate_syntax raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.validator.validate_syntax("test code")


class TestHealingMetrics(unittest.TestCase):
    """Test cases for HealingMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = HealingMetrics()
    
    def test_record_success_not_implemented(self):
        """Test that record_success raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.metrics.record_success(30.5)


if __name__ == "__main__":
    unittest.main()