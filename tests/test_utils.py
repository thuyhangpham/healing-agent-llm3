"""
Test Suite for Utilities

Unit tests for utility functions including
logging, configuration, and file operations.
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import StructuredLogger, StructuredFormatter
from utils.config import Settings
from utils.file_utils import FileUtils


class TestStructuredLogger(unittest.TestCase):
    """Test cases for StructuredLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = StructuredLogger("test_logger", "DEBUG")
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.logger.name, "test_logger")
        self.assertEqual(self.logger.logger.level, 10)  # DEBUG level
    
    def test_log_methods_exist(self):
        """Test that all log methods exist."""
        self.assertTrue(hasattr(self.logger, 'info'))
        self.assertTrue(hasattr(self.logger, 'error'))
        self.assertTrue(hasattr(self.logger, 'warning'))
        self.assertTrue(hasattr(self.logger, 'debug'))


class TestStructuredFormatter(unittest.TestCase):
    """Test cases for StructuredFormatter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = StructuredFormatter()
    
    def test_formatter_creates_json(self):
        """Test that formatter creates JSON output."""
        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        
        formatted = self.formatter.format(record)
        
        # Should be valid JSON
        import json
        parsed = json.loads(formatted)
        self.assertIn('timestamp', parsed)
        self.assertIn('level', parsed)
        self.assertIn('message', parsed)


class TestSettings(unittest.TestCase):
    """Test cases for Settings class."""
    
    def test_settings_initialization(self):
        """Test settings initialization."""
        settings = Settings()
        self.assertIsNotNone(settings.app_name)
        self.assertIsNotNone(settings.ollama_base_url)
        self.assertIsNotNone(settings.ollama_model)
    
    def test_settings_validation(self):
        """Test settings validation."""
        settings = Settings()
        result = settings.validate()
        self.assertTrue(result)


class TestFileUtils(unittest.TestCase):
    """Test cases for FileUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        self.test_content = "Test content for file operations"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        FileUtils.ensure_directory(new_dir)
        self.assertTrue(os.path.exists(new_dir))
    
    def test_write_and_read_file(self):
        """Test file write and read operations."""
        FileUtils.write_file(self.test_file, self.test_content)
        self.assertTrue(os.path.exists(self.test_file))
        
        content = FileUtils.read_file(self.test_file)
        self.assertEqual(content, self.test_content)
    
    def test_file_exists(self):
        """Test file existence check."""
        self.assertFalse(FileUtils.file_exists(self.test_file))
        
        FileUtils.write_file(self.test_file, self.test_content)
        self.assertTrue(FileUtils.file_exists(self.test_file))
    
    def test_write_and_read_json(self):
        """Test JSON write and read operations."""
        test_data = {"key": "value", "number": 42}
        json_file = self.test_file.replace('.txt', '.json')
        
        FileUtils.write_json(json_file, test_data)
        self.assertTrue(os.path.exists(json_file))
        
        read_data = FileUtils.read_json(json_file)
        self.assertEqual(read_data, test_data)
    
    def test_create_backup(self):
        """Test backup creation."""
        FileUtils.write_file(self.test_file, self.test_content)
        
        backup_path = FileUtils.create_backup(self.test_file)
        self.assertTrue(os.path.exists(backup_path))
        
        backup_content = FileUtils.read_file(backup_path)
        self.assertEqual(backup_content, self.test_content)


if __name__ == "__main__":
    unittest.main()