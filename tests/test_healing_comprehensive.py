"""
Comprehensive Test Suite for Healing System

Unit tests for healing functionality including
error handling, code patching, LLM integration, and metrics.
"""

import asyncio
import ast
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.healing_agent import HealingAgent, HealingStatus, ErrorContext, ErrorSeverity
from core.llm_client import LLMClient, LLMResponse
from core.error_detector import ErrorDetector, ErrorCategory
from core.code_patcher import CodePatcher, PatchStatus
from core.healing_metrics import HealingMetrics, HealingEvent
from healing.validator import CodeValidator, ValidationResult, ValidationStatus


class TestHealingAgent(unittest.TestCase):
    """Test cases for HealingAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'auto_heal_enabled': True,
            'max_healing_attempts': 3,
            'mttr_target_seconds': 60,
            'success_rate_target': 0.8,
            'backup_enabled': True
        }
        self.healing_agent = HealingAgent("test_healing_agent", self.config)
    
    def test_initialization(self):
        """Test healing agent initialization."""
        self.assertEqual(self.healing_agent.agent_id, "test_healing_agent")
        self.assertTrue(self.healing_agent.auto_heal_enabled)
        self.assertEqual(self.healing_agent.max_healing_attempts, 3)
        self.assertEqual(self.healing_agent.mttr_target, 60)
        self.assertEqual(self.healing_agent.success_rate_target, 0.8)
        self.assertTrue(self.healing_agent.backup_enabled)
        self.assertEqual(self.healing_agent.status, HealingStatus.IDLE)
    
    def test_create_error_context(self):
        """Test error context creation."""
        error_data = {
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'traceback': 'Traceback...',
            'agent_name': 'test_agent',
            'function_name': 'test_function',
            'file_path': '/test/file.py',
            'line_number': 10,
            'severity': 'medium'
        }
        
        error_context = self.healing_agent._create_error_context(error_data)
        
        self.assertIsInstance(error_context, ErrorContext)
        self.assertEqual(error_context.error_type, 'ValueError')
        self.assertEqual(error_context.error_message, 'Test error')
        self.assertEqual(error_context.agent_name, 'test_agent')
        self.assertEqual(error_context.severity, ErrorSeverity.MEDIUM)
    
    def test_max_healing_attempts_reached(self):
        """Test handling of max healing attempts."""
        error_data = {
            'error_type': 'TestError',
            'error_message': 'Test error',
            'healing_attempts': 3,  # Max attempts reached
            'agent_name': 'test_agent'
        }
        
        error_context = self.healing_agent._create_error_context(error_data)
        
        # Should not attempt healing if max attempts reached
        self.assertEqual(error_context.healing_attempts, 3)
        self.assertEqual(error_context.max_attempts, 3)
    
    @patch('agents.healing_agent.LLMClient')
    @patch('agents.healing_agent.ErrorDetector')
    @patch('agents.healing_agent.CodePatcher')
    @patch('agents.healing_agent.HealingMetrics')
    async def test_handle_error_success(self, mock_metrics, mock_patcher, mock_detector, mock_llm):
        """Test successful error handling."""
        # Setup mocks
        mock_llm_instance = AsyncMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.initialize.return_value = True
        
        mock_detector_instance = AsyncMock()
        mock_detector.return_value = mock_detector_instance
        mock_detector_instance.analyze_error.return_value = {
            'repairable': True,
            'error_category': 'web_scraping_failure',
            'confidence': 0.9
        }
        
        mock_patcher_instance = AsyncMock()
        mock_patcher.return_value = mock_patcher_instance
        mock_patcher_instance.apply_fix.return_value = True
        mock_patcher_instance.validate_fix.return_value = True
        
        mock_metrics_instance = AsyncMock()
        mock_metrics.return_value = mock_metrics_instance
        
        # Initialize healing agent
        await self.healing_agent.initialize()
        
        # Test error handling
        error_data = {
            'error_type': 'TestError',
            'error_message': 'Test error',
            'agent_name': 'test_agent',
            'file_path': '/test/file.py',
            'function_name': 'test_function'
        }
        
        result = await self.healing_agent.handle_error(error_data)
        
        self.assertTrue(result.success)
        self.assertEqual(result.status, HealingStatus.COMPLETED)
        self.assertGreater(result.time_to_repair, 0)
    
    async def test_handle_error_disabled(self):
        """Test error handling when auto-heal is disabled."""
        self.healing_agent.auto_heal_enabled = False
        
        error_data = {
            'error_type': 'TestError',
            'error_message': 'Test error',
            'agent_name': 'test_agent'
        }
        
        result = await self.healing_agent.handle_error(error_data)
        
        self.assertFalse(result.success)
        self.assertEqual(result.status, HealingStatus.FAILED)
        self.assertIn('Auto-healing disabled', result.fix_description)


class TestLLMClient(unittest.TestCase):
    """Test cases for LLMClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'base_url': 'http://localhost:11434',
            'model': 'llama3',
            'temperature': 0.3,
            'max_tokens': 4096,
            'timeout': 30
        }
        self.llm_client = LLMClient(self.config)
    
    def test_initialization(self):
        """Test LLM client initialization."""
        self.assertEqual(self.llm_client.config.base_url, 'http://localhost:11434')
        self.assertEqual(self.llm_client.config.model, 'llama3')
        self.assertEqual(self.llm_client.config.temperature, 0.3)
        self.assertEqual(self.llm_client.config.max_tokens, 4096)
    
    @patch('aiohttp.ClientSession.get')
    async def test_test_connection_success(self, mock_get):
        """Test successful connection test."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'models': [{'name': 'llama3'}]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await self.llm_client._test_connection()
        self.assertTrue(result)
    
    @patch('aiohttp.ClientSession.get')
    async def test_test_connection_failure(self, mock_get):
        """Test failed connection test."""
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await self.llm_client._test_connection()
        self.assertFalse(result)
    
    def test_extract_code_with_python_block(self):
        """Test code extraction from Python code block."""
        response = """Here's the fix:
```python
def fixed_function():
    return "fixed"
```
This should work."""
        
        code = self.llm_client._extract_code(response)
        self.assertEqual(code.strip(), 'def fixed_function():\n    return "fixed"')
    
    def test_extract_code_with_generic_block(self):
        """Test code extraction from generic code block."""
        response = """Here's the fix:
```
def fixed_function():
    return "fixed"
```
This should work."""
        
        code = self.llm_client._extract_code(response)
        self.assertEqual(code.strip(), 'def fixed_function():\n    return "fixed"')
    
    def test_extract_code_no_block(self):
        """Test code extraction when no code block present."""
        response = "def fixed_function():\n    return 'fixed'"
        
        code = self.llm_client._extract_code(response)
        self.assertEqual(code.strip(), "def fixed_function():\n    return 'fixed'")
    
    def test_extract_code_none(self):
        """Test code extraction when no code found."""
        response = "Here's some text but no code."
        
        code = self.llm_client._extract_code(response)
        self.assertIsNone(code)


class TestErrorDetector(unittest.TestCase):
    """Test cases for ErrorDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'similarity_threshold': 0.7,
            'pattern_min_frequency': 3,
            'html_diff_threshold': 0.8
        }
        self.error_detector = ErrorDetector(self.config)
    
    def test_categorize_error_css_selector(self):
        """Test CSS selector error categorization."""
        error_message = "CSS selector '.content' not found"
        traceback = "CSS selector failed"
        error_type = "SelectorError"
        
        category = self.error_detector._categorize_error(error_message, traceback, error_type)
        self.assertEqual(category, ErrorCategory.CSS_SELECTOR_FAILURE)
    
    def test_categorize_error_network(self):
        """Test network error categorization."""
        error_message = "Connection refused"
        traceback = "Network error occurred"
        error_type = "ConnectionError"
        
        category = self.error_detector._categorize_error(error_message, traceback, error_type)
        self.assertEqual(category, ErrorCategory.NETWORK_ERROR)
    
    def test_categorize_error_unknown(self):
        """Test unknown error categorization."""
        error_message = "Unknown error occurred"
        traceback = "Something went wrong"
        error_type = "UnknownError"
        
        category = self.error_detector._categorize_error(error_message, traceback, error_type)
        self.assertEqual(category, ErrorCategory.UNKNOWN_ERROR)
    
    def test_determine_severity_critical(self):
        """Test critical severity determination."""
        category = ErrorCategory.AUTHENTICATION_ERROR
        error_context = MagicMock()
        
        severity = self.error_detector._determine_severity(category, error_context)
        self.assertEqual(severity.value, 4)  # CRITICAL
    
    def test_determine_severity_high(self):
        """Test high severity determination."""
        category = ErrorCategory.WEB_SCRAPING_FAILURE
        error_context = MagicMock()
        
        severity = self.error_detector._determine_severity(category, error_context)
        self.assertEqual(severity.value, 3)  # HIGH
    
    def test_assess_repairability_css_selector(self):
        """Test repairability assessment for CSS selector errors."""
        category = ErrorCategory.CSS_SELECTOR_FAILURE
        severity = MagicMock()
        html_diff = None
        pattern_matches = []
        
        repairable, confidence = self.error_detector._assess_repairability(
            category, severity, html_diff, pattern_matches
        )
        
        self.assertTrue(repairable)
        self.assertGreaterEqual(confidence, 0.8)
    
    def test_assess_repairability_network(self):
        """Test repairability assessment for network errors."""
        category = ErrorCategory.NETWORK_ERROR
        severity = MagicMock()
        html_diff = None
        pattern_matches = []
        
        repairable, confidence = self.error_detector._assess_repairability(
            category, severity, html_diff, pattern_matches
        )
        
        self.assertFalse(repairable)
        self.assertLessEqual(confidence, 0.5)


class TestCodePatcher(unittest.TestCase):
    """Test cases for CodePatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'backup_enabled': True,
            'validation_level': 'basic_execution',
            'auto_rollback': True,
            'max_patch_size': 100000,
            'test_timeout': 30
        }
        self.code_patcher = CodePatcher(self.config)
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_file.py')
        
        # Write test code
        with open(self.test_file, 'w') as f:
            f.write('def broken_function():\n    raise ValueError("broken")\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_inputs_success(self):
        """Test successful input validation."""
        fix_code = 'def fixed_function():\n    return "fixed"'
        
        result = asyncio.run(self.code_patcher._validate_inputs(self.test_file, fix_code))
        self.assertTrue(result)
    
    def test_validate_inputs_file_not_found(self):
        """Test input validation with non-existent file."""
        fix_code = 'def fixed_function():\n    return "fixed"'
        
        result = asyncio.run(self.code_patcher._validate_inputs('/non/existent/file.py', fix_code))
        self.assertFalse(result)
    
    def test_validate_inputs_patch_too_large(self):
        """Test input validation with oversized patch."""
        fix_code = 'x' * 200000  # Larger than max_patch_size
        
        result = asyncio.run(self.code_patcher._validate_inputs(self.test_file, fix_code))
        self.assertFalse(result)
    
    def test_validate_patch_syntax_valid(self):
        """Test patch validation with valid syntax."""
        fix_code = 'def fixed_function():\n    return "fixed"'
        
        result = asyncio.run(self.code_patcher._validate_patch(fix_code, self.test_file))
        self.assertTrue(result.is_valid)
        self.assertTrue(result.syntax_valid)
    
    def test_validate_patch_syntax_invalid(self):
        """Test patch validation with invalid syntax."""
        fix_code = 'def fixed_function(\n    return "fixed"'  # Missing closing parenthesis
        
        result = asyncio.run(self.code_patcher._validate_patch(fix_code, self.test_file))
        self.assertFalse(result.is_valid)
        self.assertFalse(result.syntax_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_get_module_name_from_path(self):
        """Test module name extraction from file path."""
        # Test relative path
        module_name = self.code_patcher._get_module_name_from_path('agents/test_agent.py')
        self.assertEqual(module_name, 'agents.test_agent')
        
        # Test absolute path
        abs_path = os.path.abspath('agents/test_agent.py')
        module_name = self.code_patcher._get_module_name_from_path(abs_path)
        self.assertTrue(module_name.endswith('agents.test_agent'))
    
    @patch('shutil.copy2')
    @patch('pathlib.Path.mkdir')
    def test_create_backup(self, mock_mkdir, mock_copy):
        """Test backup creation."""
        mock_mkdir.return_value = None
        mock_copy.return_value = None
        
        result = asyncio.run(self.code_patcher._create_backup(self.test_file, 'test_patch'))
        
        self.assertTrue(result)  # Should return True on successful backup
        mock_copy.assert_called_once()


class TestHealingMetrics(unittest.TestCase):
    """Test cases for HealingMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'metrics_file': 'test_metrics.json',
            'research_data_dir': 'test_research',
            'max_events': 1000,
            'export_interval': 3600,
            'retention_days': 30,
            'mttr_target': 60.0,
            'success_rate_target': 0.8
        }
        self.metrics = HealingMetrics(self.config)
    
    def test_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.mttr_target, 60.0)
        self.assertEqual(self.metrics.success_rate_target, 0.8)
        self.assertEqual(self.metrics.max_events, 1000)
        self.assertEqual(self.metrics.export_interval, 3600)
        self.assertEqual(self.metrics.retention_days, 30)
    
    async def test_record_healing_event_success(self):
        """Test recording successful healing event."""
        await self.metrics.record_healing_event(
            error_id='test_error_1',
            success=True,
            time_to_repair=45.5,
            error_type='TestError',
            agent_name='test_agent',
            error_category='web_scraping_failure',
            severity=2,
            confidence=0.9
        )
        
        self.assertEqual(len(self.metrics.healing_events), 1)
        event = self.metrics.healing_events[0]
        self.assertEqual(event.error_id, 'test_error_1')
        self.assertTrue(event.success)
        self.assertEqual(event.time_to_repair, 45.5)
        self.assertEqual(event.agent_name, 'test_agent')
    
    async def test_record_healing_event_failure(self):
        """Test recording failed healing event."""
        await self.metrics.record_healing_event(
            error_id='test_error_2',
            success=False,
            time_to_repair=120.0,
            error_type='TestError',
            agent_name='test_agent',
            error_category='network_error',
            severity=3,
            confidence=0.3
        )
        
        self.assertEqual(len(self.metrics.healing_events), 1)
        event = self.metrics.healing_events[0]
        self.assertEqual(event.error_id, 'test_error_2')
        self.assertFalse(event.success)
        self.assertEqual(event.time_to_repair, 120.0)
    
    def test_get_success_rate_all_time(self):
        """Test success rate calculation for all time."""
        # Add test events
        self.metrics.healing_events = [
            HealingEvent(
                event_id='1', timestamp=datetime.now(), error_id='1', error_type='Test',
                agent_name='test', success=True, time_to_repair=30.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.9, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='2', timestamp=datetime.now(), error_id='2', error_type='Test',
                agent_name='test', success=False, time_to_repair=0.0,
                fix_applied=False, backup_created=False, validation_passed=False,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.0, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='3', timestamp=datetime.now(), error_id='3', error_type='Test',
                agent_name='test', success=True, time_to_repair=45.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.8, mttr_target=60.0, success_rate_target=0.8
            )
        ]
        
        success_rate = self.metrics.get_success_rate()
        self.assertEqual(success_rate, 66.67)  # 2 out of 3 successful
    
    def test_get_success_rate_time_window(self):
        """Test success rate calculation with time window."""
        # Add test events with different timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=25)  # More than 24 hours ago
        
        self.metrics.healing_events = [
            HealingEvent(
                event_id='1', timestamp=old_time, error_id='1', error_type='Test',
                agent_name='test', success=True, time_to_repair=30.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.9, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='2', timestamp=now, error_id='2', error_type='Test',
                agent_name='test', success=False, time_to_repair=0.0,
                fix_applied=False, backup_created=False, validation_passed=False,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.0, mttr_target=60.0, success_rate_target=0.8
            )
        ]
        
        # With 24-hour window, only the recent event should count
        success_rate = self.metrics.get_success_rate(time_window=24)
        self.assertEqual(success_rate, 0.0)  # 0 out of 1 successful in last 24h
    
    def test_get_average_mttr(self):
        """Test MTTR calculation."""
        # Add test events
        self.metrics.healing_events = [
            HealingEvent(
                event_id='1', timestamp=datetime.now(), error_id='1', error_type='Test',
                agent_name='test', success=True, time_to_repair=30.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.9, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='2', timestamp=datetime.now(), error_id='2', error_type='Test',
                agent_name='test', success=True, time_to_repair=60.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.8, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='3', timestamp=datetime.now(), error_id='3', error_type='Test',
                agent_name='test', success=False, time_to_repair=0.0,
                fix_applied=False, backup_created=False, validation_passed=False,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.0, mttr_target=60.0, success_rate_target=0.8
            )
        ]
        
        avg_mttr = self.metrics.get_average_mttr()
        self.assertEqual(avg_mttr, 45.0)  # (30 + 60) / 2 successful events
    
    def test_get_agent_performance(self):
        """Test agent performance calculation."""
        # Add test events for different agents
        self.metrics.healing_events = [
            HealingEvent(
                event_id='1', timestamp=datetime.now(), error_id='1', error_type='Test',
                agent_name='agent1', success=True, time_to_repair=30.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.9, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='2', timestamp=datetime.now(), error_id='2', error_type='Test',
                agent_name='agent1', success=False, time_to_repair=0.0,
                fix_applied=False, backup_created=False, validation_passed=False,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.0, mttr_target=60.0, success_rate_target=0.8
            ),
            HealingEvent(
                event_id='3', timestamp=datetime.now(), error_id='3', error_type='Test',
                agent_name='agent2', success=True, time_to_repair=45.0,
                fix_applied=True, backup_created=True, validation_passed=True,
                rollback_performed=False, error_category='test', severity=1,
                confidence=0.8, mttr_target=60.0, success_rate_target=0.8
            )
        ]
        
        performance = self.metrics.get_agent_performance()
        
        # Check agent1 performance
        agent1_stats = performance['agent1']
        self.assertEqual(agent1_stats['total'], 2)
        self.assertEqual(agent1_stats['success'], 1)
        self.assertEqual(agent1_stats['success_rate'], 50.0)
        self.assertEqual(agent1_stats['avg_mttr'], 30.0)
        
        # Check agent2 performance
        agent2_stats = performance['agent2']
        self.assertEqual(agent2_stats['total'], 1)
        self.assertEqual(agent2_stats['success'], 1)
        self.assertEqual(agent2_stats['success_rate'], 100.0)
        self.assertEqual(agent2_stats['avg_mttr'], 45.0)


class TestCodeValidator(unittest.TestCase):
    """Test cases for CodeValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'validation_level': 'full_validation',
            'timeout_seconds': 30,
            'enable_security_checks': True,
            'enable_logic_validation': True
        }
        self.validator = CodeValidator(self.config)
    
    async def test_validate_code_valid(self):
        """Test validation of valid code."""
        valid_code = '''
def test_function():
    """A simple test function."""
    return "Hello, World!"

if __name__ == "__main__":
    print(test_function())
'''
        
        result = await self.validator.validate_code(valid_code, 'test.py')
        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertTrue(result.syntax_valid)
        self.assertGreaterEqual(result.confidence_score, 0.8)
    
    async def test_validate_code_syntax_error(self):
        """Test validation of code with syntax error."""
        invalid_code = '''
def test_function(
    """Missing closing parenthesis."""
"""
    
    return "Hello"
'''
        
        result = await self.validator.validate_code(invalid_code, 'test.py')
        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertFalse(result.syntax_valid)
        self.assertGreater(len(result.issues), 0)
        
        # Check for syntax error issue
        syntax_issues = [issue for issue in result.issues if issue.issue_type == 'syntax_error']
        self.assertGreater(len(syntax_issues), 0)
    
    async def test_validate_code_security_warning(self):
        """Test validation of code with security issues."""
        code_with_security_issues = '''
import os

def test_function():
    eval("print('dangerous')")
    os.system("rm -rf /")
    return "executed"
'''
        
        result = await self.validator.validate_code(code_with_security_issues, 'test.py')
        
        # Should have security warnings
        security_issues = [issue for issue in result.issues if issue.issue_type == 'security_warning']
        self.assertGreater(len(security_issues), 0)
        
        # Check for specific dangerous patterns
        dangerous_patterns = ['eval', 'os.system']
        for pattern in dangerous_patterns:
            found = any(pattern in issue.message.lower() for issue in security_issues)
            self.assertTrue(found, f"Should detect dangerous pattern: {pattern}")
    
    async def test_validate_code_complexity_warning(self):
        """Test validation of code with high complexity."""
        complex_code = '''
def complex_function(x):
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    if x > 40:
                        return "very high"
                    else:
                        return "high"
                else:
                    return "medium-high"
            else:
                return "medium"
        else:
            return "low"
    else:
        return "negative"
'''
        
        result = await self.validator.validate_code(complex_code, 'test.py')
        
        # Should have complexity warning
        complexity_issues = [issue for issue in result.issues if issue.issue_type == 'complexity_warning']
        self.assertGreater(len(complexity_issues), 0)
    
    def test_calculate_cyclomatic_complexity(self):
        """Test cyclomatic complexity calculation."""
        # Simple function
        simple_code = '''
def simple_function():
    if True:
        return True
'''
        
        tree = ast.parse(simple_code)
        func_node = tree.body[0]
        complexity = self.validator._calculate_cyclomatic_complexity(func_node)
        self.assertEqual(complexity, 2)  # Base 1 + 1 if statement
        
        # Complex function
        complex_code = '''
def complex_function(x):
    if x > 0:
        for i in range(10):
            if i > 5:
                try:
                    result = i / x
                except:
                    result = 0
            else:
                result = i
    else:
        result = 0
    return result
'''
        
        tree = ast.parse(complex_code)
        func_node = tree.body[0]
        complexity = self.validator._calculate_cyclomatic_complexity(func_node)
        self.assertGreater(complexity, 5)  # Should be higher due to multiple control structures


class TestIntegration(unittest.TestCase):
    """Integration tests for healing system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        self.data_dir = os.path.join(self.temp_dir, 'data')
        
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test configuration
        self.healing_config = {
            'healing_agent': {
                'auto_heal_enabled': True,
                'max_healing_attempts': 3,
                'mttr_target_seconds': 60,
                'success_rate_target': 0.8,
                'backup_enabled': True
            },
            'llm': {
                'base_url': 'http://localhost:11434',
                'model': 'llama3',
                'temperature': 0.3,
                'max_tokens': 4096
            },
            'metrics': {
                'metrics_file': os.path.join(self.data_dir, 'test_metrics.json'),
                'research_data_dir': os.path.join(self.data_dir, 'research')
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('core.llm_client.aiohttp.ClientSession')
    async def test_healing_workflow_integration(self, mock_session):
        """Test complete healing workflow integration."""
        # Mock LLM responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'models': [{'name': 'llama3'}]
        })
        mock_session.return_value.__aenter__.return_value = mock_response
        
        # Initialize components
        llm_client = LLMClient(self.healing_config['llm'])
        error_detector = ErrorDetector()
        code_patcher = CodePatcher({'backup_dir': self.data_dir})
        metrics = HealingMetrics(self.healing_config['metrics'])
        
        # Initialize components
        await llm_client.initialize()
        await error_detector.initialize()
        await code_patcher.initialize()
        await metrics.initialize()
        
        # Create test error context
        error_context = ErrorContext(
            error_id='test_integration',
            timestamp=datetime.now(),
            error_type='ValueError',
            error_message='CSS selector not found',
            traceback_str='Traceback...',
            agent_name='test_agent',
            function_name='scrape_data',
            file_path=os.path.join(self.temp_dir, 'test_agent.py'),
            line_number=10,
            severity=ErrorSeverity.MEDIUM
        )
        
        # Test error analysis
        analysis = await error_detector.analyze_error(error_context)
        self.assertIsInstance(analysis, dict)
        self.assertIn('repairable', analysis)
        
        # Test metrics recording
        await metrics.record_healing_event(
            error_id=error_context.error_id,
            success=True,
            time_to_repair=45.5,
            error_type=error_context.error_type,
            agent_name=error_context.agent_name,
            error_category='web_scraping_failure'
        )
        
        # Verify metrics
        success_rate = metrics.get_success_rate()
        self.assertGreater(success_rate, 0)
        
        avg_mttr = metrics.get_average_mttr()
        self.assertEqual(avg_mttr, 45.5)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)