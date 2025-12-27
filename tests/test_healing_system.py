"""
Comprehensive Tests for Healing System

This module provides comprehensive test coverage for the self-healing system,
including unit tests, integration tests, and end-to-end healing workflows.

Test Coverage:
- Healing agent core functionality
- LLM client integration
- Error detection and analysis
- Code patching and hot-reload
- Metrics collection and reporting
- Orchestrator integration
- Chaos engineering scenarios
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.healing_agent import HealingAgent, ErrorContext, HealingResult, HealingStatus
from core.llm_client import LLMClient, LLMConfig, LLMResponse
from core.error_detector import ErrorDetector, ErrorCategory, ErrorSeverity
from core.code_patcher import CodePatcher, PatchStatus, ValidationResult
from core.healing_metrics import HealingMetrics, HealingEvent, MTTRCalculation


class TestHealingAgent(unittest.TestCase):
    """Test cases for HealingAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'auto_heal_enabled': True,
            'max_healing_attempts': 3,
            'mttr_target_seconds': 60,
            'success_rate_target': 0.8,
            'backup_enabled': True
        }
        self.healing_agent = HealingAgent("test_healing_agent", self.config)
    
    def test_healing_agent_initialization(self):
        """Test healing agent initialization"""
        self.assertEqual(self.healing_agent.agent_id, "test_healing_agent")
        self.assertTrue(self.healing_agent.auto_heal_enabled)
        self.assertEqual(self.healing_agent.max_healing_attempts, 3)
        self.assertEqual(self.healing_agent.mttr_target, 60)
        self.assertEqual(self.healing_agent.success_rate_target, 0.8)
        self.assertTrue(self.healing_agent.backup_enabled)
    
    def test_create_error_context(self):
        """Test error context creation"""
        error_data = {
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'agent_name': 'test_agent',
            'function_name': 'test_function',
            'file_path': '/test/file.py',
            'line_number': 10,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        error_context = self.healing_agent._create_error_context(error_data)
        
        self.assertIsInstance(error_context, ErrorContext)
        self.assertEqual(error_context.error_type, 'ValueError')
        self.assertEqual(error_context.error_message, 'Test error')
        self.assertEqual(error_context.agent_name, 'test_agent')
        self.assertEqual(error_context.function_name, 'test_function')
        self.assertEqual(error_context.file_path, '/test/file.py')
        self.assertEqual(error_context.line_number, 10)
    
    def test_create_healing_result(self):
        """Test healing result creation"""
        error_context = ErrorContext(
            error_id='test_error',
            timestamp=datetime.now(),
            error_type='ValueError',
            error_message='Test error',
            traceback_str='Test traceback',
            agent_name='test_agent',
            function_name='test_function',
            file_path='/test/file.py',
            line_number=10
        )
        
        result = self.healing_agent._create_result(
            error_context, HealingStatus.COMPLETED, True, "Test fix"
        )
        
        self.assertIsInstance(result, HealingResult)
        self.assertEqual(result.error_id, 'test_error')
        self.assertEqual(result.status, HealingStatus.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(result.fix_description, "Test fix")
    
    @patch('agents.healing_agent.LLMClient')
    @patch('agents.healing_agent.ErrorDetector')
    @patch('agents.healing_agent.CodePatcher')
    @patch('agents.healing_agent.HealingMetrics')
    async def test_handle_error_success(self, mock_metrics, mock_patcher, mock_detector, mock_llm):
        """Test successful error handling"""
        # Mock components
        mock_llm.return_value.generate_fix = AsyncMock(return_value="fixed code")
        mock_detector.return_value.analyze_error = AsyncMock(return_value={
            'repairable': True,
            'error_category': 'test_category',
            'confidence': 0.8
        })
        mock_patcher.return_value.apply_fix = AsyncMock(return_value=True)
        mock_patcher.return_value.validate_fix = AsyncMock(return_value=True)
        
        # Initialize healing agent
        await self.healing_agent.initialize()
        
        # Test error handling
        error_context = {
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'agent_name': 'test_agent',
            'function_name': 'test_function',
            'file_path': '/test/file.py',
            'line_number': 10,
            'timestamp': datetime.now().isoformat()
        }
        
        result = await self.healing_agent.handle_error(error_context)
        
        self.assertTrue(result.success)
        self.assertEqual(result.status, HealingStatus.COMPLETED)
        self.assertTrue(result.fix_applied)
        self.assertTrue(result.validation_passed)
    
    async def test_handle_error_disabled(self):
        """Test error handling when auto-heal is disabled"""
        self.healing_agent.auto_heal_enabled = False
        
        error_context = {
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'agent_name': 'test_agent',
            'function_name': 'test_function',
            'file_path': '/test/file.py',
            'line_number': 10,
            'timestamp': datetime.now().isoformat()
        }
        
        result = await self.healing_agent.handle_error(error_context)
        
        self.assertFalse(result.success)
        self.assertEqual(result.status, HealingStatus.FAILED)
        self.assertIn("Auto-healing disabled", result.fix_description)


class TestLLMClient(unittest.TestCase):
    """Test cases for LLMClient"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'base_url': 'http://localhost:11434',
            'model': 'llama3',
            'temperature': 0.3,
            'max_tokens': 4096,
            'timeout': 30,
            'max_retries': 3
        }
        self.llm_client = LLMClient(self.config)
    
    def test_llm_client_initialization(self):
        """Test LLM client initialization"""
        self.assertEqual(self.llm_client.config.base_url, 'http://localhost:11434')
        self.assertEqual(self.llm_client.config.model, 'llama3')
        self.assertEqual(self.llm_client.config.temperature, 0.3)
        self.assertEqual(self.llm_client.config.max_tokens, 4096)
        self.assertEqual(self.llm_client.config.timeout, 30)
        self.assertEqual(self.llm_client.config.max_retries, 3)
    
    def test_extract_code_from_response(self):
        """Test code extraction from LLM response"""
        response_with_code = """
        Here is the fixed code:
        ```python
        def fixed_function():
            return "fixed"
        ```
        This should work.
        """
        
        extracted_code = self.llm_client._extract_code(response_with_code)
        
        self.assertIn('def fixed_function():', extracted_code)
        self.assertIn('return "fixed"', extracted_code)
    
    def test_extract_code_no_blocks(self):
        """Test code extraction when no code blocks are present"""
        response_no_blocks = "def simple_function():\n    return 'test'"
        
        extracted_code = self.llm_client._extract_code(response_no_blocks)
        
        self.assertEqual(extracted_code, "def simple_function():\n    return 'test'")
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate_response_success(self, mock_post):
        """Test successful LLM response generation"""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'response': 'Generated response',
            'model': 'llama3',
            'done': True,
            'total_duration': 1000000000,
            'prompt_eval_count': 10,
            'eval_count': 20
        })
        
        mock_post.return_value.__aenter__.return_value = mock_post.return_value.__aexit__.return_value
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await self.llm_client.initialize()
        
        response = await self.llm_client._generate_response("Test prompt")
        
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, 'Generated response')
        self.assertEqual(response.model, 'llama3')
        self.assertTrue(response.done)
        self.assertEqual(response.prompt_eval_count, 10)
        self.assertEqual(response.eval_count, 20)


class TestErrorDetector(unittest.TestCase):
    """Test cases for ErrorDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'similarity_threshold': 0.7,
            'pattern_min_frequency': 3,
            'html_diff_threshold': 0.8
        }
        self.error_detector = ErrorDetector(self.config)
    
    def test_error_detector_initialization(self):
        """Test error detector initialization"""
        self.assertEqual(self.error_detector.similarity_threshold, 0.7)
        self.assertEqual(self.error_detector.pattern_min_frequency, 3)
        self.assertEqual(self.error_detector.html_diff_threshold, 0.8)
        self.assertIsInstance(self.error_detector.error_patterns, dict)
        self.assertIsInstance(self.error_detector.error_history, list)
    
    def test_categorize_error_css_selector(self):
        """Test CSS selector error categorization"""
        error_message = "CSS selector 'div.test' not found"
        traceback_str = "SelectorNotFoundError: No element found"
        error_type = "SelectorNotFoundError"
        
        category = self.error_detector._categorize_error(error_message, traceback_str, error_type)
        
        self.assertEqual(category, ErrorCategory.CSS_SELECTOR_FAILURE)
    
    def test_categorize_error_network(self):
        """Test network error categorization"""
        error_message = "Connection refused"
        traceback_str = "ConnectionError: Failed to connect"
        error_type = "ConnectionError"
        
        category = self.error_detector._categorize_error(error_message, traceback_str, error_type)
        
        self.assertEqual(category, ErrorCategory.NETWORK_ERROR)
    
    def test_determine_severity_critical(self):
        """Test critical severity determination"""
        error_context = Mock()
        error_context.error_message = "Authentication failed"
        
        severity = self.error_detector._determine_severity(
            ErrorCategory.AUTHENTICATION_ERROR, error_context
        )
        
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
    
    def test_determine_severity_low(self):
        """Test low severity determination"""
        error_context = Mock()
        error_context.error_message = "Warning: deprecated feature used"
        
        severity = self.error_detector._determine_severity(
            ErrorCategory.SYNTAX_ERROR, error_context
        )
        
        self.assertEqual(severity, ErrorSeverity.LOW)
    
    async def test_analyze_error_success(self):
        """Test successful error analysis"""
        await self.error_detector.initialize()
        
        error_context = Mock()
        error_context.error_id = 'test_error'
        error_context.error_type = 'ValueError'
        error_context.error_message = 'CSS selector not found'
        error_context.agent_name = 'test_agent'
        error_context.timestamp = datetime.now()
        error_context.html_snapshot = '<html><body>Test</body></html>'
        
        analysis = await self.error_detector.analyze_error(error_context)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('category', analysis)
        self.assertIn('severity', analysis)
        self.assertIn('repairable', analysis)
        self.assertIn('confidence', analysis)


class TestCodePatcher(unittest.TestCase):
    """Test cases for CodePatcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'backup_enabled': True,
            'validation_level': 'syntax_only',
            'auto_rollback': True,
            'max_patch_size': 100000
        }
        self.code_patcher = CodePatcher(self.config)
        
        # Create temporary test file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_file.py')
        with open(self.test_file, 'w') as f:
            f.write('def broken_function():\n    raise ValueError("Broken")\n')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_code_patcher_initialization(self):
        """Test code patcher initialization"""
        self.assertTrue(self.code_patcher.backup_enabled)
        self.assertEqual(self.code_patcher.validation_level.value, 'syntax_only')
        self.assertTrue(self.code_patcher.auto_rollback)
        self.assertEqual(self.code_patcher.max_patch_size, 100000)
    
    def test_validate_patch_syntax_valid(self):
        """Test validation of syntactically correct patch"""
        valid_code = 'def fixed_function():\n    return "fixed"\n'
        
        result = asyncio.run(self.code_patcher._validate_patch(valid_code, self.test_file))
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertTrue(result.syntax_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_patch_syntax_invalid(self):
        """Test validation of syntactically incorrect patch"""
        invalid_code = 'def broken_function(\n    return "broken"\n'
        
        result = asyncio.run(self.code_patcher._validate_patch(invalid_code, self.test_file))
        
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertFalse(result.syntax_valid)
        self.assertGreater(len(result.errors), 0)
    
    async def test_apply_fix_success(self):
        """Test successful fix application"""
        await self.code_patcher.initialize()
        
        fix_code = 'def fixed_function():\n    return "fixed"\n'
        
        result = await self.code_patcher.apply_fix(self.test_file, fix_code)
        
        self.assertTrue(result)
        
        # Verify file was patched
        with open(self.test_file, 'r') as f:
            content = f.read()
        self.assertIn('def fixed_function():', content)
        self.assertIn('return "fixed"', content)
    
    def test_get_module_name_from_path(self):
        """Test module name extraction from file path"""
        # Test relative path
        module_name = self.code_patcher._get_module_name_from_path('agents/test_agent.py')
        self.assertEqual(module_name, 'agents.test_agent')
        
        # Test absolute path
        abs_path = os.path.abspath('agents/test_agent.py')
        module_name = self.code_patcher._get_module_name_from_path(abs_path)
        self.assertEqual(module_name, 'agents.test_agent')


class TestHealingMetrics(unittest.TestCase):
    """Test cases for HealingMetrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'metrics_dir': 'test_metrics',
            'export_dir': 'test_exports',
            'mttr_target_seconds': 60,
            'success_rate_target': 0.8
        }
        self.healing_metrics = HealingMetrics(self.config)
    
    def test_healing_metrics_initialization(self):
        """Test healing metrics initialization"""
        self.assertEqual(self.healing_metrics.mttr_target, 60)
        self.assertEqual(self.healing_metrics.success_rate_target, 0.8)
        self.assertIsInstance(self.healing_metrics.healing_events, list)
        self.assertIsInstance(self.healing_metrics.mttr_history, list)
        self.assertIsInstance(self.healing_metrics.success_rate_history, list)
    
    async def test_record_healing_event(self):
        """Test recording of healing events"""
        await self.healing_metrics.initialize()
        
        await self.healing_metrics.record_healing_event(
            error_id='test_error',
            success=True,
            time_to_repair=45.5,
            error_type='ValueError',
            agent_name='test_agent',
            category='test_category',
            severity='medium'
        )
        
        self.assertEqual(len(self.healing_metrics.healing_events), 1)
        
        event = self.healing_metrics.healing_events[0]
        self.assertEqual(event.error_id, 'test_error')
        self.assertTrue(event.success)
        self.assertEqual(event.time_to_repair, 45.5)
        self.assertEqual(event.error_type, 'ValueError')
        self.assertEqual(event.agent_name, 'test_agent')
    
    async def test_calculate_mttr(self):
        """Test MTTR calculation"""
        await self.healing_metrics.initialize()
        
        # Record some test events
        await self.healing_metrics.record_healing_event('error1', True, 30.0, 'ValueError', 'agent1')
        await self.healing_metrics.record_healing_event('error2', True, 60.0, 'ValueError', 'agent2')
        await self.healing_metrics.record_healing_event('error3', True, 90.0, 'ValueError', 'agent3')
        
        mttr_calc = await self.healing_metrics.calculate_mttr(24)  # 24 hours
        
        self.assertIsInstance(mttr_calc, MTTRCalculation)
        self.assertEqual(mttr_calc.successful_events, 3)
        self.assertEqual(mttr_calc.mttr_seconds, 60.0)  # (30+60+90)/3
        self.assertEqual(mttr_calc.median_ttr, 60.0)
    
    async def test_calculate_success_rate(self):
        """Test success rate calculation"""
        await self.healing_metrics.initialize()
        
        # Record some test events
        await self.healing_metrics.record_healing_event('error1', True, 30.0, 'ValueError', 'agent1')
        await self.healing_metrics.record_healing_event('error2', False, 0.0, 'ValueError', 'agent2')
        await self.healing_metrics.record_healing_event('error3', True, 60.0, 'ValueError', 'agent3')
        await self.healing_metrics.record_healing_event('error4', False, 0.0, 'ValueError', 'agent4')
        
        success_rate_calc = await self.healing_metrics.calculate_success_rate(24)
        
        self.assertIsInstance(success_rate_calc, type(await self.healing_metrics.calculate_success_rate(24)))
        self.assertEqual(success_rate_calc.total_attempts, 4)
        self.assertEqual(success_rate_calc.successful_attempts, 2)
        self.assertEqual(success_rate_calc.failed_attempts, 2)
        self.assertEqual(success_rate_calc.success_rate, 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete healing system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        self.data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(self.config_dir)
        os.makedirs(self.data_dir)
        
        # Create test configuration
        self.healing_config = {
            'auto_heal_enabled': True,
            'max_healing_attempts': 3,
            'mttr_target_seconds': 60,
            'backup_enabled': True,
            'backup_dir': os.path.join(self.data_dir, 'backups'),
            'temp_dir': os.path.join(self.data_dir, 'temp')
        }
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_end_to_end_healing_workflow(self):
        """Test complete end-to-end healing workflow"""
        # Create test file with error
        test_file = os.path.join(self.temp_dir, 'test_agent.py')
        with open(test_file, 'w') as f:
            f.write('''
def scrape_data():
    # This will fail - selector doesn't exist
    from bs4 import BeautifulSoup
    html = "<html><body><div class='content'>Test</div></body></html>"
    soup = BeautifulSoup(html, 'html.parser')
    return soup.select_one('div.nonexistent').text
''')
        
        # Initialize healing system
        healing_agent = HealingAgent("integration_test", self.healing_config)
        
        # Mock LLM client to return a fix
        with patch('agents.healing_agent.LLMClient') as mock_llm:
            mock_llm.return_value.generate_fix = AsyncMock(return_value='''
def scrape_data():
    from bs4 import BeautifulSoup
    html = "<html><body><div class='content'>Test</div></body></html>"
    soup = BeautifulSoup(html, 'html.parser')
    return soup.select_one('div.content').text
''')
            
            await healing_agent.initialize()
            
            # Simulate error
            error_context = {
                'error_type': 'AttributeError',
                'error_message': "'NoneType' object has no attribute 'text'",
                'agent_name': 'test_agent',
                'function_name': 'scrape_data',
                'file_path': test_file,
                'line_number': 6,
                'timestamp': datetime.now().isoformat(),
                'traceback': 'Traceback...',
                'additional_context': {'url': 'http://test.com'}
            }
            
            # Handle error
            result = await healing_agent.handle_error(error_context)
            
            # Verify healing was successful
            self.assertTrue(result.success)
            self.assertEqual(result.status, HealingStatus.COMPLETED)
            self.assertTrue(result.fix_applied)
            self.assertTrue(result.validation_passed)
            
            # Verify file was actually fixed
            with open(test_file, 'r') as f:
                fixed_content = f.read()
            self.assertIn('div.content', fixed_content)
            self.assertNotIn('div.nonexistent', fixed_content)
    
    async def test_healing_with_rollback(self):
        """Test healing with rollback on validation failure"""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test_agent.py')
        original_content = 'def original_function():\n    return "original"\n'
        with open(test_file, 'w') as f:
            f.write(original_content)
        
        # Initialize healing system with validation that will fail
        healing_config = self.healing_config.copy()
        healing_config['auto_rollback'] = True
        
        healing_agent = HealingAgent("rollback_test", healing_config)
        
        # Mock components to simulate validation failure
        with patch('agents.healing_agent.LLMClient') as mock_llm, \
             patch('agents.healing_agent.CodePatcher') as mock_patcher:
            
            mock_llm.return_value.generate_fix = AsyncMock(return_value='def broken():\n    syntax error')
            mock_patcher.return_value.apply_fix = AsyncMock(return_value=True)
            mock_patcher.return_value.validate_fix = AsyncMock(return_value=False)
            
            await healing_agent.initialize()
            
            error_context = {
                'error_type': 'ValueError',
                'error_message': 'Test error',
                'agent_name': 'test_agent',
                'function_name': 'test_function',
                'file_path': test_file,
                'line_number': 1,
                'timestamp': datetime.now().isoformat()
            }
            
            result = await healing_agent.handle_error(error_context)
            
            # Verify rollback occurred
            self.assertFalse(result.success)
            self.assertTrue(result.rollback_performed)
            
            # Verify original content was restored
            with open(test_file, 'r') as f:
                current_content = f.read()
            self.assertEqual(current_content, original_content)


class TestChaosEngineering(unittest.TestCase):
    """Test cases for chaos engineering scenarios"""
    
    def setUp(self):
        """Set up chaos test fixtures"""
        self.config = {
            'chaos_engineering': {
                'enabled': True,
                'max_concurrent_tests': 1,
                'test_timeout': 60
            }
        }
        self.healing_metrics = HealingMetrics(self.config)
    
    async def test_chaos_test_recording(self):
        """Test chaos test result recording"""
        await self.healing_metrics.initialize()
        
        await self.healing_metrics.record_chaos_test(
            test_type='error_injection',
            target_agent='test_agent',
            error_injected='ValueError',
            healing_triggered=True,
            healing_successful=True,
            time_to_heal=45.5,
            system_impact={'recovery_time': 45.5, 'service_disruption': False},
            test_passed=True,
            notes='Test completed successfully'
        )
        
        self.assertEqual(len(self.healing_metrics.chaos_test_results), 1)
        
        test_result = self.healing_metrics.chaos_test_results[0]
        self.assertEqual(test_result.test_type, 'error_injection')
        self.assertEqual(test_result.target_agent, 'test_agent')
        self.assertEqual(test_result.error_injected, 'ValueError')
        self.assertTrue(test_result.healing_triggered)
        self.assertTrue(test_result.healing_successful)
        self.assertEqual(test_result.time_to_heal, 45.5)
        self.assertTrue(test_result.test_passed)
    
    async def test_run_chaos_test(self):
        """Test chaos test execution"""
        await self.healing_metrics.initialize()
        
        test_config = {
            'type': 'error_injection',
            'target_agent': 'test_agent',
            'error_type': 'runtime_error',
            'timeout': 30
        }
        
        result = await self.healing_metrics.run_chaos_test(test_config)
        
        self.assertIsNotNone(result)
        self.assertIn(result.test_type, 'error_injection')
        self.assertIn(result.target_agent, 'test_agent')


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)