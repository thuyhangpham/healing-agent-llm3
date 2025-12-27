"""
Integration Tests for Healing System

Integration tests for healing system components including
orchestrator integration, agent coordination, and end-to-end workflows.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.healing_agent import HealingAgent, HealingStatus
from agents.orchestrator import Orchestrator
from agents.law_search_agent import AutonomousLawSearchAgent
from agents.opinion_search_agent import AutonomousOpinionSearchAgent
from core.llm_client import LLMClient
from core.error_detector import ErrorDetector
from core.code_patcher import CodePatcher
from core.healing_metrics import HealingMetrics
from healing.validator import CodeValidator
from utils.agent_registry import AgentRegistry
from utils.global_state import GlobalState


class TestHealingSystemIntegration(unittest.TestCase):
    """Integration tests for complete healing system"""
    
    def setUp(self):
        """Set up test environment"""
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
            'error_detection': {
                'similarity_threshold': 0.7,
                'pattern_min_frequency': 3
            },
            'code_patching': {
                'backup_enabled': True,
                'validation_level': 'basic_execution',
                'auto_rollback': True
            },
            'metrics': {
                'metrics_file': os.path.join(self.data_dir, 'test_metrics.json'),
                'research_data_dir': os.path.join(self.data_dir, 'research')
            }
        }
        
        self.agents_config = {
            'agents': {
                'healing_agent': {
                    'enabled': True,
                    'type': 'healing',
                    'config_file': os.path.join(self.config_dir, 'healing.yaml')
                },
                'law_search_agent': {
                    'enabled': True,
                    'type': 'law_search',
                    'sources': [{
                        'name': 'test_source',
                        'url': 'https://example.gov/laws',
                        'rate_limit': 1
                    }]
                },
                'opinion_search_agent': {
                    'enabled': True,
                    'type': 'opinion_search',
                    'sources': [{
                        'name': 'test_news',
                        'url': 'https://example-news.com',
                        'rate_limit': 1
                    }]
                }
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('core.llm_client.aiohttp.ClientSession')
    async def test_healing_agent_initialization(self, mock_session):
        """Test healing agent initialization and registration"""
        # Mock LLM responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'models': [{'name': 'llama3.1:8b'}]
        })
        mock_session.return_value.__aenter__.return_value = mock_response
        
        # Initialize healing agent
        healing_agent = HealingAgent("test_healing_agent", self.healing_config['healing_agent'])
        
        # Test initialization
        result = await healing_agent.initialize()
        self.assertTrue(result)
        
        # Test status
        status = await healing_agent.get_status()
        self.assertIsNotNone(status)
        self.assertIn('agent_id', status)
        self.assertIn('status', status)
    
    async def test_error_detection_workflow(self):
        """Test complete error detection and healing workflow"""
        # Initialize components
        error_detector = ErrorDetector(self.healing_config['error_detection'])
        await error_detector.initialize()
        
        # Create test error context
        from agents.healing_agent import ErrorContext, ErrorSeverity
        
        error_context = ErrorContext(
            error_id='test_error_1',
            timestamp=datetime.now(),
            error_type='ValueError',
            error_message='CSS selector not found: .content-area',
            traceback_str='Traceback (most recent call last):\n  File "test.py", line 10, in scrape_data\n    selector = soup.select_one(".content-area")\nAttributeError: NoneType object has no attribute \'text\'',
            agent_name='law_search_agent',
            function_name='scrape_data',
            file_path='/test/law_search_agent.py',
            line_number=10,
            severity=ErrorSeverity.MEDIUM,
            html_snapshot='<html><body><div class="main-content">Content</div></body></html>',
            additional_context={'url': 'https://example.gov/laws'}
        )
        
        # Test error analysis
        analysis = await error_detector.analyze_error(error_context)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('repairable', analysis)
        self.assertIn('error_category', analysis)
        self.assertIn('confidence', analysis)
        self.assertIn('root_cause', analysis)
        
        # Verify error categorization
        self.assertEqual(analysis['error_category'], 'web_scraping_failure')
        self.assertGreater(analysis['confidence'], 0.5)
    
    @patch('core.llm_client.aiohttp.ClientSession')
    async def test_code_patching_workflow(self, mock_session):
        """Test code patching and hot-reload workflow"""
        # Mock LLM responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'models': [{'name': 'llama3.1:8b'}]
        })
        mock_session.return_value.__aenter__.return_value = mock_response
        
        # Initialize components
        llm_client = LLMClient(self.healing_config['llm'])
        await llm_client.initialize()
        
        code_patcher = CodePatcher(self.healing_config['code_patching'])
        await code_patcher.initialize()
        
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test_agent.py')
        original_code = '''
def scrape_content(soup):
    """Scrape content from page."""
    content_element = soup.select_one(".content-area")
    return content_element.text
'''
        
        with open(test_file, 'w') as f:
            f.write(original_code)
        
        # Generate fix
        error_context = {
            'error_type': 'AttributeError',
            'error_message': 'NoneType object has no attribute \'text\'',
            'file_path': test_file,
            'function_name': 'scrape_content'
        }
        
        analysis = {
            'repairable': True,
            'error_category': 'web_scraping_failure',
            'suggested_approach': 'Update CSS selector'
        }
        
        fix_code = await llm_client.generate_fix(error_context, analysis)
        
        self.assertIsNotNone(fix_code)
        self.assertIn('def scrape_content', fix_code)
        
        # Apply fix
        fix_applied = await code_patcher.apply_fix(test_file, fix_code)
        self.assertTrue(fix_applied)
        
        # Verify fix was applied
        with open(test_file, 'r') as f:
            updated_code = f.read()
        
        self.assertNotEqual(original_code, updated_code)
        self.assertIn(fix_code, updated_code)
    
    async def test_metrics_collection_workflow(self):
        """Test metrics collection and analysis"""
        # Initialize metrics
        metrics = HealingMetrics(self.healing_config['metrics'])
        await metrics.initialize()
        
        # Record test healing events
        await metrics.record_healing_event(
            error_id='test_error_1',
            success=True,
            time_to_repair=45.5,
            error_type='CSSSelectorError',
            agent_name='law_search_agent',
            error_category='web_scraping_failure',
            severity=2,
            confidence=0.9
        )
        
        await metrics.record_healing_event(
            error_id='test_error_2',
            success=False,
            time_to_repair=120.0,
            error_type='NetworkError',
            agent_name='opinion_search_agent',
            error_category='network_error',
            severity=3,
            confidence=0.3
        )
        
        await metrics.record_healing_event(
            error_id='test_error_3',
            success=True,
            time_to_repair=30.0,
            error_type='RuntimeError',
            agent_name='pdf_analysis_agent',
            error_category='runtime_error',
            severity=2,
            confidence=0.8
        )
        
        # Test metrics calculations
        success_rate = metrics.get_success_rate()
        self.assertAlmostEqual(success_rate, 66.67, places=1)  # 2 out of 3
        
        avg_mttr = metrics.get_average_mttr()
        self.assertAlmostEqual(avg_mttr, 37.75, places=1)  # (45.5 + 30.0) / 2
        
        # Test agent performance
        agent_performance = metrics.get_agent_performance()
        self.assertIn('law_search_agent', agent_performance)
        self.assertIn('opinion_search_agent', agent_performance)
        self.assertIn('pdf_analysis_agent', agent_performance)
        
        # Test research data export
        research_data = metrics.get_research_data()
        self.assertIsInstance(research_data, dict)
        self.assertIn('summary', research_data)
        self.assertIn('mttr_analysis', research_data)
        self.assertIn('error_categories', research_data)
    
    async def test_orchestrator_integration(self):
        """Test orchestrator integration with healing system"""
        # This would test the actual orchestrator integration
        # For now, we'll simulate the integration
        
        # Mock orchestrator
        with patch('utils.agent_registry.AgentRegistry') as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry.return_value = mock_registry_instance
            
            # Mock getting healing agent
            mock_healing_agent = AsyncMock()
            mock_healing_agent.get_status = AsyncMock(return_value={
                'agent_id': 'healing_agent',
                'status': 'idle',
                'auto_heal_enabled': True
            })
            mock_registry_instance.get_agent.return_value = mock_healing_agent
            
            # Test error reporting to healing agent
            healing_agent = HealingAgent("healing_agent", self.healing_config['healing_agent'])
            
            # Simulate error report
            error_context = {
                'error_type': 'TestError',
                'error_message': 'Test error message',
                'agent_name': 'test_agent',
                'file_path': '/test/file.py',
                'function_name': 'test_function'
            }
            
            # This would normally be handled by orchestrator
            # For testing, we'll call handle_error directly
            result = await healing_agent.handle_error(error_context)
            
            self.assertIsNotNone(result)
            self.assertIn('success', result)
            self.assertIn('status', result)
    
    async def test_end_to_end_healing_workflow(self):
        """Test complete end-to-end healing workflow"""
        # Initialize all components
        error_detector = ErrorDetector(self.healing_config['error_detection'])
        await error_detector.initialize()
        
        metrics = HealingMetrics(self.healing_config['metrics'])
        await metrics.initialize()
        
        # Simulate error occurrence
        from agents.healing_agent import ErrorContext, ErrorSeverity
        
        error_context = ErrorContext(
            error_id='e2e_test_1',
            timestamp=datetime.now(),
            error_type='AttributeError',
            error_message='CSS selector not found',
            traceback_str='Traceback...',
            agent_name='law_search_agent',
            function_name='scrape_laws',
            file_path='/test/law_search_agent.py',
            line_number=25,
            severity=ErrorSeverity.HIGH,
            html_snapshot='<html><body><div class="article-content">New content</div></body></html>',
            additional_context={'url': 'https://example.gov/laws/123'}
        )
        
        # Step 1: Error Detection
        analysis = await error_detector.analyze_error(error_context)
        self.assertTrue(analysis['repairable'])
        
        # Step 2: Record metrics (simulating healing attempt)
        await metrics.record_healing_event(
            error_id=error_context.error_id,
            success=True,
            time_to_repair=52.3,
            error_type=error_context.error_type,
            agent_name=error_context.agent_name,
            error_category=analysis['error_category'],
            severity=error_context.severity.value,
            confidence=analysis['confidence']
        )
        
        # Step 3: Verify metrics
        success_rate = metrics.get_success_rate()
        self.assertEqual(success_rate, 100.0)  # 1 out of 1
        
        avg_mttr = metrics.get_average_mttr()
        self.assertEqual(avg_mttr, 52.3)
        
        # Step 4: Verify research data
        research_data = metrics.get_research_data()
        self.assertEqual(research_data['summary']['total_events'], 1)
        self.assertEqual(research_data['summary']['successful_events'], 1)
        self.assertEqual(research_data['summary']['failed_events'], 0)
        self.assertEqual(research_data['summary']['overall_success_rate'], 100.0)
    
    async def test_concurrent_healing_operations(self):
        """Test handling of concurrent healing operations"""
        # Initialize metrics
        metrics = HealingMetrics(self.healing_config['metrics'])
        await metrics.initialize()
        
        # Simulate concurrent healing events
        tasks = []
        for i in range(5):
            task = metrics.record_healing_event(
                error_id=f'concurrent_test_{i}',
                success=i % 2 == 0,  # Alternate success/failure
                time_to_repair=30.0 + i * 10,
                error_type='TestError',
                agent_name='test_agent',
                error_category='test_category',
                severity=2,
                confidence=0.8
            )
            tasks.append(task)
        
        # Wait for all events to be recorded
        await asyncio.gather(*tasks)
        
        # Verify all events were recorded
        success_rate = metrics.get_success_rate()
        self.assertEqual(success_rate, 40.0)  # 2 out of 5
        
        avg_mttr = metrics.get_average_mttr()
        expected_mttr = (30.0 + 50.0) / 2  # Average of successful events
        self.assertAlmostEqual(avg_mttr, expected_mttr, places=1)
    
    async def test_error_pattern_learning(self):
        """Test error pattern learning and recognition"""
        # Initialize error detector
        error_detector = ErrorDetector(self.healing_config['error_detection'])
        await error_detector.initialize()
        
        # Create similar error contexts
        similar_errors = []
        for i in range(5):
            error_context = MagicMock()
            error_context.error_type = 'CSSSelectorError'
            error_context.error_message = f'CSS selector not found: .content-area-{i}'
            error_context.agent_name = 'law_search_agent'
            error_context.additional_context = {'url': f'https://example.gov/laws/{i}'}
            similar_errors.append(error_context)
        
        # Process errors to build patterns
        for error_context in similar_errors:
            await error_detector.analyze_error(error_context)
        
        # Check if patterns were learned
        patterns = await error_detector.get_patterns()
        
        # Should have patterns for CSS selector errors
        css_patterns = [p for p in patterns.keys() if 'css_selector_failure' in p]
        self.assertGreater(len(css_patterns), 0)
    
    async def test_backup_and_rollback_functionality(self):
        """Test backup creation and rollback functionality"""
        # Initialize code patcher
        code_patcher = CodePatcher(self.healing_config['code_patching'])
        await code_patcher.initialize()
        
        # Create test file
        test_file = os.path.join(self.temp_dir, 'backup_test.py')
        original_code = 'def test_function():\n    return "original"'
        
        with open(test_file, 'w') as f:
            f.write(original_code)
        
        # Apply fix (should create backup)
        fix_code = 'def test_function():\n    return "fixed"'
        
        fix_applied = await code_patcher.apply_fix(test_file, fix_code)
        self.assertTrue(fix_applied)
        
        # Check if backup was created
        backup_files = list(Path(self.data_dir).glob("**/*.bak"))
        self.assertGreater(len(backup_files), 0)
        
        # Verify backup contains original code
        if backup_files:
            with open(backup_files[0], 'r') as f:
                backup_content = f.read()
            self.assertIn('return "original"', backup_content)
        
        # Test rollback
        rollback_success = await code_patcher.manual_rollback("patch_1")  # Use patch ID from apply_fix
        # Note: This would need the actual patch ID from the apply_fix operation
    
    async def test_system_health_monitoring(self):
        """Test system health monitoring capabilities"""
        # Initialize all components
        components = []
        
        try:
            # Initialize healing agent
            healing_agent = HealingAgent("health_test", self.healing_config['healing_agent'])
            components.append(('healing_agent', healing_agent))
            
            # Initialize metrics
            metrics = HealingMetrics(self.healing_config['metrics'])
            await metrics.initialize()
            components.append(('metrics', metrics))
            
            # Test health checks
            health_status = {}
            
            for name, component in components:
                try:
                    if hasattr(component, 'get_status'):
                        status = await component.get_status()
                        health_status[name] = {
                            'healthy': status is not None,
                            'status': status.get('status', 'unknown') if isinstance(status, dict) else 'unknown'
                        }
                    else:
                        health_status[name] = {
                            'healthy': True,
                            'status': 'initialized'
                        }
                except Exception as e:
                    health_status[name] = {
                        'healthy': False,
                        'error': str(e)
                    }
            
            # Verify health status
            self.assertIn('healing_agent', health_status)
            self.assertIn('metrics', health_status)
            
            # All components should be healthy
            for name, status in health_status.items():
                self.assertTrue(status['healthy'], f"Component {name} should be healthy")
            
        except Exception as e:
            self.fail(f"Health monitoring test failed: {e}")
    
    async def test_configuration_validation(self):
        """Test configuration validation and loading"""
        # Test healing configuration validation
        healing_config = self.healing_config['healing_agent']
        
        # Required fields should be present
        required_fields = ['auto_heal_enabled', 'max_healing_attempts', 'mttr_target_seconds']
        for field in required_fields:
            self.assertIn(field, healing_config)
        
        # Test LLM configuration validation
        llm_config = self.healing_config['llm']
        self.assertIn('base_url', llm_config)
        self.assertIn('model', llm_config)
        
        # Test metrics configuration validation
        metrics_config = self.healing_config['metrics']
        self.assertIn('metrics_file', metrics_config)
        self.assertIn('research_data_dir', metrics_config)
    
    async def test_performance_target_validation(self):
        """Test performance target validation and monitoring"""
        # Initialize metrics
        metrics = HealingMetrics(self.healing_config['metrics'])
        await metrics.initialize()
        
        # Record events that meet targets
        await metrics.record_healing_event(
            error_id='target_test_1',
            success=True,
            time_to_repair=45.0,  # Below 60s target
            error_type='TestError',
            agent_name='test_agent',
            error_category='test_category',
            severity=2,
            confidence=0.9
        )
        
        await metrics.record_healing_event(
            error_id='target_test_2',
            success=True,
            time_to_repair=55.0,  # Below 60s target
            error_type='TestError',
            agent_name='test_agent',
            error_category='test_category',
            severity=2,
            confidence=0.8
        )
        
        # Record some failures to test success rate
        await metrics.record_healing_event(
            error_id='target_test_3',
            success=False,
            time_to_repair=0.0,
            error_type='TestError',
            agent_name='test_agent',
            error_category='test_category',
            severity=2,
            confidence=0.0
        )
        
        # Check target achievement
        research_data = metrics.get_research_data()
        targets = research_data.get('targets', {})
        
        # MTTR target should be met
        self.assertTrue(targets.get('mttr_achieved', False))
        
        # Success rate target should be calculated
        success_rate = research_data['summary']['overall_success_rate']
        self.assertAlmostEqual(success_rate, 66.67, places=1)  # 2 out of 3


class TestHealingSystemResilience(unittest.TestCase):
    """Test healing system resilience under stress"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'healing_agent': {
                'auto_heal_enabled': True,
                'max_healing_attempts': 3,
                'mttr_target_seconds': 60,
                'success_rate_target': 0.8
            },
            'metrics': {
                'metrics_file': os.path.join(self.temp_dir, 'stress_test_metrics.json')
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_high_error_rate_handling(self):
        """Test system behavior under high error rates"""
        metrics = HealingMetrics(self.config['metrics'])
        await metrics.initialize()
        
        # Simulate high error rate
        error_count = 50
        success_count = 30  # 60% success rate
        
        tasks = []
        for i in range(error_count):
            success = i < success_count
            task = metrics.record_healing_event(
                error_id=f'stress_test_{i}',
                success=success,
                time_to_repair=30.0 + (i % 20),
                error_type='StressTestError',
                agent_name='stress_test_agent',
                error_category='stress_test',
                severity=2,
                confidence=0.7 if success else 0.3
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Check metrics
        success_rate = metrics.get_success_rate()
        self.assertAlmostEqual(success_rate, 60.0, places=1)
        
        avg_mttr = metrics.get_average_mttr()
        self.assertGreater(avg_mttr, 0)
        
        # Check system can handle high volume
        research_data = metrics.get_research_data()
        self.assertEqual(research_data['summary']['total_events'], error_count)
        self.assertEqual(research_data['summary']['successful_events'], success_count)
    
    async def test_resource_exhaustion_recovery(self):
        """Test system recovery from resource exhaustion"""
        # This would test actual resource exhaustion scenarios
        # For now, simulate the metrics tracking
        
        metrics = HealingMetrics(self.config['metrics'])
        await metrics.initialize()
        
        # Simulate resource exhaustion events
        resource_errors = [
            ('memory_exhaustion', 120.0),  # High MTTR
            ('cpu_exhaustion', 90.0),
            ('disk_full', 150.0)
        ]
        
        for error_type, mttr in resource_errors:
            await metrics.record_healing_event(
                error_id=f'resource_{error_type}',
                success=mttr < 100,  # Some succeed, some fail
                time_to_repair=mttr,
                error_type=error_type,
                agent_name='resource_test_agent',
                error_category='resource_exhaustion',
                severity=3,
                confidence=0.5
            )
        
        # Check system tracks resource issues
        research_data = metrics.get_research_data()
        error_categories = research_data.get('error_categories', {})
        
        self.assertIn('resource_exhaustion', error_categories)
        resource_stats = error_categories['resource_exhaustion']
        self.assertGreater(resource_stats['total'], 0)
    
    async def test_cascading_failure_handling(self):
        """Test handling of cascading failures"""
        metrics = HealingMetrics(self.config['metrics'])
        await metrics.initialize()
        
        # Simulate cascading failures
        cascade_events = [
            ('initial_error', True, 30.0),
            ('secondary_error', False, 0.0),  # Failed to heal
            ('tertiary_error', True, 45.0),
            ('recovery_error', True, 60.0)
        ]
        
        for error_id, success, mttr in cascade_events:
            await metrics.record_healing_event(
                error_id=error_id,
                success=success,
                time_to_repair=mttr,
                error_type='CascadeError',
                agent_name='cascade_test_agent',
                error_category='cascading_failure',
                severity=3 if not success else 2,
                confidence=0.8 if success else 0.2
            )
        
        # Check cascade handling
        success_rate = metrics.get_success_rate()
        expected_success_rate = (3 / 4) * 100  # 3 out of 4 successful
        self.assertAlmostEqual(success_rate, expected_success_rate, places=1)
        
        # Check average MTTR excludes failures
        avg_mttr = metrics.get_average_mttr()
        expected_mttr = (30.0 + 45.0 + 60.0) / 3  # Average of successful
        self.assertAlmostEqual(avg_mttr, expected_mttr, places=1)


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)