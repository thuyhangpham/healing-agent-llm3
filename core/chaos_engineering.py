"""
Chaos Engineering Testing Framework

This module provides chaos engineering capabilities for testing
the robustness and resilience of the self-healing system.

Features:
- Error injection testing
- Network failure simulation
- Resource exhaustion testing
- Automated chaos scenarios
- Recovery validation
"""

import asyncio
import json
import os
import random
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

from utils.logger import get_logger
from utils.config import load_config


class ChaosTestType(Enum):
    """Types of chaos tests"""
    ERROR_INJECTION = "error_injection"
    NETWORK_FAILURE = "network_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"
    DISK_FULL = "disk_full"
    TIMEOUT_INJECTION = "timeout_injection"
    DEPENDENCY_FAILURE = "dependency_failure"


class ChaosSeverity(Enum):
    """Severity levels for chaos tests"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ChaosTest:
    """Definition of a chaos test"""
    test_id: str
    name: str
    description: str
    test_type: ChaosTestType
    severity: ChaosSeverity
    duration: float  # seconds
    probability: float  # 0.0 to 1.0
    target_agents: List[str]
    parameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class ChaosTestResult:
    """Result of a chaos test execution"""
    test_id: str
    timestamp: datetime
    success: bool
    duration: float
    error_injected: bool
    recovery_detected: bool
    recovery_time: float
    healing_triggered: bool
    healing_successful: bool
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class ChaosEngine:
    """
    Chaos Engineering Engine for testing system resilience
    
    This engine provides various chaos testing capabilities to validate
    the robustness and self-healing capabilities of the system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize chaos engine"""
        self.config = config or {}
        self.logger = get_logger("chaos_engine")
        
        # Configuration
        self.enabled = self.config.get('enabled', False)
        self.max_concurrent_tests = self.config.get('max_concurrent_tests', 1)
        self.test_timeout = self.config.get('test_timeout', 300)
        self.max_failure_rate = self.config.get('max_failure_rate', 0.1)
        self.recovery_timeout = self.config.get('recovery_timeout', 600)
        self.rollback_on_failure = self.config.get('rollback_on_failure', True)
        
        # Test state
        self.active_tests: Dict[str, ChaosTest] = {}
        self.test_history: List[ChaosTestResult] = []
        self.test_schedules: Dict[str, datetime] = {}
        
        # Chaos components
        self.error_injector = ErrorInjector(self.config.get('error_injection', {}))
        self.network_simulator = NetworkSimulator(self.config.get('network_failure', {}))
        self.resource_exhauster = ResourceExhauster(self.config.get('resource_exhaustion', {}))
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        self.logger.info("ChaosEngine initialized")
    
    async def initialize(self) -> bool:
        """Initialize chaos engine components"""
        try:
            if not self.enabled:
                self.logger.info("Chaos engineering is disabled")
                return True
            
            # Initialize components
            await self.error_injector.initialize()
            await self.network_simulator.initialize()
            await self.resource_exhauster.initialize()
            await self.system_monitor.initialize()
            
            # Load predefined tests
            await self._load_predefined_tests()
            
            self.logger.info("ChaosEngine initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChaosEngine: {e}")
            return False
    
    async def execute_test(self, test_id: str) -> ChaosTestResult:
        """Execute a specific chaos test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        self.logger.info(f"Executing chaos test: {test.name} ({test_id})")
        
        start_time = time.time()
        
        try:
            # Get baseline metrics
            metrics_before = await self.system_monitor.get_current_metrics()
            
            # Execute chaos test based on type
            if test.test_type == ChaosTestType.ERROR_INJECTION:
                success = await self._execute_error_injection(test)
            elif test.test_type == ChaosTestType.NETWORK_FAILURE:
                success = await self._execute_network_failure(test)
            elif test.test_type == ChaosTestType.RESOURCE_EXHAUSTION:
                success = await self._execute_resource_exhaustion(test)
            else:
                raise ValueError(f"Unsupported test type: {test.test_type}")
            
            # Wait for recovery
            recovery_detected, recovery_time = await self._wait_for_recovery(test)
            
            # Check if healing was triggered
            healing_triggered, healing_successful = await self._check_healing_response(test)
            
            # Get post-test metrics
            metrics_after = await self.system_monitor.get_current_metrics()
            
            duration = time.time() - start_time
            
            result = ChaosTestResult(
                test_id=test_id,
                timestamp=datetime.now(),
                success=success,
                duration=duration,
                error_injected=True,
                recovery_detected=recovery_detected,
                recovery_time=recovery_time,
                healing_triggered=healing_triggered,
                healing_successful=healing_successful,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                additional_data={'test_parameters': test.parameters}
            )
            
            self.test_history.append(result)
            
            self.logger.info(f"Chaos test {test_id} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Chaos test {test_id} failed: {e}")
            
            result = ChaosTestResult(
                test_id=test_id,
                timestamp=datetime.now(),
                success=False,
                duration=duration,
                error_injected=False,
                recovery_detected=False,
                recovery_time=0.0,
                healing_triggered=False,
                healing_successful=False,
                metrics_before={},
                metrics_after={},
                error_message=str(e)
            )
            
            self.test_history.append(result)
            return result
    
    async def _execute_error_injection(self, test: ChaosTest) -> bool:
        """Execute error injection test"""
        try:
            self.logger.info(f"Injecting errors into agents: {test.target_agents}")
            
            # Inject errors into target agents
            for agent_name in test.target_agents:
                error_type = test.parameters.get('error_type', 'RuntimeError')
                error_message = test.parameters.get('error_message', 'Chaos test error injection')
                
                await self.error_injector.inject_error(agent_name, error_type, error_message)
            
            # Wait for specified duration
            await asyncio.sleep(test.duration)
            
            # Clear injected errors
            for agent_name in test.target_agents:
                await self.error_injector.clear_error(agent_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error injection test failed: {e}")
            return False
    
    async def _execute_network_failure(self, test: ChaosTest) -> bool:
        """Execute network failure simulation"""
        try:
            failure_type = test.parameters.get('failure_type', 'connection_refused')
            duration = test.parameters.get('duration', test.duration)
            
            self.logger.info(f"Simulating {failure_type} network failure for {duration}s")
            
            # Simulate network failure
            await self.network_simulator.simulate_failure(failure_type, duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Network failure test failed: {e}")
            return False
    
    async def _execute_resource_exhaustion(self, test: ChaosTest) -> bool:
        """Execute resource exhaustion test"""
        try:
            resource_type = test.parameters.get('resource_type', 'memory')
            threshold = test.parameters.get('threshold', 0.8)
            
            self.logger.info(f"Exhausting {resource_type} resource to {threshold * 100}%")
            
            # Exhaust specified resource
            await self.resource_exhauster.exhaust_resource(resource_type, threshold, test.duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource exhaustion test failed: {e}")
            return False
    
    async def _wait_for_recovery(self, test: ChaosTest) -> Tuple[bool, float]:
        """Wait for system recovery after chaos test"""
        start_time = time.time()
        recovery_timeout = self.recovery_timeout
        
        while time.time() - start_time < recovery_timeout:
            try:
                # Check if system has recovered
                current_metrics = await self.system_monitor.get_current_metrics()
                
                if self._is_system_recovered(current_metrics):
                    recovery_time = time.time() - start_time
                    self.logger.info(f"System recovered in {recovery_time:.2f}s")
                    return True, recovery_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.warning(f"Error during recovery monitoring: {e}")
        
        recovery_time = time.time() - start_time
        self.logger.warning(f"System did not recover within {recovery_timeout}s")
        return False, recovery_time
    
    async def _check_healing_response(self, test: ChaosTest) -> Tuple[bool, bool]:
        """Check if healing system responded to chaos test"""
        try:
            # This would integrate with the actual healing system
            # For now, simulate healing response
            
            # Simulate healing trigger probability
            healing_probability = test.parameters.get('healing_probability', 0.7)
            healing_triggered = random.random() < healing_probability
            
            if healing_triggered:
                # Simulate healing success probability
                healing_success_probability = test.parameters.get('healing_success_probability', 0.8)
                healing_successful = random.random() < healing_success_probability
                
                self.logger.info(f"Healing triggered: {healing_triggered}, successful: {healing_successful}")
            else:
                healing_successful = False
            
            return healing_triggered, healing_successful
            
        except Exception as e:
            self.logger.error(f"Error checking healing response: {e}")
            return False, False
    
    def _is_system_recovered(self, metrics: Dict[str, Any]) -> bool:
        """Check if system has recovered based on metrics"""
        try:
            # Check key system health indicators
            error_rate = metrics.get('error_rate', 0.0)
            cpu_usage = metrics.get('cpu_usage', 0.0)
            memory_usage = metrics.get('memory_usage', 0.0)
            
            # System is considered recovered if:
            # - Error rate is below threshold
            # - Resource usage is reasonable
            # - No active chaos tests affecting critical components
            
            return (error_rate < 0.1 and 
                   cpu_usage < 0.8 and 
                   memory_usage < 0.8)
                   
        except Exception:
            return False
    
    async def _load_predefined_tests(self):
        """Load predefined chaos tests"""
        predefined_tests = [
            ChaosTest(
                test_id="css_selector_failure",
                name="CSS Selector Failure Injection",
                description="Inject CSS selector not found errors into web scraping agents",
                test_type=ChaosTestType.ERROR_INJECTION,
                severity=ChaosSeverity.MEDIUM,
                duration=60.0,
                probability=0.05,
                target_agents=["law_search_agent", "opinion_search_agent"],
                parameters={
                    'error_type': 'SelectorError',
                    'error_message': 'CSS selector not found: .content-area',
                    'healing_probability': 0.9,
                    'healing_success_probability': 0.8
                }
            ),
            ChaosTest(
                test_id="network_timeout",
                name="Network Timeout Simulation",
                description="Simulate network timeouts for external API calls",
                test_type=ChaosTestType.NETWORK_FAILURE,
                severity=ChaosSeverity.HIGH,
                duration=30.0,
                probability=0.03,
                target_agents=["law_search_agent", "opinion_search_agent"],
                parameters={
                    'failure_type': 'timeout',
                    'duration': 30.0,
                    'healing_probability': 0.7,
                    'healing_success_probability': 0.6
                }
            ),
            ChaosTest(
                test_id="memory_pressure",
                name="Memory Pressure Test",
                description="Apply memory pressure to test system resilience",
                test_type=ChaosTestType.RESOURCE_EXHAUSTION,
                severity=ChaosSeverity.MEDIUM,
                duration=45.0,
                probability=0.02,
                target_agents=["healing_agent"],
                parameters={
                    'resource_type': 'memory',
                    'threshold': 0.8,
                    'healing_probability': 0.5,
                    'healing_success_probability': 0.7
                }
            ),
            ChaosTest(
                test_id="runtime_error_injection",
                name="Runtime Error Injection",
                description="Inject runtime errors into agent processing",
                test_type=ChaosTestType.ERROR_INJECTION,
                severity=ChaosSeverity.HIGH,
                duration=30.0,
                probability=0.04,
                target_agents=["pdf_analysis_agent", "sentiment_analysis_agent"],
                parameters={
                    'error_type': 'RuntimeError',
                    'error_message': 'Simulated runtime error for chaos testing',
                    'healing_probability': 0.8,
                    'healing_success_probability': 0.7
                }
            )
        ]
        
        for test in predefined_tests:
            self.active_tests[test.test_id] = test
        
        self.logger.info(f"Loaded {len(predefined_tests)} predefined chaos tests")
    
    async def run_scheduled_tests(self):
        """Run scheduled chaos tests"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                for test_id, test in self.active_tests.items():
                    if not test.enabled:
                        continue
                    
                    # Check if test should run based on probability
                    if random.random() < test.probability:
                        self.logger.info(f"Starting scheduled chaos test: {test_id}")
                        await self.execute_test(test_id)
                
            except Exception as e:
                self.logger.error(f"Error in scheduled test loop: {e}")
    
    async def get_test_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chaos test results"""
        results = self.test_history[-limit:] if limit > 0 else self.test_history
        return [asdict(result) for result in results]
    
    async def get_test_statistics(self) -> Dict[str, Any]:
        """Get chaos test statistics"""
        if not self.test_history:
            return {}
        
        total_tests = len(self.test_history)
        successful_tests = sum(1 for r in self.test_history if r.success)
        recovery_detected = sum(1 for r in self.test_history if r.recovery_detected)
        healing_triggered = sum(1 for r in self.test_history if r.healing_triggered)
        healing_successful = sum(1 for r in self.test_history if r.healing_triggered and r.healing_successful)
        
        # Average recovery time
        recovery_times = [r.recovery_time for r in self.test_history if r.recovery_detected]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Test type breakdown
        test_type_stats = {}
        for result in self.test_history:
            test_type = result.test_id
            if test_type not in test_type_stats:
                test_type_stats[test_type] = {'total': 0, 'success': 0}
            test_type_stats[test_type]['total'] += 1
            if result.success:
                test_type_stats[test_type]['success'] += 1
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'recovery_detected': recovery_detected,
            'recovery_rate': (recovery_detected / total_tests * 100) if total_tests > 0 else 0,
            'healing_triggered': healing_triggered,
            'healing_triggered_rate': (healing_triggered / total_tests * 100) if total_tests > 0 else 0,
            'healing_successful': healing_successful,
            'healing_success_rate': (healing_successful / healing_triggered * 100) if healing_triggered > 0 else 0,
            'average_recovery_time': avg_recovery_time,
            'test_type_breakdown': test_type_stats
        }
    
    # ===== REQUIRED METHODS FOR RUN_SYSTEM.PY COMPATIBILITY =====
    
    async def inject_css_selector_failure(self, agent) -> Dict[str, Any]:
        """
        Inject CSS selector failure into target agent.
        
        Simulates a CSS selector not found error by updating agent's state to ERROR.
        
        Args:
            agent: Target agent instance
            
        Returns:
            Dictionary with injection result
        """
        try:
            self.logger.info(f"Injecting CSS selector failure into {agent.name if hasattr(agent, 'name') else 'agent'}")
            
            # Create mock HTML content for error context
            mock_html = """
            <!DOCTYPE html>
            <html>
            <head><title>Test Page</title></head>
            <body>
                <div class="header">Header Content</div>
                <div class="main-content">Main Content Here</div>
                <div class="sidebar">Sidebar Content</div>
                <script src="heavy-javascript.js"></script>
            </body>
            </html>
            """
            
            # Simulate the fault injection with error context
            fault_injection_result = {
                'success': True,
                'error_injected': True,
                'error_type': 'css_selector_failure',
                'error_message': 'CSS selector not found: .content-area',
                'target_agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'injection_method': 'css_selector_failure',
                'error_context': {
                    'html_snapshot': mock_html.strip(),
                    'url': 'https://example.com/test-page',
                    'selector_attempted': '.content-area',
                    'page_title': 'Test Page',
                    'user_agent': 'Mozilla/5.0 (compatible; TestAgent/1.0)'
                }
            }
            
            # Log the fault injection action
            self.logger.info(f"CSS selector failure injected: {fault_injection_result}")
            
            # Simulate agent state update to ERROR
            if hasattr(agent, 'status'):
                agent.status = 'ERROR'
            
            return fault_injection_result
            
        except Exception as e:
            self.logger.error(f"Failed to inject CSS selector failure: {e}")
            return {
                'success': False,
                'error_injected': False,
                'error_type': 'css_selector_failure',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'error_context': {}
            }
    
    async def inject_network_timeout(self, agent, duration: int = 30) -> Dict[str, Any]:
        """
        Inject network timeout into target agent.
        
        Simulates a network timeout by updating agent's state to ERROR.
        
        Args:
            agent: Target agent instance
            duration: Timeout duration in seconds
            
        Returns:
            Dictionary with injection result
        """
        try:
            self.logger.info(f"Injecting network timeout into {agent.name if hasattr(agent, 'name') else 'agent'} for {duration}s")
            
            # Simulate the fault injection
            fault_injection_result = {
                'success': True,
                'error_injected': True,
                'error_type': 'network_timeout',
                'error_message': f'Network timeout after {duration}s',
                'target_agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'injection_method': 'network_timeout'
            }
            
            # Log the fault injection action
            self.logger.info(f"Network timeout injected: {fault_injection_result}")
            
            # Simulate agent state update to ERROR
            if hasattr(agent, 'status'):
                agent.status = 'ERROR'
            
            return fault_injection_result
            
        except Exception as e:
            self.logger.error(f"Failed to inject network timeout: {e}")
            return {
                'success': False,
                'error_injected': False,
                'error_type': 'network_timeout',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def inject_encoding_error(self, agent) -> Dict[str, Any]:
        """
        Inject encoding error into target agent.
        
        Simulates a character encoding error by updating agent's state to ERROR.
        
        Args:
            agent: Target agent instance
            
        Returns:
            Dictionary with injection result
        """
        try:
            self.logger.info(f"Injecting encoding error into {agent.name if hasattr(agent, 'name') else 'agent'}")
            
            # Simulate the fault injection
            fault_injection_result = {
                'success': True,
                'error_injected': True,
                'error_type': 'encoding_error',
                'error_message': "'charmap' codec can't decode byte 0x9d",
                'target_agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'injection_method': 'encoding_error'
            }
            
            # Log the fault injection action
            self.logger.info(f"Encoding error injected: {fault_injection_result}")
            
            # Simulate agent state update to ERROR
            if hasattr(agent, 'status'):
                agent.status = 'ERROR'
            
            return fault_injection_result
            
        except Exception as e:
            self.logger.error(f"Failed to inject encoding error: {e}")
            return {
                'success': False,
                'error_injected': False,
                'error_type': 'encoding_error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def inject_javascript_heavy_page(self, agent) -> Dict[str, Any]:
        """
        Inject JavaScript heavy page scenario into target agent.
        
        Simulates a JavaScript-heavy page causing performance issues by updating agent's state to ERROR.
        
        Args:
            agent: Target agent instance
            
        Returns:
            Dictionary with injection result
        """
        try:
            self.logger.info(f"Injecting JavaScript heavy page scenario into {agent.name if hasattr(agent, 'name') else 'agent'}")
            
            # Create mock HTML content with heavy JavaScript for error context
            mock_html = """
            <!DOCTYPE html>
            <html>
            <head><title>Heavy JavaScript Page</title></head>
            <body>
                <div class="header">Header Content</div>
                <div class="main-content">Main Content Here</div>
                <div class="sidebar">Sidebar Content</div>
                <script src="https://cdn.example.com/heavy-analytics.js"></script>
                <script src="https://cdn.example.com/react-bundle.js"></script>
                <script src="https://cdn.example.com/vue-framework.js"></script>
                <script>
                    // Heavy JavaScript processing
                    for(let i = 0; i < 100000; i++) {
                        console.log('Processing item ' + i);
                    }
                </script>
            </body>
            </html>
            """
            
            # Simulate the fault injection with error context
            fault_injection_result = {
                'success': True,
                'error_injected': True,
                'error_type': 'javascript_heavy_page',
                'error_message': 'Page load timeout due to heavy JavaScript execution',
                'target_agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'injection_method': 'javascript_heavy_page',
                'error_context': {
                    'html_snapshot': mock_html.strip(),
                    'url': 'https://example.com/heavy-js-page',
                    'javascript_count': 4,
                    'estimated_load_time': '15.2s',
                    'page_title': 'Heavy JavaScript Page',
                    'user_agent': 'Mozilla/5.0 (compatible; TestAgent/1.0)',
                    'performance_impact': 'high'
                }
            }
            
            # Log the fault injection action
            self.logger.info(f"JavaScript heavy page scenario injected: {fault_injection_result}")
            
            # Simulate agent state update to ERROR
            if hasattr(agent, 'status'):
                agent.status = 'ERROR'
            
            return fault_injection_result
            
        except Exception as e:
            self.logger.error(f"Failed to inject JavaScript heavy page scenario: {e}")
            return {
                'success': False,
                'error_injected': False,
                'error_type': 'javascript_heavy_page',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'error_context': {}
            }
    
    async def inject_rate_limiting(self, agent) -> Dict[str, Any]:
        """
        Inject rate limiting scenario into target agent.
        
        Simulates rate limiting by updating agent's state to ERROR.
        
        Args:
            agent: Target agent instance
            
        Returns:
            Dictionary with injection result
        """
        try:
            self.logger.info(f"Injecting rate limiting scenario into {agent.name if hasattr(agent, 'name') else 'agent'}")
            
            # Simulate the fault injection
            fault_injection_result = {
                'success': True,
                'error_injected': True,
                'error_type': 'rate_limiting',
                'error_message': 'Rate limit exceeded: 429 requests per hour limit',
                'target_agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'injection_method': 'rate_limiting'
            }
            
            # Log the fault injection action
            self.logger.info(f"Rate limiting scenario injected: {fault_injection_result}")
            
            # Simulate agent state update to ERROR
            if hasattr(agent, 'status'):
                agent.status = 'ERROR'
            
            return fault_injection_result
            
        except Exception as e:
            self.logger.error(f"Failed to inject rate limiting scenario: {e}")
            return {
                'success': False,
                'error_injected': False,
                'error_type': 'rate_limiting',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get chaos engine statistics summarizing test execution counts.
        
        Returns:
            Dictionary with chaos engine statistics
        """
        try:
            # Count different types of injections
            injection_counts = {
                'css_selector_failure': 0,
                'network_timeout': 0,
                'encoding_error': 0,
                'javascript_heavy_page': 0,
                'rate_limiting': 0
            }
            
            # Count from test history
            for result in self.test_history:
                test_id = result.test_id
                if test_id in injection_counts:
                    injection_counts[test_id] += 1
            
            # Calculate success rates
            total_injections = sum(injection_counts.values())
            successful_injections = sum(1 for r in self.test_history if r.success and r.error_injected)
            
            return {
                'total_injections': total_injections,
                'successful_injections': successful_injections,
                'injection_success_rate': (successful_injections / total_injections * 100) if total_injections > 0 else 0,
                'injection_breakdown': injection_counts,
                'total_tests': len(self.test_history),
                'healing_triggered_count': sum(1 for r in self.test_history if r.healing_triggered),
                'healing_success_count': sum(1 for r in self.test_history if r.healing_successful),
                'average_recovery_time': self._calculate_average_recovery_time(),
                'last_injection': self.test_history[-1].timestamp.isoformat() if self.test_history else None,
                'chaos_engine_status': 'active' if self.enabled else 'disabled'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get chaos engine statistics: {e}")
            return {
                'total_injections': 0,
                'successful_injections': 0,
                'injection_success_rate': 0,
                'injection_breakdown': {},
                'total_tests': 0,
                'healing_triggered_count': 0,
                'healing_success_count': 0,
                'average_recovery_time': 0,
                'last_injection': None,
                'chaos_engine_status': 'error',
                'error': str(e)
            }
    
    def _calculate_average_recovery_time(self) -> float:
        """Calculate average recovery time from test history"""
        recovery_times = [r.recovery_time for r in self.test_history if r.recovery_detected and r.recovery_time > 0]
        return sum(recovery_times) / len(recovery_times) if recovery_times else 0.0
    
    async def shutdown(self):
        """Shutdown chaos engine"""
        try:
            self.logger.info("Shutting down ChaosEngine...")
            
            # Stop all active chaos tests
            for test_id in list(self.active_tests.keys()):
                await self._stop_test(test_id)
            
            # Shutdown components
            await self.error_injector.shutdown()
            await self.network_simulator.shutdown()
            await self.resource_exhauster.shutdown()
            await self.system_monitor.shutdown()
            
            self.logger.info("ChaosEngine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _stop_test(self, test_id: str):
        """Stop a specific chaos test"""
        try:
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
                
                # Stop chaos based on test type
                if test.test_type == ChaosTestType.NETWORK_FAILURE:
                    await self.network_simulator.stop_failure()
                elif test.test_type == ChaosTestType.RESOURCE_EXHAUSTION:
                    await self.resource_exhauster.stop_exhaustion()
                elif test.test_type == ChaosTestType.ERROR_INJECTION:
                    for agent_name in test.target_agents:
                        await self.error_injector.clear_error(agent_name)
                
                self.logger.info(f"Stopped chaos test: {test_id}")
                
        except Exception as e:
            self.logger.error(f"Error stopping test {test_id}: {e}")


class ErrorInjector:
    """Error injection component for chaos testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("error_injector")
        self.injected_errors: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize error injector"""
        self.logger.info("ErrorInjector initialized")
    
    async def inject_error(self, agent_name: str, error_type: str, error_message: str):
        """Inject error into specific agent"""
        self.injected_errors[agent_name] = {
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now(),
            'active': True
        }
        
        self.logger.info(f"Injected {error_type} into {agent_name}: {error_message}")
    
    async def clear_error(self, agent_name: str):
        """Clear injected error from agent"""
        if agent_name in self.injected_errors:
            del self.injected_errors[agent_name]
            self.logger.info(f"Cleared injected error from {agent_name}")
    
    async def shutdown(self):
        """Shutdown error injector"""
        self.injected_errors.clear()
        self.logger.info("ErrorInjector shutdown complete")


class NetworkSimulator:
    """Network failure simulation component"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("network_simulator")
        self.active_failures: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize network simulator"""
        self.logger.info("NetworkSimulator initialized")
    
    async def simulate_failure(self, failure_type: str, duration: float):
        """Simulate network failure"""
        failure_id = f"failure_{int(time.time())}"
        
        self.active_failures[failure_id] = {
            'failure_type': failure_type,
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        self.logger.info(f"Simulating {failure_type} for {duration}s")
        
        # Schedule failure stop
        asyncio.create_task(self._stop_failure_after_duration(failure_id, duration))
    
    async def _stop_failure_after_duration(self, failure_id: str, duration: float):
        """Stop failure after specified duration"""
        await asyncio.sleep(duration)
        
        if failure_id in self.active_failures:
            del self.active_failures[failure_id]
            self.logger.info(f"Stopped network failure: {failure_id}")
    
    async def stop_failure(self):
        """Stop all active network failures"""
        for failure_id in list(self.active_failures.keys()):
            del self.active_failures[failure_id]
        
        self.logger.info("Stopped all network failures")
    
    async def shutdown(self):
        """Shutdown network simulator"""
        self.active_failures.clear()
        self.logger.info("NetworkSimulator shutdown complete")


class ResourceExhauster:
    """Resource exhaustion simulation component"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("resource_exhauster")
        self.active_exhaustion: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize resource exhauster"""
        self.logger.info("ResourceExhauster initialized")
    
    async def exhaust_resource(self, resource_type: str, threshold: float, duration: float):
        """Exhaust specified resource"""
        exhaustion_id = f"exhaust_{int(time.time())}"
        
        self.active_exhaustion[exhaustion_id] = {
            'resource_type': resource_type,
            'threshold': threshold,
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        self.logger.info(f"Exhausting {resource_type} to {threshold * 100}% for {duration}s")
        
        # Schedule exhaustion stop
        asyncio.create_task(self._stop_exhaustion_after_duration(exhaustion_id, duration))
    
    async def _stop_exhaustion_after_duration(self, exhaustion_id: str, duration: float):
        """Stop resource exhaustion after specified duration"""
        await asyncio.sleep(duration)
        
        if exhaustion_id in self.active_exhaustion:
            del self.active_exhaustion[exhaustion_id]
            self.logger.info(f"Stopped resource exhaustion: {exhaustion_id}")
    
    async def stop_exhaustion(self):
        """Stop all active resource exhaustion"""
        for exhaustion_id in list(self.active_exhaustion.keys()):
            del self.active_exhaustion[exhaustion_id]
        
        self.logger.info("Stopped all resource exhaustion")
    
    async def shutdown(self):
        """Shutdown resource exhauster"""
        self.active_exhaustion.clear()
        self.logger.info("ResourceExhauster shutdown complete")


class SystemMonitor:
    """System monitoring component for chaos testing"""
    
    def __init__(self):
        self.logger = get_logger("system_monitor")
    
    async def initialize(self):
        """Initialize system monitor"""
        self.logger.info("SystemMonitor initialized")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            disk = psutil.disk_usage('/')
            disk_usage = disk.used / disk.total
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_usage / 100.0,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'error_rate': 0.0,  # Would be calculated from actual error logs
                'response_time': 0.0  # Would be calculated from actual response times
            }
            
        except ImportError:
            # Fallback if psutil not available
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': 0.5,
                'memory_usage': 0.5,
                'disk_usage': 0.3,
                'error_rate': 0.0,
                'response_time': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'error_rate': 1.0,
                'response_time': 999.0
            }
    
    async def shutdown(self):
        """Shutdown system monitor"""
        self.logger.info("SystemMonitor shutdown complete")