"""
Test Agent

Simple test agent for demonstrating the multi-agent workflow system.
This agent performs basic operations to validate the orchestration framework.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from agents.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """
    Simple test agent for workflow demonstration.
    
    Capabilities:
    - Basic task processing
    - Data transformation
    - Status reporting
    - Error simulation (for testing)
    """
    
    def _on_initialize(self):
        """Test agent specific initialization."""
        self.test_data = {
            'initialized_at': datetime.utcnow().isoformat(),
            'tasks_processed': 0,
            'test_mode': True
        }
        
        # Add test-specific event handlers
        self.add_event_handler('test_operation', self._handle_test_operation)
        
        self.logger.info(f"TestAgent {self.name} initialized with test data")
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a test task.
        
        Supported task types:
        - 'echo': Echo back the input data
        - 'transform': Transform input data
        - 'compute': Perform simple computation
        - 'status': Return agent status
        - 'error': Simulate an error (for testing)
        """
        task_type = task.get('type', 'unknown')
        task_data = task.get('data', {})
        
        self.logger.info(f"TestAgent {self.name} processing task: {task_type}")
        
        # Update test data
        self.test_data['tasks_processed'] += 1
        self.test_data['last_task_type'] = task_type
        self.test_data['last_task_at'] = datetime.utcnow().isoformat()
        
        try:
            if task_type == 'echo':
                return await self._handle_echo_task(task_data)
            elif task_type == 'transform':
                return await self._handle_transform_task(task_data)
            elif task_type == 'compute':
                return await self._handle_compute_task(task_data)
            elif task_type == 'status':
                return await self._handle_status_task(task_data)
            elif task_type == 'error':
                return await self._handle_error_task(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"TestAgent {self.name} task failed: {e}")
            raise
    
    async def _handle_echo_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle echo task - return input data unchanged."""
        return {
            'status': 'success',
            'task_type': 'echo',
            'echoed_data': data,
            'processed_by': self.name,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_transform_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transform task - transform input data."""
        transformed_data = {}
        
        # Transform various data types
        for key, value in data.items():
            if isinstance(value, str):
                transformed_data[f"{key}_upper"] = value.upper()
                transformed_data[f"{key}_length"] = len(value)
            elif isinstance(value, (int, float)):
                transformed_data[f"{key}_doubled"] = value * 2
                transformed_data[f"{key}_squared"] = value ** 2
            elif isinstance(value, list):
                transformed_data[f"{key}_count"] = len(value)
                transformed_data[f"{key}_reversed"] = list(reversed(value))
            else:
                transformed_data[f"{key}_type"] = type(value).__name__
        
        return {
            'status': 'success',
            'task_type': 'transform',
            'original_data': data,
            'transformed_data': transformed_data,
            'processed_by': self.name,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_compute_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compute task - perform simple computations."""
        numbers = data.get('numbers', [])
        operation = data.get('operation', 'sum')
        
        if not numbers:
            raise ValueError("No numbers provided for computation")
        
        result = None
        if operation == 'sum':
            result = sum(numbers)
        elif operation == 'average':
            result = sum(numbers) / len(numbers)
        elif operation == 'max':
            result = max(numbers)
        elif operation == 'min':
            result = min(numbers)
        elif operation == 'product':
            result = 1
            for num in numbers:
                result *= num
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            'status': 'success',
            'task_type': 'compute',
            'operation': operation,
            'numbers': numbers,
            'result': result,
            'processed_by': self.name,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_status_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status task - return agent status and metrics."""
        return {
            'status': 'success',
            'task_type': 'status',
            'agent_info': {
                'name': self.name,
                'status': self.get_status().value,
                'metrics': self.get_metrics(),
                'test_data': self.test_data,
                'config': self.config
            },
            'processed_by': self.name,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_error_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error task - simulate an error for testing."""
        error_type = data.get('error_type', 'general')
        error_message = data.get('error_message', 'Test error for demonstration')
        
        if error_type == 'value':
            raise ValueError(error_message)
        elif error_type == 'runtime':
            raise RuntimeError(error_message)
        elif error_type == 'timeout':
            # Simulate timeout by sleeping
            await asyncio.sleep(data.get('timeout', 5))
            raise TimeoutError(error_message)
        else:
            raise Exception(error_message)
    
    async def _handle_test_operation(self, event_data: Dict[str, Any]):
        """Handle custom test operation events."""
        operation = event_data.get('operation')
        self.logger.info(f"TestAgent {self.name} received test operation: {operation}")
    
    def get_test_capabilities(self) -> list:
        """Get list of test agent capabilities."""
        return [
            'echo',
            'data_transformation', 
            'computation',
            'status_reporting',
            'error_simulation'
        ]
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of test agent state."""
        return {
            'agent_name': self.name,
            'test_data': self.test_data,
            'capabilities': self.get_test_capabilities(),
            'current_status': self.get_status().value,
            'metrics': self.get_metrics(),
            'queue_length': len(self.task_queue),
            'current_task': self.current_task is not None
        }