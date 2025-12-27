"""
Tests for Base Agent Class

Test suite for the core agent functionality that all agents inherit from.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from agents.base_agent import BaseAgent, AgentStatus, TaskPriority


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    def _on_initialize(self):
        """Test initialization logic."""
        self.test_initialized = True
    
    async def _process_task(self, task: dict) -> dict:
        """Test task processing."""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'error_task':
            raise ValueError("Test error")
        
        return {
            'status': 'completed',
            'task_type': task_type,
            'processed_by': self.name,
            'timestamp': datetime.utcnow().isoformat()
        }


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    def agent_config(self):
        """Sample agent configuration."""
        return {
            'max_retries': 2,
            'timeout': 60,
            'retry_delay': 0.1,
            'enable_metrics': True
        }
    
    @pytest.fixture
    def test_agent(self, agent_config):
        """Create a test agent instance."""
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            return TestAgent("test_agent", agent_config)
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.name == "test_agent"
        assert test_agent.status == AgentStatus.IDLE
        assert test_agent.test_initialized is True
        assert test_agent.metrics['tasks_completed'] == 0
        assert test_agent.metrics['tasks_failed'] == 0
        assert len(test_agent.task_history) == 0
        assert len(test_agent.task_queue) == 0
    
    def test_agent_configuration_loading(self, agent_config):
        """Test configuration loading with defaults."""
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            agent = TestAgent("test_agent", agent_config)
            
            # Check that config was merged with defaults
            assert agent.config['max_retries'] == 2
            assert agent.config['timeout'] == 60
            assert agent.config['retry_delay'] == 0.1
            assert agent.config['enable_metrics'] is True
            assert 'log_level' in agent.config
    
    def test_agent_configuration_defaults(self):
        """Test configuration loading with defaults only."""
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            agent = TestAgent("test_agent")
            
            # Check default values
            assert agent.config['max_retries'] == 3
            assert agent.config['timeout'] == 300
            assert agent.config['retry_delay'] == 1.0
            assert agent.config['enable_metrics'] is True
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self, test_agent):
        """Test successful task execution."""
        task = {
            'id': 'test_task_1',
            'type': 'test_task',
            'data': {'message': 'hello'}
        }
        
        result = await test_agent.execute(task)
        
        assert result['status'] == 'completed'
        assert result['task_type'] == 'test_task'
        assert result['processed_by'] == 'test_agent'
        assert test_agent.metrics['tasks_completed'] == 1
        assert test_agent.metrics['tasks_failed'] == 0
        assert len(test_agent.task_history) == 1
        assert test_agent.task_history[0]['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self, test_agent):
        """Test task execution with failure."""
        task = {
            'id': 'error_task',
            'type': 'error_task'
        }
        
        with pytest.raises(ValueError, match="Test error"):
            await test_agent.execute(task)
        
        assert test_agent.metrics['tasks_completed'] == 0
        assert test_agent.metrics['tasks_failed'] == 1
        assert len(test_agent.task_history) == 1
        assert test_agent.task_history[0]['status'] == 'failed'
        assert test_agent.status == AgentStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_task_execution_retry(self, test_agent):
        """Test task execution with retry logic."""
        # Create an agent that fails twice then succeeds
        class RetryTestAgent(TestAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
            
            async def _process_task(self, task: dict) -> dict:
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise ValueError(f"Attempt {self.attempt_count} failed")
                return {'status': 'success', 'attempts': self.attempt_count}
        
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            retry_agent = RetryTestAgent("retry_agent", {'max_retries': 3, 'retry_delay': 0.01})
            
            task = {'id': 'retry_task', 'type': 'test'}
            result = await retry_agent.execute(task)
            
            assert result['status'] == 'success'
            assert result['attempts'] == 3
            assert retry_agent.metrics['tasks_completed'] == 1
    
    @pytest.mark.asyncio
    async def test_task_execution_max_retries_exceeded(self):
        """Test task execution when max retries are exceeded."""
        class FailingAgent(TestAgent):
            async def _process_task(self, task: dict) -> dict:
                raise ValueError("Always fails")
        
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            failing_agent = FailingAgent("failing_agent", {'max_retries': 2, 'retry_delay': 0.01})
            
            task = {'id': 'failing_task', 'type': 'test'}
            
            with pytest.raises(ValueError, match="Always fails"):
                await failing_agent.execute(task)
            
            assert failing_agent.metrics['tasks_completed'] == 0
            assert failing_agent.metrics['tasks_failed'] == 1
    
    def test_state_management(self, test_agent):
        """Test state management functionality."""
        # Test setting and getting state
        test_state = {'key1': 'value1', 'key2': 'value2'}
        test_agent.set_state(test_state)
        
        assert test_agent.get_state('key1') == 'value1'
        assert test_agent.get_state('key2') == 'value2'
        assert test_agent.get_state('nonexistent', 'default') == 'default'
        
        # Test updating state
        test_agent.update_state('key1', 'new_value')
        assert test_agent.get_state('key1') == 'new_value'
    
    def test_workflow_state_management(self, test_agent):
        """Test workflow state management."""
        mock_workflow_state = Mock()
        test_agent.set_workflow_state(mock_workflow_state)
        
        assert test_agent.get_workflow_state() == mock_workflow_state
    
    def test_status_management(self, test_agent):
        """Test agent status management."""
        # Test initial status
        assert test_agent.get_status() == AgentStatus.IDLE
        
        # Test status change
        test_agent.set_status(AgentStatus.WORKING)
        assert test_agent.get_status() == AgentStatus.WORKING
        
        # Test status change event
        event_handler = Mock()
        test_agent.add_event_handler('status_changed', event_handler)
        
        test_agent.set_status(AgentStatus.COMPLETED)
        
        # Check event was called
        event_handler.assert_called_once()
        call_args = event_handler.call_args[0][0]
        assert call_args['old_status'] == AgentStatus.WORKING.value
        assert call_args['new_status'] == AgentStatus.COMPLETED.value
    
    def test_task_queue_management(self, test_agent):
        """Test task queue functionality."""
        # Test adding tasks to queue
        task1 = {'id': 'task1', 'type': 'test'}
        task2 = {'id': 'task2', 'type': 'test'}
        task3 = {'id': 'task3', 'type': 'test'}
        
        test_agent.add_task_to_queue(task1, TaskPriority.LOW)
        test_agent.add_task_to_queue(task2, TaskPriority.HIGH)
        test_agent.add_task_to_queue(task3, TaskPriority.MEDIUM)
        
        # Check queue length and priority ordering
        assert len(test_agent.task_queue) == 3
        assert test_agent.task_queue[0]['priority'] == TaskPriority.HIGH.value
        assert test_agent.task_queue[1]['priority'] == TaskPriority.MEDIUM.value
        assert test_agent.task_queue[2]['priority'] == TaskPriority.LOW.value
        
        # Test getting next task
        next_task = test_agent.get_next_task()
        assert next_task['id'] == 'task2'
        assert len(test_agent.task_queue) == 2
        
        # Test clearing queue
        test_agent.clear_task_queue()
        assert len(test_agent.task_queue) == 0
    
    def test_metrics_tracking(self, test_agent):
        """Test performance metrics tracking."""
        # Test initial metrics
        metrics = test_agent.get_metrics()
        assert metrics['tasks_completed'] == 0
        assert metrics['tasks_failed'] == 0
        assert metrics['total_execution_time'] == 0.0
        assert metrics['average_execution_time'] == 0.0
        assert metrics['current_status'] == AgentStatus.IDLE.value
        assert metrics['task_queue_length'] == 0
        assert metrics['current_task'] is False
    
    def test_event_handlers(self, test_agent):
        """Test event handler functionality."""
        # Test adding event handlers
        handler1 = Mock()
        handler2 = Mock()
        
        test_agent.add_event_handler('test_event', handler1)
        test_agent.add_event_handler('test_event', handler2)
        
        # Test event emission
        test_data = {'test': 'data'}
        asyncio.run(test_agent._emit_event('test_event', test_data))
        
        # Check both handlers were called
        handler1.assert_called_once_with(test_data)
        handler2.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_disabled_agent_execution(self, test_agent):
        """Test that disabled agents cannot execute tasks."""
        test_agent.set_status(AgentStatus.DISABLED)
        
        task = {'id': 'test_task', 'type': 'test'}
        
        with pytest.raises(RuntimeError, match="Agent test_agent is disabled"):
            await test_agent.execute(task)
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, test_agent):
        """Test agent shutdown functionality."""
        # Add some tasks and set working status
        test_agent.add_task_to_queue({'id': 'task1', 'type': 'test'})
        test_agent.set_status(AgentStatus.WORKING)
        
        # Shutdown the agent
        await test_agent.shutdown()
        
        # Check shutdown effects
        assert test_agent.get_status() == AgentStatus.DISABLED
        assert len(test_agent.task_queue) == 0
        assert test_agent.current_task is None
    
    def test_string_representations(self, test_agent):
        """Test string representations of agent."""
        # Test __str__
        str_repr = str(test_agent)
        assert "test_agent" in str_repr
        assert "idle" in str_repr
        
        # Test __repr__
        repr_str = repr(test_agent)
        assert "test_agent" in repr_str
        assert "BaseAgent" in repr_str
        assert "tasks_completed=0" in repr_str
        assert "tasks_failed=0" in repr_str


class TestAgentStatusAndPriority:
    """Test cases for AgentStatus and TaskPriority enums."""
    
    def test_agent_status_enum(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.INITIALIZING.value == "initializing"
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.WORKING.value == "working"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.DISABLED.value == "disabled"
    
    def test_task_priority_enum(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.LOW.value == 1
        assert TaskPriority.MEDIUM.value == 2
        assert TaskPriority.HIGH.value == 3
        assert TaskPriority.CRITICAL.value == 4


class TestAgentIntegration:
    """Integration tests for agent functionality."""
    
    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self):
        """Test complete task lifecycle from queue to completion."""
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            agent = TestAgent("lifecycle_agent")
            
            # Add task to queue
            task = {'id': 'lifecycle_task', 'type': 'test', 'data': {'test': True}}
            agent.add_task_to_queue(task, TaskPriority.HIGH)
            
            # Get task from queue
            queued_task = agent.get_next_task()
            assert queued_task is not None
            assert queued_task['id'] == 'lifecycle_task'
            
            # Execute task
            result = await agent.execute(queued_task)
            
            # Verify completion
            assert result['status'] == 'completed'
            assert agent.metrics['tasks_completed'] == 1
            assert len(agent.task_history) == 1
            assert agent.get_status() == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration with workflow state."""
        mock_workflow_state = Mock()
        
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler') as mock_error_handler:
            
            # Setup mock error handler
            mock_error_handler.return_value.capture_error.return_value = {
                'type': 'ValueError',
                'message': 'Test error',
                'agent': 'error_agent'
            }
            
            agent = TestAgent("error_agent")
            agent.set_workflow_state(mock_workflow_state)
            
            # Execute failing task
            task = {'id': 'error_task', 'type': 'error_task'}
            
            try:
                await agent.execute(task)
            except ValueError:
                pass  # Expected
            
            # Verify error was reported to workflow state
            mock_workflow_state.add_error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])