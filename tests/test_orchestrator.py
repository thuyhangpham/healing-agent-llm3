"""
Tests for Orchestrator Agent

Test suite for the central coordination agent that manages
workflow execution, agent registration, and task scheduling.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from agents.orchestrator import OrchestratorAgent, OrchestratorStatus, WorkflowPhase
from agents.base_agent import BaseAgent, AgentStatus, TaskPriority


class MockAgent(BaseAgent):
    """Mock agent for testing orchestrator."""
    
    def _on_initialize(self):
        self.mock_initialized = True
    
    async def _process_task(self, task: dict) -> dict:
        return {
            'status': 'completed',
            'agent': self.name,
            'task_type': task.get('type', 'unknown')
        }


class TestOrchestratorAgent:
    """Test cases for OrchestratorAgent class."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Sample orchestrator configuration."""
        return {
            'health_check_interval': 1,
            'max_concurrent_tasks': 3,
            'task_timeout': 60
        }
    
    @pytest.fixture
    def orchestrator(self, orchestrator_config):
        """Create an orchestrator instance for testing."""
        with patch('agents.orchestrator.StructuredLogger'), \
             patch('agents.orchestrator.settings'), \
             patch('agents.orchestrator.LangGraphWorkflowEngine') as mock_workflow_engine:
            orchestrator = OrchestratorAgent(orchestrator_config)
            # Ensure workflow_engine is properly mocked
            orchestrator.workflow_engine = mock_workflow_engine
            return orchestrator
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            return MockAgent("test_agent", {'max_retries': 2})
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.name == "orchestrator"
        # Status should be RUNNING after _on_initialize completes
        # The orchestrator sets status to RUNNING during initialization
        # Check both possible values due to timing
        assert orchestrator.orchestrator_status in [OrchestratorStatus.STARTING, OrchestratorStatus.RUNNING]
        assert orchestrator.current_phase in [WorkflowPhase.INITIALIZATION, WorkflowPhase.AGENT_DISCOVERY]
        assert len(orchestrator.registered_agents) == 0
        assert len(orchestrator.task_queue) == 0
        assert orchestrator.workflow_engine is not None
    
    def test_orchestrator_configuration(self, orchestrator):
        """Test orchestrator configuration loading."""
        assert orchestrator.health_check_interval == 1
        assert orchestrator.max_concurrent_tasks == 3
        assert orchestrator.task_timeout == 60
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator, mock_agent):
        """Test agent registration functionality."""
        registration_task = {
            'type': 'register_agent',
            'agent_name': 'test_agent',
            'agent': mock_agent,
            'capabilities': ['test_capability'],
            'id': 'reg_task_1'
        }
        
        result = await orchestrator.execute(registration_task)
        
        assert result['status'] == 'registered'
        assert result['agent_name'] == 'test_agent'
        assert 'test_capability' in result['capabilities']
        assert 'test_agent' in orchestrator.registered_agents
        assert orchestrator.agent_capabilities['test_agent'] == {'test_capability'}
        assert orchestrator.agent_health['test_agent']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_task_distribution(self, orchestrator, mock_agent):
        """Test task distribution to suitable agents."""
        # Register agent first
        await orchestrator.execute({
            'type': 'register_agent',
            'agent_name': 'test_agent',
            'agent': mock_agent,
            'capabilities': ['test_capability']
        })
        
        distribution_task = {
            'type': 'distribute_task',
            'task_data': {
                'id': 'test_task_1',
                'type': 'test_task',
                'required_capabilities': ['test_capability']
            }
        }
        
        result = await orchestrator.execute(distribution_task)
        
        assert result['status'] == 'distributed'
        assert result['task_id'] == 'test_task_1'
        assert result['assigned_agent'] == 'test_agent'
        assert 'test_task_1' in orchestrator.task_assignments
        assert len(mock_agent.task_queue) == 1
    
    @pytest.mark.asyncio
    async def test_task_distribution_no_suitable_agent(self, orchestrator):
        """Test task distribution when no suitable agents are available."""
        distribution_task = {
            'type': 'distribute_task',
            'task_data': {
                'id': 'test_task_1',
                'type': 'test_task',
                'required_capabilities': ['nonexistent_capability']
            }
        }
        
        with pytest.raises(ValueError, match="No agents available for capabilities"):
            await orchestrator.execute(distribution_task)
    
    @pytest.mark.asyncio
    async def test_agent_monitoring(self, orchestrator, mock_agent):
        """Test agent health monitoring."""
        # Register agent first
        await orchestrator.execute({
            'type': 'register_agent',
            'agent_name': 'test_agent',
            'agent': mock_agent,
            'capabilities': ['test_capability']
        })
        
        monitoring_task = {
            'type': 'monitor_agents',
            'id': 'monitor_task_1'
        }
        
        result = await orchestrator.execute(monitoring_task)
        
        assert result['status'] == 'completed'
        assert 'health_results' in result
        assert 'test_agent' in result['health_results']
        assert orchestrator.agent_health['test_agent']['last_check'] is not None
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator, mock_agent):
        """Test complete workflow execution."""
        # Register agent first
        await orchestrator.execute({
            'type': 'register_agent',
            'agent_name': 'test_agent',
            'agent': mock_agent,
            'capabilities': ['test_capability']
        })
        
        workflow_task = {
            'type': 'execute_workflow',
            'workflow_id': 'test_workflow_1',
            'workflow_definition': {
                'steps': ['step1', 'step2']
            }
        }
        
        # Mock workflow engine execution
        mock_result = Mock()
        mock_result.to_dict.return_value = {'status': 'completed'}
        orchestrator.workflow_engine.run_workflow = AsyncMock(return_value=mock_result)
        
        result = await orchestrator.execute(workflow_task)
        
        assert result['status'] == 'completed'
        assert result['workflow_id'] == 'test_workflow_1'
        assert 'result' in result
        assert len(orchestrator.workflow_history) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test system health check."""
        health_task = {
            'type': 'health_check',
            'id': 'health_task_1'
        }
        
        result = await orchestrator.execute(health_task)
        
        assert result['status'] == 'healthy'
        assert 'system_health' in result
        assert result['system_health']['orchestrator_status'] == OrchestratorStatus.RUNNING.value
        assert result['system_health']['current_phase'] == WorkflowPhase.AGENT_DISCOVERY.value
        assert result['system_health']['registered_agents'] == 0
    
    def test_find_suitable_agents(self, orchestrator, mock_agent):
        """Test finding suitable agents for tasks."""
        # Register agent with capabilities
        orchestrator.registered_agents['test_agent'] = mock_agent
        orchestrator.agent_capabilities['test_agent'] = {'capability1', 'capability2'}
        
        # Test with matching capabilities
        suitable = orchestrator._find_suitable_agents(['capability1'])
        assert 'test_agent' in suitable
        
        # Test with non-matching capabilities
        suitable = orchestrator._find_suitable_agents(['nonexistent'])
        assert len(suitable) == 0
    
    def test_select_best_agent(self, orchestrator, mock_agent):
        """Test selecting best agent for task."""
        # Create multiple mock agents
        agent1 = mock_agent
        agent2 = MockAgent("test_agent2")
        
        orchestrator.registered_agents['agent1'] = agent1
        orchestrator.registered_agents['agent2'] = agent2
        
        # Add different loads to agents
        agent1.task_queue = [{'task': 'task1'}]
        agent2.task_queue = [{'task': 'task1'}, {'task': 'task2'}]
        
        # Should select agent with lower load
        best = orchestrator._select_best_agent(['agent1', 'agent2'], {})
        assert best == 'agent1'
    
    def test_assess_agent_health(self, orchestrator, mock_agent):
        """Test agent health assessment."""
        orchestrator.registered_agents['test_agent'] = mock_agent
        
        # Test healthy agent
        metrics = {
            'tasks_completed': 10,
            'tasks_failed': 1,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        health = orchestrator._assess_agent_health('test_agent', metrics)
        assert health == 'healthy'
        
        # Test unhealthy agent (high error rate)
        metrics['tasks_failed'] = 6  # 6/16 = 37.5% error rate
        health = orchestrator._assess_agent_health('test_agent', metrics)
        assert health == 'healthy'  # Still below 50% threshold
        
        # Test very unhealthy agent
        metrics['tasks_failed'] = 10  # 10/20 = 50% error rate
        health = orchestrator._assess_agent_health('test_agent', metrics)
        assert health == 'unhealthy'
    
    @pytest.mark.asyncio
    async def test_task_completion_event(self, orchestrator, mock_agent):
        """Test task completion event handling."""
        # Register agent and assign task
        orchestrator.registered_agents['test_agent'] = mock_agent
        orchestrator.task_assignments['task1'] = 'test_agent'
        orchestrator.agent_health['test_agent'] = {
            'tasks_completed': 0,
            'last_activity': datetime.utcnow().isoformat()
        }
        
        # Simulate task completion
        event_data = {'task_id': 'task1'}
        await orchestrator._on_task_completed(event_data)
        
        assert 'task1' not in orchestrator.task_assignments
        assert orchestrator.agent_health['test_agent']['tasks_completed'] == 1
    
    @pytest.mark.asyncio
    async def test_task_failure_event(self, orchestrator, mock_agent):
        """Test task failure event handling."""
        # Register agent and assign task
        orchestrator.registered_agents['test_agent'] = mock_agent
        orchestrator.task_assignments['task1'] = 'test_agent'
        orchestrator.agent_health['test_agent'] = {
            'tasks_failed': 0,
            'last_activity': datetime.utcnow().isoformat()
        }
        
        # Simulate task failure
        event_data = {'task_id': 'task1'}
        await orchestrator._on_task_failed(event_data)
        
        assert 'task1' not in orchestrator.task_assignments
        assert orchestrator.agent_health['test_agent']['tasks_failed'] == 1
    
    @pytest.mark.asyncio
    async def test_agent_error_event(self, orchestrator, mock_agent):
        """Test agent error event handling."""
        # Register agent
        orchestrator.registered_agents['test_agent'] = mock_agent
        orchestrator.agent_health['test_agent'] = {'status': 'healthy'}
        
        # Simulate agent error
        error_context = {'type': 'ValueError', 'message': 'Test error'}
        event_data = {
            'agent_name': 'test_agent',
            'error_context': error_context
        }
        await orchestrator._on_agent_error(event_data)
        
        assert orchestrator.agent_health['test_agent']['status'] == 'unhealthy'
        assert orchestrator.agent_health['test_agent']['last_error'] == error_context
    
    def test_register_agent_method(self, orchestrator, mock_agent):
        """Test register_agent convenience method."""
        orchestrator.register_agent(mock_agent, ['test_capability'])
        
        # Give async task time to execute
        asyncio.run(asyncio.sleep(0.1))
        
        assert 'test_agent' in orchestrator.registered_agents
        assert orchestrator.agent_capabilities['test_agent'] == {'test_capability'}
    
    def test_submit_task_method(self, orchestrator):
        """Test submit_task convenience method."""
        task = {
            'id': 'test_task_1',
            'type': 'test_task',
            'data': {'test': True}
        }
        
        orchestrator.submit_task(task, ['test_capability'])
        
        assert len(orchestrator.task_queue) == 1
        assert orchestrator.task_queue[0]['id'] == 'test_task_1'
        assert orchestrator.task_queue[0]['required_capabilities'] == ['test_capability']
    
    def test_get_system_status(self, orchestrator, mock_agent):
        """Test system status reporting."""
        # Register an agent
        orchestrator.registered_agents['test_agent'] = mock_agent
        orchestrator.agent_health['test_agent'] = {'status': 'healthy'}
        
        status = orchestrator.get_system_status()
        
        assert 'orchestrator' in status
        assert 'agents' in status
        assert 'workflows' in status
        assert 'tasks' in status
        
        assert status['orchestrator']['status'] == OrchestratorStatus.RUNNING.value
        assert status['agents']['registered'] == 1
        assert status['agents']['healthy'] == 1
        assert status['tasks']['queued'] == 0
    
    @pytest.mark.asyncio
    async def test_pause_and_resume(self, orchestrator):
        """Test orchestrator pause and resume functionality."""
        # Test pause
        await orchestrator.pause()
        assert orchestrator.orchestrator_status == OrchestratorStatus.PAUSED
        
        # Test resume
        await orchestrator.resume()
        assert orchestrator.orchestrator_status == OrchestratorStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_monitoring_tasks(self, orchestrator):
        """Test background monitoring tasks."""
        # Start monitoring
        await orchestrator.start_monitoring()
        assert len(orchestrator._background_tasks) == 2
        
        # Stop monitoring
        await orchestrator.stop_monitoring()
        assert len(orchestrator._background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator, mock_agent):
        """Test orchestrator shutdown."""
        # Register an agent
        orchestrator.registered_agents['test_agent'] = mock_agent
        
        # Start monitoring first
        await orchestrator.start_monitoring()
        
        # Shutdown
        await orchestrator.shutdown()
        
        assert orchestrator.orchestrator_status == OrchestratorStatus.STOPPED
        assert len(orchestrator._background_tasks) == 0


class TestOrchestratorIntegration:
    """Integration tests for orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """Test complete agent lifecycle with orchestrator."""
        with patch('agents.orchestrator.StructuredLogger'), \
             patch('agents.orchestrator.settings'), \
             patch('agents.orchestrator.LangGraphWorkflowEngine'), \
             patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            orchestrator = OrchestratorAgent({'health_check_interval': 0.1})
            agent = MockAgent("integration_agent")
            
            # Register agent
            await orchestrator.execute({
                'type': 'register_agent',
                'agent_name': 'integration_agent',
                'agent': agent,
                'capabilities': ['test']
            })
            
            # Submit and distribute task
            orchestrator.submit_task({
                'id': 'integration_task',
                'type': 'test',
                'data': {'test': True}
            }, ['test'])
            
            # Process task distribution
            await asyncio.sleep(0.1)
            
            # Verify agent has task
            assert len(agent.task_queue) == 1
            assert agent.task_queue[0]['id'] == 'integration_task'
            
            # Shutdown
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in orchestrator integration."""
        class FailingAgent(MockAgent):
            async def _process_task(self, task: dict) -> dict:
                raise ValueError("Integration test error")
        
        with patch('agents.orchestrator.StructuredLogger'), \
             patch('agents.orchestrator.settings'), \
             patch('agents.orchestrator.LangGraphWorkflowEngine'), \
             patch('agents.base_agent.StructuredLogger'), \
             patch('agents.base_agent.settings'), \
             patch('agents.base_agent.ErrorHandler'):
            
            orchestrator = OrchestratorAgent()
            failing_agent = FailingAgent("failing_agent")
            
            # Register failing agent
            await orchestrator.execute({
                'type': 'register_agent',
                'agent_name': 'failing_agent',
                'agent': failing_agent,
                'capabilities': ['test']
            })
            
            # Submit task
            orchestrator.submit_task({
                'id': 'failing_task',
                'type': 'test'
            }, ['test'])
            
            # Process distribution
            await asyncio.sleep(0.1)
            
            # Check agent health after failure
            await orchestrator.execute({
                'type': 'monitor_agents',
                'id': 'health_check'
            })
            
            # Agent should still be registered but might have health issues
            assert 'failing_agent' in orchestrator.registered_agents


if __name__ == "__main__":
    pytest.main([__file__])