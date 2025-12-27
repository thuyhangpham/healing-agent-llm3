"""
Tests for LangGraph Workflow Engine

Test suite for the core workflow orchestration system.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from utils.workflow_engine import WorkflowState, LangGraphWorkflowEngine


class TestWorkflowState:
    """Test cases for WorkflowState class."""
    
    def test_workflow_state_initialization(self):
        """Test WorkflowState initialization."""
        state = WorkflowState()
        
        assert state.agents == {}
        assert state.active_workflows == []
        assert state.current_task is None
        assert state.workflow_history == []
        assert state.errors == []
        assert state.healing_requests == []
        assert state.start_time is not None
        assert state.last_update is not None
    
    def test_register_agent(self):
        """Test agent registration."""
        state = WorkflowState()
        agent_info = {"type": "test", "status": "active"}
        
        state.register_agent("test_agent", agent_info)
        
        assert "test_agent" in state.agents
        assert state.agents["test_agent"] == agent_info
    
    def test_update_agent_status(self):
        """Test agent status update."""
        state = WorkflowState()
        agent_info = {"type": "test", "status": "active"}
        state.register_agent("test_agent", agent_info)
        
        state.update_agent_status("test_agent", "working", {"task": "test_task"})
        
        assert state.agents["test_agent"]["status"] == "working"
        assert state.agents["test_agent"]["task"] == "test_task"
    
    def test_add_workflow_step(self):
        """Test adding workflow steps."""
        state = WorkflowState()
        step_details = {"action": "test", "result": "success"}
        
        state.add_workflow_step("test_step", step_details)
        
        assert len(state.workflow_history) == 1
        assert state.workflow_history[0]["type"] == "test_step"
        assert state.workflow_history[0]["details"] == step_details
    
    def test_set_and_clear_current_task(self):
        """Test setting and clearing current task."""
        state = WorkflowState()
        task = {"agent": "test_agent", "type": "test_task"}
        
        state.set_current_task(task)
        assert state.current_task == task
        
        state.clear_current_task()
        assert state.current_task is None
    
    def test_error_and_healing_request_management(self):
        """Test error and healing request management."""
        state = WorkflowState()
        error = {"type": "test_error", "message": "Test error"}
        healing_request = {"id": "req1", "type": "heal"}
        
        state.add_error(error)
        state.add_healing_request(healing_request)
        
        assert len(state.errors) == 1
        assert state.errors[0] == error
        assert len(state.healing_requests) == 1
        assert state.healing_requests[0] == healing_request
    
    def test_get_agent_status(self):
        """Test getting agent status."""
        state = WorkflowState()
        agent_info = {"type": "test", "status": "active"}
        state.register_agent("test_agent", agent_info)
        
        status = state.get_agent_status("test_agent")
        assert status == "active"
        
        unknown_status = state.get_agent_status("unknown_agent")
        assert unknown_status == "unknown"
    
    def test_get_active_agents(self):
        """Test getting active agents."""
        state = WorkflowState()
        state.register_agent("active_agent", {"status": "active"})
        state.register_agent("inactive_agent", {"status": "inactive"})
        
        active_agents = state.get_active_agents()
        assert "active_agent" in active_agents
        assert "inactive_agent" not in active_agents
    
    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = WorkflowState()
        state.register_agent("test_agent", {"status": "active"})
        
        state_dict = state.to_dict()
        
        assert "agents" in state_dict
        assert "active_workflows" in state_dict
        assert "current_task" in state_dict
        assert "workflow_history" in state_dict
        assert "errors" in state_dict
        assert "healing_requests" in state_dict
        assert "start_time" in state_dict
        assert "last_update" in state_dict


class TestLangGraphWorkflowEngine:
    """Test cases for LangGraphWorkflowEngine class."""
    
    @pytest.fixture
    def workflow_engine(self):
        """Create a workflow engine instance for testing."""
        with patch('utils.workflow_engine.StructuredLogger'), \
             patch('utils.workflow_engine.settings'):
            return LangGraphWorkflowEngine()
    
    def test_workflow_engine_initialization(self, workflow_engine):
        """Test workflow engine initialization."""
        assert workflow_engine.state is not None
        assert workflow_engine.graph is None
        assert workflow_engine.compiled_graph is None
        assert workflow_engine.memory is not None
    
    def test_create_workflow_graph(self, workflow_engine):
        """Test creating workflow graph."""
        graph = workflow_engine.create_workflow_graph()
        
        assert graph is not None
        assert workflow_engine.graph == graph
    
    def test_register_agent(self, workflow_engine):
        """Test registering an agent."""
        agent_info = {"type": "test", "status": "active"}
        
        workflow_engine.register_agent("test_agent", agent_info)
        
        assert "test_agent" in workflow_engine.state.agents
        assert workflow_engine.state.agents["test_agent"] == agent_info
    
    def test_get_agent_status(self, workflow_engine):
        """Test getting agent status."""
        agent_info = {"type": "test", "status": "active"}
        workflow_engine.register_agent("test_agent", agent_info)
        
        status = workflow_engine.get_agent_status("test_agent")
        assert status == "active"
    
    def test_compile_workflow(self, workflow_engine):
        """Test compiling workflow."""
        workflow_engine.create_workflow_graph()
        compiled = workflow_engine.compile_workflow()
        
        assert compiled is not None
        assert workflow_engine.compiled_graph == compiled
    
    def test_get_workflow_state(self, workflow_engine):
        """Test getting workflow state."""
        state_dict = workflow_engine.get_workflow_state()
        
        assert isinstance(state_dict, dict)
        assert "agents" in state_dict
        assert "active_workflows" in state_dict
    
    @pytest.mark.asyncio
    async def test_run_workflow_basic(self, workflow_engine):
        """Test basic workflow execution."""
        # Mock the compiled graph
        mock_result = Mock()
        mock_result.to_dict.return_value = {"status": "completed"}
        
        from unittest.mock import AsyncMock
        with patch.object(workflow_engine, 'compiled_graph') as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)
            
            result = await workflow_engine.run_workflow()
            
            assert result is not None
            mock_graph.ainvoke.assert_called_once()
    
    def test_workflow_nodes(self, workflow_engine):
        """Test workflow node functions."""
        state = WorkflowState()
        
        # Test start workflow
        result_state = workflow_engine._start_workflow(state)
        assert "main" in result_state.active_workflows
        assert result_state.start_time is not None
        
        # Test execute task with no task
        result_state = workflow_engine._execute_task(state)
        assert result_state.current_task is None
        
        # Test execute task with task
        task = {"agent": "test_agent", "type": "test_task"}
        state.set_current_task(task)
        result_state = workflow_engine._execute_task(state)
        assert result_state.current_task is None
        
        # Test check agents
        result_state = workflow_engine._check_agents(state)
        assert result_state is not None
        
        # Test handle healing request
        healing_request = {"id": "req1", "type": "heal"}
        state.add_healing_request(healing_request)
        result_state = workflow_engine._handle_healing_request(state)
        assert len(result_state.healing_requests) == 0
        
        # Test handle error
        error = {"type": "test_error", "message": "Test error"}
        state.add_error(error)
        result_state = workflow_engine._handle_error(state)
        assert len(result_state.errors) == 0
        
        # Test end workflow
        state.start_time = datetime.utcnow()
        result_state = workflow_engine._end_workflow(state)
        assert result_state is not None


class TestWorkflowIntegration:
    """Integration tests for the workflow system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution(self):
        """Test full workflow execution with multiple agents."""
        with patch('utils.workflow_engine.StructuredLogger'), \
             patch('utils.workflow_engine.settings'):
            
            engine = LangGraphWorkflowEngine()
            
            # Register agents
            engine.register_agent("data_agent", {"type": "data", "status": "active"})
            engine.register_agent("sentiment_agent", {"type": "sentiment", "status": "active"})
            
            # Create and compile workflow
            engine.create_workflow_graph()
            engine.compile_workflow()
            
            # Mock the execution to avoid actual LangGraph complexity
            from unittest.mock import AsyncMock
            with patch.object(engine.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
                mock_state = WorkflowState()
                mock_state.register_agent("data_agent", {"type": "data", "status": "completed"})
                mock_state.register_agent("sentiment_agent", {"type": "sentiment", "status": "completed"})
                mock_invoke.return_value = mock_state
                
                result = await engine.run_workflow()
                
                assert result is not None
                assert len(result.agents) == 2
    
    def test_workflow_state_persistence(self):
        """Test workflow state management and persistence."""
        state = WorkflowState()
        
        # Add various data to state
        state.register_agent("agent1", {"type": "test", "status": "active"})
        state.add_workflow_step("test_step", {"action": "test"})
        state.set_current_task({"agent": "agent1", "type": "test_task"})
        state.add_error({"type": "test_error"})
        state.add_healing_request({"id": "req1"})
        
        # Convert to dict and verify structure
        state_dict = state.to_dict()
        
        assert len(state_dict["agents"]) == 1
        assert len(state_dict["workflow_history"]) == 1
        assert state_dict["current_task"] is not None
        assert len(state_dict["errors"]) == 1
        assert len(state_dict["healing_requests"]) == 1
        assert state_dict["start_time"] is not None
        assert state_dict["last_update"] is not None


if __name__ == "__main__":
    pytest.main([__file__])