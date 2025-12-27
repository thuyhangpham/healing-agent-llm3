"""
Tests for Agent Registry and Discovery System

Test suite for the centralized agent registry that handles
registration, discovery, health monitoring, and configuration.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from utils.agent_registry import (
    AgentRegistry, AgentInfo, AgentRegistrationStatus,
    get_agent_registry, register_agent, unregister_agent,
    discover_agents_by_capability, discover_agents_by_capabilities
)


class TestAgentInfo:
    """Test cases for AgentInfo dataclass."""
    
    def test_agent_info_creation(self):
        """Test AgentInfo creation."""
        timestamp = datetime.utcnow()
        capabilities = {"test", "processing"}
        
        info = AgentInfo(
            name="test_agent",
            agent_type="test",
            capabilities=capabilities,
            status=AgentRegistrationStatus.REGISTERED,
            registered_at=timestamp,
            config={"key": "value"},
            metadata={"version": "1.0"}
        )
        
        assert info.name == "test_agent"
        assert info.agent_type == "test"
        assert info.capabilities == capabilities
        assert info.status == AgentRegistrationStatus.REGISTERED
        assert info.registered_at == timestamp
        assert info.config == {"key": "value"}
        assert info.metadata == {"version": "1.0"}
        assert info.health_status == "unknown"
        assert info.performance_metrics is None


class TestAgentRegistry:
    """Test cases for AgentRegistry class."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary agent configuration file."""
        config_data = {
            'agents': {
                'test_agent1': {
                    'type': 'test',
                    'enabled': True,
                    'capabilities': ['test', 'processing'],
                    'config': {'key1': 'value1'}
                },
                'test_agent2': {
                    'type': 'test',
                    'enabled': False,
                    'capabilities': ['analysis'],
                    'config': {'key2': 'value2'}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    @pytest.fixture
    def agent_registry(self, temp_config_file):
        """Create an agent registry instance for testing."""
        with patch('utils.agent_registry.StructuredLogger'), \
             patch('utils.agent_registry.settings'):
            return AgentRegistry(config_file=temp_config_file)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.name = "mock_agent"
        agent.get_status = Mock(return_value="idle")
        return agent
    
    def test_registry_initialization(self, agent_registry):
        """Test registry initialization."""
        assert len(agent_registry._agents) == 0
        assert len(agent_registry._capabilities_index) == 0
        assert len(agent_registry.agent_configs) == 2  # From temp_config_file
        assert agent_registry.config_file.endswith('.yaml')
    
    def test_register_agent_success(self, agent_registry, mock_agent):
        """Test successful agent registration."""
        capabilities = ["test", "processing"]
        config_overrides = {"override_key": "override_value"}
        
        result = agent_registry.register_agent(
            mock_agent, capabilities, config_overrides
        )
        
        print(f"Registration result: {result}")
        print(f"Registry agents: {list(agent_registry._agents.keys())}")
        
        assert result is True
        assert "mock_agent" in agent_registry._agents
        
        agent_info = agent_registry.get_agent_info("mock_agent")
        assert agent_info.name == "mock_agent"
        assert agent_info.capabilities == set(capabilities)
        assert agent_info.status == AgentRegistrationStatus.REGISTERED
        assert agent_info.config["override_key"] == "override_value"
        assert "test" in agent_registry._capabilities_index
        assert "processing" in agent_registry._capabilities_index
        assert "mock_agent" in agent_registry._capabilities_index["test"]
        assert "mock_agent" in agent_registry._capabilities_index["processing"]
    
    def test_register_duplicate_agent(self, agent_registry, mock_agent):
        """Test registering the same agent twice."""
        # First registration should succeed
        result1 = agent_registry.register_agent(mock_agent, ["test"])
        assert result1 is True
        
        # Second registration should fail
        result2 = agent_registry.register_agent(mock_agent, ["test"])
        assert result2 is False
    
    def test_unregister_agent_success(self, agent_registry, mock_agent):
        """Test successful agent unregistration."""
        # Register first
        agent_registry.register_agent(mock_agent, ["test"])
        assert "mock_agent" in agent_registry._agents
        
        # Unregister
        result = agent_registry.unregister_agent("mock_agent", "test reason")
        assert result is True
        assert "mock_agent" not in agent_registry._agents
        assert "mock_agent" not in agent_registry._capabilities_index.get("test", set())
    
    def test_unregister_nonexistent_agent(self, agent_registry):
        """Test unregistering a non-existent agent."""
        result = agent_registry.unregister_agent("nonexistent", "test")
        assert result is False
    
    def test_discover_agents_by_capability(self, agent_registry, mock_agent):
        """Test discovering agents by capability."""
        # Register agents with different capabilities
        agent_registry.register_agent(mock_agent, ["test", "processing"])
        
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["test", "analysis"])
        
        agent3 = Mock()
        agent3.name = "agent3"
        agent_registry.register_agent(agent3, ["processing"])
        
        # Discover by "test" capability
        test_agents = agent_registry.discover_agents_by_capability("test")
        assert len(test_agents) == 2
        assert "mock_agent" in test_agents
        assert "agent2" in test_agents
        assert "agent3" not in test_agents
        
        # Discover by "processing" capability
        processing_agents = agent_registry.discover_agents_by_capability("processing")
        assert len(processing_agents) == 2
        assert "mock_agent" in processing_agents
        assert "agent3" in processing_agents
        assert "agent2" not in processing_agents
        
        # Discover by non-existent capability
        no_agents = agent_registry.discover_agents_by_capability("nonexistent")
        assert len(no_agents) == 0
    
    def test_discover_agents_by_capabilities_require_all(self, agent_registry, mock_agent):
        """Test discovering agents with all required capabilities."""
        # Register agents
        agent_registry.register_agent(mock_agent, ["test", "processing"])
        
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["test", "analysis"])
        
        agent3 = Mock()
        agent3.name = "agent3"
        agent_registry.register_agent(agent3, ["test", "processing", "analysis"])
        
        # Require all capabilities (agent3 should match)
        all_capable = agent_registry.discover_agents_by_capabilities(
            ["test", "processing", "analysis"], require_all=True
        )
        assert len(all_capable) == 1
        assert "agent3" in all_capable
        
        # Require at least one capability (all should match)
        any_capable = agent_registry.discover_agents_by_capabilities(
            ["test", "processing", "analysis"], require_all=False
        )
        assert len(any_capable) == 3
        assert "mock_agent" in any_capable
        assert "agent2" in any_capable
        assert "agent3" in any_capable
    
    def test_get_agent_info(self, agent_registry, mock_agent):
        """Test getting agent information."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        # Get existing agent info
        info = agent_registry.get_agent_info("mock_agent")
        assert info is not None
        assert info.name == "mock_agent"
        assert info.status == AgentRegistrationStatus.REGISTERED
        
        # Get non-existent agent info
        info = agent_registry.get_agent_info("nonexistent")
        assert info is None
    
    def test_get_all_agents(self, agent_registry, mock_agent):
        """Test getting all agents with filtering."""
        # Register agents
        agent_registry.register_agent(mock_agent, ["test"])
        
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["analysis"])
        
        # Get all agents
        all_agents = agent_registry.get_all_agents()
        assert len(all_agents) == 2
        assert "mock_agent" in all_agents
        assert "agent2" in all_agents
        
        # Filter by status
        registered_agents = agent_registry.get_all_agents(
            status_filter=[AgentRegistrationStatus.REGISTERED]
        )
        assert len(registered_agents) == 2
        
        # Unregister one agent
        agent_registry.unregister_agent("agent2")
        registered_only = agent_registry.get_all_agents(
            status_filter=[AgentRegistrationStatus.REGISTERED]
        )
        assert len(registered_only) == 1
        assert "mock_agent" in registered_only
    
    def test_update_agent_heartbeat(self, agent_registry, mock_agent):
        """Test updating agent heartbeat."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        initial_info = agent_registry.get_agent_info("mock_agent")
        assert initial_info.last_heartbeat is None
        
        # Update heartbeat
        result = agent_registry.update_agent_heartbeat("mock_agent")
        assert result is True
        
        updated_info = agent_registry.get_agent_info("mock_agent")
        assert updated_info.last_heartbeat is not None
        assert updated_info.last_heartbeat > initial_info.registered_at
        
        # Update heartbeat for non-existent agent
        result = agent_registry.update_agent_heartbeat("nonexistent")
        assert result is False
    
    def test_update_agent_health(self, agent_registry, mock_agent):
        """Test updating agent health."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        metrics = {"tasks_completed": 10, "tasks_failed": 2}
        result = agent_registry.update_agent_health(
            "mock_agent", "healthy", metrics
        )
        assert result is True
        
        info = agent_registry.get_agent_info("mock_agent")
        assert info.health_status == "healthy"
        assert info.performance_metrics == metrics
        
        # Update health for non-existent agent
        result = agent_registry.update_agent_health(
            "nonexistent", "healthy", {}
        )
        assert result is False
    
    def test_get_agent_config(self, agent_registry):
        """Test getting agent configuration."""
        config = agent_registry.get_agent_config("test_agent1")
        assert config["type"] == "test"
        assert config["enabled"] is True
        assert config["capabilities"] == ["test", "processing"]
        assert config["config"]["key1"] == "value1"
        
        # Get non-existent agent config
        config = agent_registry.get_agent_config("nonexistent")
        assert config == {}
    
    def test_is_agent_enabled(self, agent_registry):
        """Test checking if agent is enabled."""
        assert agent_registry.is_agent_enabled("test_agent1") is True
        assert agent_registry.is_agent_enabled("test_agent2") is False
        assert agent_registry.is_agent_enabled("nonexistent") is True  # Default
    
    def test_get_capabilities_summary(self, agent_registry, mock_agent):
        """Test getting capabilities summary."""
        # Register agents
        agent_registry.register_agent(mock_agent, ["test", "processing"])
        
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["test", "analysis"])
        
        agent3 = Mock()
        agent3.name = "agent3"
        agent_registry.register_agent(agent3, ["processing"])
        
        summary = agent_registry.get_capabilities_summary()
        assert summary["test"] == 2  # mock_agent and agent2
        assert summary["processing"] == 2  # mock_agent and agent3
        assert summary["analysis"] == 1  # agent2 only
    
    def test_event_listeners(self, agent_registry, mock_agent):
        """Test agent registry event system."""
        events_received = []
        
        def test_listener(agent_info):
            events_received.append(agent_info)
        
        # Add listener
        agent_registry.add_event_listener("registered", test_listener)
        
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        # Check event was received
        assert len(events_received) == 1
        assert events_received[0].name == "mock_agent"
        assert events_received[0].status == AgentRegistrationStatus.REGISTERED
        
        # Remove listener
        agent_registry.remove_event_listener("registered", test_listener)
        
        # Register another agent
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["analysis"])
        
        # Should not receive event
        assert len(events_received) == 1
    
    def test_health_monitoring(self, agent_registry, mock_agent):
        """Test health monitoring functionality."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        # Start monitoring
        asyncio.run(agent_registry.start_health_monitoring())
        assert agent_registry._monitoring_task is not None
        
        # Stop monitoring
        asyncio.run(agent_registry.stop_health_monitoring())
        assert agent_registry._monitoring_task is None
    
    def test_get_registry_stats(self, agent_registry, mock_agent):
        """Test getting registry statistics."""
        # Register some agents
        agent_registry.register_agent(mock_agent, ["test"])
        
        agent2 = Mock()
        agent2.name = "agent2"
        agent_registry.register_agent(agent2, ["analysis"])
        
        agent3 = Mock()
        agent3.name = "agent3"
        agent_registry.register_agent(agent3, ["processing"])
        
        # Unregister one
        agent_registry.unregister_agent("agent2")
        
        stats = agent_registry.get_registry_stats()
        assert stats["total_agents"] == 2
        assert stats["registered_agents"] == 2
        assert stats["unregistered_agents"] == 1
        assert stats["suspended_agents"] == 0
        assert stats["error_agents"] == 0
        assert stats["capabilities_count"] == 2  # test and processing
        assert "test" in stats["capabilities_breakdown"]
        assert "processing" in stats["capabilities_breakdown"]
        assert "analysis" not in stats["capabilities_breakdown"]
    
    def test_export_registry(self, agent_registry, mock_agent):
        """Test registry export functionality."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
            
            result = agent_registry.export_registry(temp_file)
            assert result is True
            
            # Check file was created and has content
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
            
            # Cleanup
            os.unlink(temp_file)
    
    def test_cleanup_stale_agents(self, agent_registry, mock_agent):
        """Test cleanup of stale agents."""
        # Register agent
        agent_registry.register_agent(mock_agent, ["test"])
        
        # Manually set old heartbeat
        info = agent_registry.get_agent_info("mock_agent")
        old_time = datetime.utcnow() - timedelta(hours=25)
        info.last_heartbeat = old_time
        
        # Run cleanup with 24 hour threshold
        cleaned_count = asyncio.run(
            agent_registry.cleanup_stale_agents(max_age_hours=24)
        )
        
        assert cleaned_count == 1
        assert agent_registry.get_agent_info("mock_agent") is None


class TestAgentRegistryIntegration:
    """Integration tests for agent registry."""
    
    def test_singleton_pattern(self):
        """Test singleton pattern for agent registry."""
        with patch('utils.agent_registry.StructuredLogger'), \
             patch('utils.agent_registry.settings'):
            
            # Reset singleton
            import utils.agent_registry
            utils.agent_registry._global_registry = None
            
            # Get instance twice
            registry1 = get_agent_registry()
            registry2 = get_agent_registry()
            
            # Should be same instance
            assert registry1 is registry2
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('utils.agent_registry.StructuredLogger'), \
             patch('utils.agent_registry.settings'), \
             patch('utils.agent_registry.BaseAgent'):
            
            # Reset singleton
            import utils.agent_registry
            utils.agent_registry._global_registry = None
            
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            
            # Test convenience functions
            result = register_agent(mock_agent, ["test"])
            assert result is True
            
            agents = discover_agents_by_capability("test")
            assert "test_agent" in agents
            
            multi_capable = discover_agents_by_capabilities(["test", "other"])
            assert "test_agent" in multi_capable
            
            unregister_result = unregister_agent("test_agent")
            assert unregister_result is True


if __name__ == "__main__":
    pytest.main([__file__])