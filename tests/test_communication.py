"""
Tests for Inter-Agent Communication System

Test suite for the communication patterns and message broker
that enable agents to exchange information and coordinate tasks.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from utils.communication import (
    AgentMessage, MessageType, MessagePriority, CommunicationProtocol,
    MessageBroker, CommunicationManager,
    get_communication_manager, send_status_update, send_error_report,
    send_heartbeat, broadcast_task_request
)


class TestAgentMessage:
    """Test cases for AgentMessage dataclass."""
    
    def test_message_creation(self):
        """Test AgentMessage creation."""
        timestamp = datetime.utcnow()
        payload = {"key": "value"}
        
        message = AgentMessage(
            id="test_msg_1",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=timestamp,
            priority=MessagePriority.HIGH,
            payload=payload,
            correlation_id="corr_123",
            expires_at=timestamp + timedelta(hours=1),
            metadata={"version": "1.0"}
        )
        
        assert message.id == "test_msg_1"
        assert message.message_type == MessageType.DIRECT_MESSAGE
        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.timestamp == timestamp
        assert message.priority == MessagePriority.HIGH
        assert message.payload == payload
        assert message.correlation_id == "corr_123"
        assert message.expires_at == timestamp + timedelta(hours=1)
        assert message.metadata == {"version": "1.0"}
    
    def test_message_to_dict(self):
        """Test message serialization."""
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.BROADCAST,
            sender="test_agent",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            payload={"test": True}
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["id"] == "test_msg"
        assert message_dict["message_type"] == MessageType.BROADCAST.value
        assert message_dict["sender"] == "test_agent"
        assert message_dict["payload"]["test"] is True
    
    def test_message_expiration(self):
        """Test message expiration logic."""
        # Non-expired message
        future_time = datetime.utcnow() + timedelta(hours=1)
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            expires_at=future_time
        )
        
        assert not message.is_expired()
        
        # Expired message
        past_time = datetime.utcnow() - timedelta(hours=1)
        message.expires_at = past_time
        assert message.is_expired()
        
        # Message without expiration
        message_no_expiration = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow()
        )
        
        assert not message_no_expiration.is_expired()


class TestCommunicationProtocol:
    """Test cases for CommunicationProtocol class."""
    
    @pytest.fixture
    def protocol(self):
        """Create a communication protocol instance."""
        with patch('utils.communication.StructuredLogger'):
            return CommunicationProtocol()
    
    def test_register_handler(self, protocol):
        """Test handler registration."""
        handler = Mock()
        
        protocol.register_handler(MessageType.DIRECT_MESSAGE, handler)
        
        assert MessageType.DIRECT_MESSAGE.value in protocol.message_handlers
        assert handler in protocol.message_handlers[MessageType.DIRECT_MESSAGE.value]
    
    def test_unregister_handler(self, protocol):
        """Test handler unregistration."""
        handler = Mock()
        
        protocol.register_handler(MessageType.DIRECT_MESSAGE, handler)
        protocol.unregister_handler(MessageType.DIRECT_MESSAGE, handler)
        
        assert handler not in protocol.message_handlers.get(MessageType.DIRECT_MESSAGE.value, [])
    
    def test_message_filters(self, protocol):
        """Test message filter functionality."""
        filter_func = Mock(return_value=True)
        filter_func2 = Mock(return_value=False)
        
        protocol.add_message_filter(filter_func)
        protocol.add_message_filter(filter_func2)
        
        assert len(protocol.message_filters) == 2
        
        # Test filtering
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            payload={}
        )
        
        # Should pass first filter, fail second
        assert protocol.should_process_message(message) is False
        
        # Remove second filter
        protocol.remove_message_filter(filter_func2)
        assert protocol.should_process_message(message) is True
    
    @pytest.mark.asyncio
    async def test_handle_message(self, protocol):
        """Test message handling."""
        results = []
        
        async def handler1(message):
            results.append("handler1_result")
            return "handler1_result"
        
        async def handler2(message):
            results.append("handler2_result")
            return "handler2_result"
        
        protocol.register_handler(MessageType.DIRECT_MESSAGE, handler1)
        protocol.register_handler(MessageType.DIRECT_MESSAGE, handler2)
        
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            payload={}
        )
        
        # Should call both handlers
        result = await protocol.handle_message(message)
        
        assert result == ["handler1_result", "handler2_result"]
    
    @pytest.mark.asyncio
    async def test_handle_expired_message(self, protocol):
        """Test handling expired messages."""
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(hours=1)  # Expired
            priority=MessagePriority.MEDIUM,
            payload={}
        
        result = await protocol.handle_message(message)
        
        assert result is None  # Expired messages should not be processed


class TestMessageBroker:
    """Test cases for MessageBroker class."""
    
    @pytest.fixture
    def message_broker(self):
        """Create a message broker instance."""
        with patch('utils.communication.StructuredLogger'):
            return MessageBroker(max_queue_size=100)
    
    def test_agent_registration(self, message_broker):
        """Test agent registration."""
        message_broker.register_agent("agent1", "address1")
        message_broker.register_agent("agent2")  # Uses default address
        
        assert message_broker._agent_addresses["agent1"] == "address1"
        assert message_broker._agent_addresses["agent2"] == "agent://agent2"
    
    def test_agent_unregistration(self, message_broker):
        """Test agent unregistration."""
        message_broker.register_agent("agent1")
        message_broker.unregister_agent("agent1")
        
        assert "agent1" not in message_broker._agent_addresses
    
    def test_message_subscription(self, message_broker):
        """Test message subscription."""
        message_broker.register_agent("agent1")
        message_broker.register_agent("agent2")
        
        # Subscribe agent1 to multiple message types
        message_broker.subscribe("agent1", [
            MessageType.DIRECT_MESSAGE,
            MessageType.BROADCAST,
            MessageType.STATUS_UPDATE
        ])
        
        # Subscribe agent2 to one message type
        message_broker.subscribe("agent2", [MessageType.ERROR_REPORT])
        
        assert len(message_broker._subscriptions["agent1"]) == 3
        assert MessageType.DIRECT_MESSAGE.value in message_broker._subscriptions["agent1"]
        assert MessageType.BROADCAST.value in message_broker._subscriptions["agent1"]
        assert MessageType.STATUS_UPDATE.value in message_broker._subscriptions["agent1"]
        
        assert len(message_broker._subscriptions["agent2"]) == 1
        assert MessageType.ERROR_REPORT.value in message_broker._subscriptions["agent2"]
    
    def test_message_unsubscription(self, message_broker):
        """Test message unsubscription."""
        message_broker.register_agent("agent1")
        message_broker.subscribe("agent1", [
            MessageType.DIRECT_MESSAGE,
            MessageType.BROADCAST
        ])
        
        # Unsubscribe from one type
        message_broker.unsubscribe("agent1", [MessageType.BROADCAST])
        
        assert MessageType.BROADCAST.value not in message_broker._subscriptions["agent1"]
        assert MessageType.DIRECT_MESSAGE.value in message_broker._subscriptions["agent1"]
    
    @pytest.mark.asyncio
    async def test_send_message(self, message_broker):
        """Test message sending."""
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            payload={"test": True}
        )
        
        result = await message_broker.send_message(message)
        
        assert result is True
        assert len(message_broker._message_queue) == 1
        assert message_broker._message_queue[0].id == "test_msg"
        assert message_broker._metrics['messages_sent'] == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_broker):
        """Test broadcast message sending."""
        message_broker.register_agent("agent1")
        message_broker.subscribe("agent1", [MessageType.BROADCAST])
        
        result = await message_broker.broadcast_message(
            sender="agent1",
            message_type=MessageType.STATUS_UPDATE,
            payload={"status": "active"},
            priority=MessagePriority.MEDIUM
        )
        
        assert result is True
        assert len(message_broker._message_queue) == 1
        assert message_broker._message_queue[0].recipient is None  # Broadcast
    
    @pytest.mark.asyncio
    async def test_direct_message(self, message_broker):
        """Test direct message sending."""
        message_broker.register_agent("agent1")
        message_broker.register_agent("agent2")
        message_broker.subscribe("agent2", [MessageType.DIRECT_MESSAGE])
        
        result = await message_broker.send_direct_message(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "test_task"},
            correlation_id="corr_123"
        )
        
        assert result is True
        assert len(message_broker._message_queue) == 1
        assert message_broker._message_queue[0].recipient == "agent2"
        assert message_broker._message_queue[0].correlation_id == "corr_123"
    
    @pytest.mark.asyncio
    async def test_message_queue_processing(self, message_broker):
        """Test message queue processing."""
        message_broker.register_agent("agent1")
        message_broker.subscribe("agent1", [MessageType.DIRECT_MESSAGE])
        
        # Send a message
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent1",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            payload={}
        )
        
        await message_broker.send_message(message)
        
        # Process queue
        await message_broker.process_message_queue()
        
        assert len(message_broker._delivered_messages) == 1
        assert message_broker._metrics['messages_delivered'] == 1
        assert message_broker._metrics['queue_size'] == 0
    
    @pytest.mark.asyncio
    async def test_expired_message_handling(self, message_broker):
        """Test expired message handling."""
        # Send expired message
        expired_message = AgentMessage(
            id="expired_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired
            priority=MessagePriority.HIGH,
            payload={}
        )
        
        await message_broker.send_message(expired_message)
        
        # Process queue
        await message_broker.process_message_queue()
        
        # Expired message should not be delivered
        assert len(message_broker._delivered_messages) == 0
        assert message_broker._metrics['messages_expired'] == 1
    
    def test_get_message_history(self, message_broker):
        """Test getting message history."""
        # Add some delivered messages
        message1 = AgentMessage(
            id="msg1",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            payload={}
        )
        
        message2 = AgentMessage(
            id="msg2",
            message_type=MessageType.BROADCAST,
            sender="agent1",
            recipient=None,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            payload={}
        )
        
        # Simulate delivery
        message_broker._delivered_messages["msg1"] = message1
        message_broker._delivered_messages["msg2"] = message2
        
        # Test history retrieval
        all_messages = message_broker.get_message_history()
        assert len(all_messages) == 2
        
        # Filter by recipient
        agent2_messages = message_broker.get_message_history(agent_name="agent2")
        assert len(agent2_messages) == 1
        assert agent2_messages[0].id == "msg1"
        
        # Filter by message type
        direct_messages = message_broker.get_message_history(message_type=MessageType.DIRECT_MESSAGE)
        assert len(direct_messages) == 1
        assert direct_messages[0].id == "msg1"
        
        # Test limit
        limited_messages = message_broker.get_message_history(limit=1)
        assert len(limited_messages) == 1
    
    def test_queue_status(self, message_broker):
        """Test queue status reporting."""
        # Add some agents and messages
        message_broker.register_agent("agent1")
        message_broker.register_agent("agent2")
        
        message = AgentMessage(
            id="test_msg",
            message_type=MessageType.DIRECT_MESSAGE,
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            payload={}
        )
        
        # Add message to queue
        asyncio.run(message_broker.send_message(message))
        
        status = message_broker.get_queue_status()
        
        assert status['registered_agents'] == 2
        assert status['queue_size'] == 1
        assert status['messages_sent'] == 1
        assert 'oldest_message_age' in status
        assert 'newest_message_age' in status


class TestCommunicationManager:
    """Test cases for CommunicationManager class."""
    
    @pytest.fixture
    def comm_manager(self):
        """Create a communication manager instance."""
        with patch('utils.communication.StructuredLogger'), \
             patch('utils.communication.MessageBroker'), \
             patch('utils.communication.get_global_state_manager'):
            return CommunicationManager()
    
    def test_manager_initialization(self, comm_manager):
        """Test communication manager initialization."""
        assert comm_manager.message_broker is not None
        assert comm_manager.state_manager is not None
        assert comm_manager.message_broker.protocol is not None
    
    @pytest.mark.asyncio
    async def test_status_update_sending(self, comm_manager):
        """Test status update message sending."""
        # Register agent
        comm_manager.register_agent("agent1")
        comm_manager.subscribe_to_messages("agent1", [MessageType.STATUS_UPDATE])
        
        result = await comm_manager.send_status_update(
            sender="orchestrator",
            agent_name="agent1",
            status="active",
            metadata={"version": "1.0"}
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_error_report_sending(self, comm_manager):
        """Test error report message sending."""
        result = await comm_manager.send_error_report(
            sender="agent1",
            error_info={
                "type": "ValueError",
                "message": "Test error",
                "stack_trace": "Error line 1"
            },
            correlation_id="error_123"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_heartbeat_sending(self, comm_manager):
        """Test heartbeat message sending."""
        result = await comm_manager.send_heartbeat("agent1")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_task_request_broadcast(self, comm_manager):
        """Test task request broadcasting."""
        result = await comm_manager.broadcast_task_request(
            sender="orchestrator",
            task_info={
                "type": "data_processing",
                "data": {"input": "test_data"}
            },
            required_capabilities=["data_processing", "analysis"],
            priority=MessagePriority.CRITICAL
        )
        
        assert result is True
    
    def test_communication_stats(self, comm_manager):
        """Test communication statistics."""
        stats = comm_manager.get_communication_stats()
        
        assert 'message_broker' in stats
        assert 'state_manager' in stats
        assert isinstance(stats['message_broker'], dict)
        assert isinstance(stats['state_manager'], dict)


class TestCommunicationIntegration:
    """Integration tests for communication system."""
    
    def test_singleton_pattern(self):
        """Test singleton pattern for communication manager."""
        with patch('utils.communication.StructuredLogger'), \
             patch('utils.communication.MessageBroker'), \
             patch('utils.communication.get_global_state_manager'):
            
            # Reset singleton
            import utils.communication
            utils.communication._global_comm_manager = None
            
            # Get instance twice
            manager1 = get_communication_manager()
            manager2 = get_communication_manager()
            
            # Should be same instance
            assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('utils.communication.StructuredLogger'), \
             patch('utils.communication.MessageBroker'), \
             patch('utils.communication.get_global_state_manager'), \
             patch('utils.communication.get_communication_manager'):
            
            # Reset singleton
            import utils.communication
            utils.communication._global_comm_manager = None
            
            # Test convenience functions
            result1 = await send_status_update("orchestrator", "agent1", "active")
            result2 = await send_error_report("agent1", {"error": "test"})
            result3 = await send_heartbeat("agent1")
            result4 = await broadcast_task_request("orchestrator", {"task": "test"})
            
            assert result1 is True
            assert result2 is True
            assert result3 is True
            assert result4 is True


if __name__ == "__main__":
    pytest.main([__file__])