"""
Inter-Agent Communication System

Provides communication patterns and protocols for agents
to exchange messages, coordinate tasks, and share information.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import threading

from agents.base_agent import BaseAgent, AgentStatus
from utils.logger import StructuredLogger
from utils.config import settings
from utils.global_state import get_global_state_manager, StateChangeType


class MessageType(Enum):
    """Types of inter-agent messages."""
    DIRECT_MESSAGE = "direct_message"
    BROADCAST = "broadcast"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    ERROR_REPORT = "error_report"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""
    id: str
    message_type: MessageType
    sender: str
    recipient: Optional[str]  # None for broadcast
    timestamp: datetime
    priority: MessagePriority
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None  # For request-response correlation
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


class CommunicationProtocol:
    """Defines communication protocols and message handling."""
    
    def __init__(self):
        self.logger = StructuredLogger("communication_protocol", settings.log_level)
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_filters: List[Callable[[AgentMessage], bool]] = []
    
    def register_handler(self, message_type: MessageType, handler: Callable[[AgentMessage], Any]):
        """Register a handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        self.logger.debug(f"Registered handler for message type: {message_type.value}")
    
    def unregister_handler(self, message_type: MessageType, handler: Callable[[AgentMessage], Any]):
        """Unregister a message handler."""
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
                self.logger.debug(f"Unregistered handler for message type: {message_type.value}")
            except ValueError:
                pass  # Handler not found
    
    def add_message_filter(self, filter_func: Callable[[AgentMessage], bool]):
        """Add a message filter."""
        self.message_filters.append(filter_func)
        self.logger.debug("Added message filter")
    
    def remove_message_filter(self, filter_func: Callable[[AgentMessage], bool]):
        """Remove a message filter."""
        try:
            self.message_filters.remove(filter_func)
            self.logger.debug("Removed message filter")
        except ValueError:
            pass  # Filter not found
    
    def should_process_message(self, message: AgentMessage) -> bool:
        """Check if message should be processed based on filters."""
        for filter_func in self.message_filters:
            try:
                if not filter_func(message):
                    return False
            except Exception as e:
                self.logger.error(f"Message filter error: {e}")
                return False
        return True
    
    async def handle_message(self, message: AgentMessage) -> Optional[Any]:
        """Handle an incoming message."""
        # Check if message should be processed
        if not self.should_process_message(message):
            return None
        
        # Check if message is expired
        if message.is_expired():
            self.logger.warning(f"Discarding expired message: {message.id}")
            return None
        
        # Get handlers for message type
        handlers = self.message_handlers.get(message.message_type.value, [])
        
        # Call all handlers
        results = []
        for handler in handlers:
            try:
                result = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Message handler error: {e}")
        
        return results if len(results) > 1 else (results[0] if results else None)


class MessageBroker:
    """
    Central message broker for inter-agent communication.
    
    Features:
    - Message routing and delivery
    - Message queuing and priority handling
    - Broadcast and direct messaging
    - Message persistence and history
    - Performance monitoring
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.logger = StructuredLogger("message_broker", settings.log_level)
        
        # Message storage
        self._message_queue: List[AgentMessage] = []
        self._delivered_messages: Dict[str, AgentMessage] = {}
        self._max_queue_size = max_queue_size
        
        # Agent subscriptions
        self._subscriptions: Dict[str, List[str]] = {}  # agent_name -> [message_types]
        self._agent_addresses: Dict[str, str] = {}  # agent_name -> address
        
        # Protocol handler
        self.protocol = CommunicationProtocol()
        
        # Performance metrics
        self._metrics = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_expired': 0,
            'queue_size': 0,
            'active_subscriptions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Message broker initialized")
    
    def register_agent(self, agent_name: str, address: str = None):
        """Register an agent with the message broker."""
        with self._lock:
            self._agent_addresses[agent_name] = address or f"agent://{agent_name}"
            self.logger.info(f"Registered agent {agent_name} at {self._agent_addresses[agent_name]}")
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the message broker."""
        with self._lock:
            if agent_name in self._agent_addresses:
                del self._agent_addresses[agent_name]
                self.logger.info(f"Unregistered agent {agent_name}")
    
    def subscribe(self, agent_name: str, message_types: List[MessageType]):
        """Subscribe an agent to specific message types."""
        with self._lock:
            if agent_name not in self._subscriptions:
                self._subscriptions[agent_name] = []
            
            for msg_type in message_types:
                if msg_type.value not in self._subscriptions[agent_name]:
                    self._subscriptions[agent_name].append(msg_type.value)
            
            self._metrics['active_subscriptions'] = len(self._subscriptions.get(agent_name, []))
            
        self.logger.info(f"Agent {agent_name} subscribed to: {[mt.value for mt in message_types]}")
    
    def unsubscribe(self, agent_name: str, message_types: List[MessageType]):
        """Unsubscribe an agent from specific message types."""
        with self._lock:
            if agent_name in self._subscriptions:
                for msg_type in message_types:
                    if msg_type.value in self._subscriptions[agent_name]:
                        self._subscriptions[agent_name].remove(msg_type.value)
            
            self._metrics['active_subscriptions'] = len(self._subscriptions.get(agent_name, []))
            
        self.logger.info(f"Agent {agent_name} unsubscribed from: {[mt.value for mt in message_types]}")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the broker."""
        try:
            with self._lock:
                # Add to queue
                self._message_queue.append(message)
                
                # Sort queue by priority (higher first)
                self._message_queue.sort(key=lambda m: m.priority.value, reverse=True)
                
                # Trim queue if needed
                if len(self._message_queue) > self._max_queue_size:
                    self._message_queue = self._message_queue[-self._max_queue_size:]
                
                self._metrics['messages_sent'] += 1
                self._metrics['queue_size'] = len(self._message_queue)
            
            self.logger.debug(f"Queued message {message.id} from {message.sender}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue message: {e}")
            return False
    
    async def broadcast_message(self, sender: str, message_type: MessageType, 
                           payload: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM,
                           expires_at: Optional[datetime] = None) -> bool:
        """Send a broadcast message to all subscribed agents."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            message_type=message_type,
            sender=sender,
            recipient=None,  # Broadcast
            timestamp=datetime.utcnow(),
            priority=priority,
            payload=payload,
            expires_at=expires_at
        )
        
        return await self.send_message(message)
    
    async def send_direct_message(self, sender: str, recipient: str, message_type: MessageType,
                              payload: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM,
                              correlation_id: Optional[str] = None,
                              expires_at: Optional[datetime] = None) -> bool:
        """Send a direct message to a specific agent."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            message_type=message_type,
            sender=sender,
            recipient=recipient,
            timestamp=datetime.utcnow(),
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            expires_at=expires_at
        )
        
        return await self.send_message(message)
    
    async def process_message_queue(self):
        """Process the message queue and deliver messages."""
        while self._message_queue:
            with self._lock:
                if not self._message_queue:
                    break
                
                message = self._message_queue.pop(0)
                self._metrics['queue_size'] = len(self._message_queue)
            
            # Check if message is expired
            if message.is_expired():
                self._metrics['messages_expired'] += 1
                self.logger.warning(f"Discarding expired message: {message.id}")
                continue
            
            # Route message
            delivered = await self._route_message(message)
            
            if delivered:
                self._delivered_messages[message.id] = message
                self._metrics['messages_delivered'] += 1
                self.logger.debug(f"Delivered message {message.id}")
            else:
                self.logger.warning(f"Failed to deliver message: {message.id}")
    
    async def _route_message(self, message: AgentMessage) -> bool:
        """Route a message to its recipient(s)."""
        try:
            if message.recipient is None:
                # Broadcast message
                return await self._deliver_broadcast(message)
            else:
                # Direct message
                return await self._deliver_direct(message)
        except Exception as e:
            self.logger.error(f"Message routing error: {e}")
            return False
    
    async def _deliver_broadcast(self, message: AgentMessage) -> bool:
        """Deliver a broadcast message to all subscribed agents."""
        delivered_count = 0
        
        with self._lock:
            subscribers = []
            for agent_name, subscriptions in self._subscriptions.items():
                if message.message_type.value in subscriptions:
                    subscribers.append(agent_name)
            
            self.logger.debug(f"Broadcasting to {len(subscribers)} subscribers")
        
        # Deliver to all subscribers
        for agent_name in subscribers:
            if await self._deliver_to_agent(agent_name, message):
                delivered_count += 1
        
        return delivered_count > 0
    
    async def _deliver_direct(self, message: AgentMessage) -> bool:
        """Deliver a direct message to a specific agent."""
        if message.recipient not in self._agent_addresses:
            self.logger.warning(f"Unknown recipient: {message.recipient}")
            return False
        
        return await self._deliver_to_agent(message.recipient, message)
    
    async def _deliver_to_agent(self, agent_name: str, message: AgentMessage) -> bool:
        """Deliver a message to a specific agent."""
        try:
            # Check if agent is subscribed to this message type
            subscriptions = self._subscriptions.get(agent_name, [])
            if message.message_type.value not in subscriptions:
                self.logger.debug(f"Agent {agent_name} not subscribed to {message.message_type.value}")
                return False
            
            # Process message through protocol
            result = await self.protocol.handle_message(message)
            
            # Store in global state
            state_manager = get_global_state_manager()
            state_manager.set_state(f"message.{message.id}", {
                'delivered_to': agent_name,
                'delivered_at': datetime.utcnow().isoformat(),
                'message_type': message.message_type.value,
                'sender': message.sender
            }, agent_name="message_broker")
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Failed to deliver message to {agent_name}: {e}")
            return False
    
    def get_message_history(self, agent_name: Optional[str] = None, 
                         message_type: Optional[MessageType] = None,
                         limit: int = 100) -> List[AgentMessage]:
        """Get message history with optional filtering."""
        with self._lock:
            messages = list(self._delivered_messages.values())
            
            # Filter by agent
            if agent_name:
                messages = [m for m in messages if m.recipient == agent_name or 
                           (m.recipient is None and agent_name in self._subscriptions.get(m.sender, []))]
            
            # Filter by message type
            if message_type:
                messages = [m for m in messages if m.message_type == message_type]
            
            # Limit results
            if limit:
                messages = messages[-limit:]
            
            return messages
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics."""
        with self._lock:
            return {
                **self._metrics,
                'registered_agents': len(self._agent_addresses),
                'active_subscriptions': sum(len(subs) for subs in self._subscriptions.values()),
                'oldest_message_age': (datetime.utcnow() - self._message_queue[0].timestamp).total_seconds() if self._message_queue else 0,
                'newest_message_age': (datetime.utcnow() - self._message_queue[-1].timestamp).total_seconds() if self._message_queue else 0
            }
    
    def clear_queue(self):
        """Clear all messages from the queue."""
        with self._lock:
            cleared_count = len(self._message_queue)
            self._message_queue.clear()
            self._metrics['queue_size'] = 0
            
        self.logger.info(f"Cleared {cleared_count} messages from queue")
    
    def clear_expired_messages(self):
        """Remove expired messages from queue and history."""
        current_time = datetime.utcnow()
        expired_count = 0
        
        with self._lock:
            # Remove expired from queue
            self._message_queue = [m for m in self._message_queue if not m.is_expired()]
            
            # Remove expired from history
            expired_messages = [msg_id for msg_id, msg in self._delivered_messages.items() if msg.is_expired()]
            for msg_id in expired_messages:
                del self._delivered_messages[msg_id]
                expired_count += 1
            
            self._metrics['messages_expired'] += expired_count
            
        if expired_count > 0:
            self.logger.info(f"Removed {expired_count} expired messages")


class CommunicationManager:
    """
    High-level communication manager that coordinates
    message brokers and provides unified communication interface.
    """
    
    def __init__(self):
        self.logger = StructuredLogger("communication_manager", settings.log_level)
        self.message_broker = MessageBroker()
        self.state_manager = get_global_state_manager()
        
        # Communication patterns
        self._setup_default_handlers()
        
        self.logger.info("Communication manager initialized")
    
    def _setup_default_handlers(self):
        """Setup default message handlers."""
        # Status update handler
        async def handle_status_update(message: AgentMessage):
            agent_name = message.payload.get('agent_name')
            status = message.payload.get('status')
            
            if agent_name:
                self.state_manager.set_state(f"agent.{agent_name}.status", status, message.sender)
                self.logger.info(f"Status update: {agent_name} -> {status}")
        
        # Error report handler
        async def handle_error_report(message: AgentMessage):
            error_info = message.payload
            self.state_manager.set_state(f"error.{message.id}", error_info, message.sender)
            self.logger.error(f"Error report from {message.sender}: {error_info}")
        
        # Heartbeat handler
        async def handle_heartbeat(message: AgentMessage):
            agent_name = message.sender
            self.state_manager.set_state(f"agent.{agent_name}.last_heartbeat", 
                                    datetime.utcnow().isoformat(), 
                                    message.sender)
        
        # Register handlers
        self.message_broker.protocol.register_handler(MessageType.STATUS_UPDATE, handle_status_update)
        self.message_broker.protocol.register_handler(MessageType.ERROR_REPORT, handle_error_report)
        self.message_broker.protocol.register_handler(MessageType.HEARTBEAT, handle_heartbeat)
    
    async def start_message_processing(self):
        """Start the background message processing loop."""
        self.logger.info("Started message processing")
        
        while True:
            try:
                await self.message_broker.process_message_queue()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    def register_agent(self, agent_name: str, address: str = None):
        """Register an agent for communication."""
        self.message_broker.register_agent(agent_name, address)
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from communication."""
        self.message_broker.unregister_agent(agent_name)
    
    def subscribe_to_messages(self, agent_name: str, message_types: List[MessageType]):
        """Subscribe an agent to message types."""
        self.message_broker.subscribe(agent_name, message_types)
    
    async def send_status_update(self, sender: str, agent_name: str, status: str, 
                           metadata: Dict[str, Any] = None) -> bool:
        """Send a status update message."""
        return await self.message_broker.send_direct_message(
            sender=sender,
            recipient=agent_name,
            message_type=MessageType.STATUS_UPDATE,
            payload={
                'agent_name': agent_name,
                'status': status,
                'metadata': metadata or {}
            }
        )
    
    async def send_error_report(self, sender: str, error_info: Dict[str, Any], 
                           correlation_id: Optional[str] = None) -> bool:
        """Send an error report message."""
        return await self.message_broker.send_direct_message(
            sender=sender,
            recipient="orchestrator",  # Error reports go to orchestrator
            message_type=MessageType.ERROR_REPORT,
            payload=error_info,
            correlation_id=correlation_id
        )
    
    async def send_heartbeat(self, agent_name: str) -> bool:
        """Send a heartbeat message."""
        return await self.message_broker.send_direct_message(
            sender=agent_name,
            recipient="orchestrator",  # Heartbeats go to orchestrator
            message_type=MessageType.HEARTBEAT,
            payload={
                'agent_name': agent_name,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def broadcast_task_request(self, sender: str, task_info: Dict[str, Any], 
                                required_capabilities: List[str] = None,
                                priority: MessagePriority = MessagePriority.HIGH) -> bool:
        """Broadcast a task request to capable agents."""
        return await self.message_broker.broadcast_message(
            sender=sender,
            message_type=MessageType.TASK_REQUEST,
            payload={
                'task_info': task_info,
                'required_capabilities': required_capabilities or [],
                'request_id': str(uuid.uuid4())
            },
            priority=priority
        )
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics."""
        broker_stats = self.message_broker.get_queue_status()
        
        return {
            'message_broker': broker_stats,
            'state_manager': self.state_manager.get_metrics()
        }


# Global communication manager instance
_global_comm_manager: Optional[CommunicationManager] = None


def get_communication_manager() -> CommunicationManager:
    """Get the singleton communication manager instance."""
    global _global_comm_manager
    if _global_comm_manager is None:
        _global_comm_manager = CommunicationManager()
    return _global_comm_manager


# Convenience functions
async def send_status_update(sender: str, agent_name: str, status: str, 
                        metadata: Dict[str, Any] = None) -> bool:
    """Send a status update using the global communication manager."""
    return await get_communication_manager().send_status_update(sender, agent_name, status, metadata)


async def send_error_report(sender: str, error_info: Dict[str, Any], 
                       correlation_id: Optional[str] = None) -> bool:
    """Send an error report using the global communication manager."""
    return await get_communication_manager().send_error_report(sender, error_info, correlation_id)


async def send_heartbeat(agent_name: str) -> bool:
    """Send a heartbeat using the global communication manager."""
    return await get_communication_manager().send_heartbeat(agent_name)


async def broadcast_task_request(sender: str, task_info: Dict[str, Any], 
                           required_capabilities: List[str] = None,
                           priority: MessagePriority = MessagePriority.HIGH) -> bool:
    """Broadcast a task request using the global communication manager."""
    return await get_communication_manager().broadcast_task_request(sender, task_info, required_capabilities, priority)