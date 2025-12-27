"""
Base Agent Class

Core agent functionality that all specialized agents inherit from.
Provides common capabilities for logging, error handling, state access,
and configuration management.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from utils.logger import StructuredLogger
from utils.config import settings
from healing.error_handler import ErrorHandler


class AgentStatus(Enum):
    """Agent status enumeration."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    COMPLETED = "completed"
    DISABLED = "disabled"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    
    Provides common functionality including:
    - Structured logging
    - Error handling and recovery
    - State management integration
    - Configuration management
    - Task execution framework
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for the agent
            config: Agent configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
        # Core components
        self.logger = StructuredLogger(f"agent.{name}", settings.log_level)
        self.error_handler = ErrorHandler()
        
        # State management
        self._state: Optional[Dict[str, Any]] = None
        self._workflow_state: Optional[Any] = None
        
        # Task management
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
        self.task_queue: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'error_count': 0
        }
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize the agent
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent with configuration and setup."""
        try:
            self.logger.info(f"Initializing agent: {self.name}")
            
            # Load agent-specific configuration
            self._load_configuration()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Perform agent-specific initialization
            self._on_initialize()
            
            # Set status to idle after successful initialization
            self.status = AgentStatus.IDLE
            self.last_updated = datetime.utcnow()
            
            self.logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise
    
    def _load_configuration(self):
        """Load agent configuration from settings."""
        # Merge with default configuration
        default_config = {
            'max_retries': 3,
            'timeout': 300,
            'retry_delay': 1.0,
            'enable_metrics': True,
            'log_level': settings.log_level
        }
        
        # Update with provided config
        if self.config:
            default_config.update(self.config)
        
        self.config = default_config
    
    def _setup_event_handlers(self):
        """Setup default event handlers."""
        self._event_handlers = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'error_occurred': [],
            'status_changed': []
        }
    
    @abstractmethod
    def _on_initialize(self):
        """
        Agent-specific initialization logic.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a specific task.
        Must be implemented by subclasses.
        
        Args:
            task: Task dictionary with task details
            
        Returns:
            Task result dictionary
        """
        pass
    
    def set_state(self, state: Dict[str, Any]):
        """Set the global state reference."""
        self._state = state
        self.logger.debug(f"Agent {self.name} state reference set")
    
    def set_workflow_state(self, workflow_state: Any):
        """Set the workflow state reference."""
        self._workflow_state = workflow_state
        self.logger.debug(f"Agent {self.name} workflow state reference set")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the global state."""
        if self._state:
            return self._state.get(key, default)
        return default
    
    def update_state(self, key: str, value: Any):
        """Update a value in the global state."""
        if self._state:
            self._state[key] = value
            self.logger.debug(f"Agent {self.name} updated state: {key}")
    
    def get_workflow_state(self) -> Optional[Any]:
        """Get the workflow state."""
        return self._workflow_state
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with error handling and logging.
        
        Args:
            task: Task dictionary with task details
            
        Returns:
            Task result dictionary
        """
        if self.status == AgentStatus.DISABLED:
            raise RuntimeError(f"Agent {self.name} is disabled")
        
        # Set current task
        self.current_task = task
        self.status = AgentStatus.WORKING
        self.last_updated = datetime.utcnow()
        
        # Add task to history
        task_record = {
            'task_id': task.get('id', f"task_{len(self.task_history)}"),
            'task_type': task.get('type', 'unknown'),
            'started_at': datetime.utcnow().isoformat(),
            'status': 'started'
        }
        self.task_history.append(task_record)
        
        # Emit task started event
        await self._emit_event('task_started', task_record)
        
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Executing task {task_record['task_id']} for agent {self.name}")
            
            # Execute the task with retry logic
            result = await self._execute_with_retry(task)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(True, execution_time)
            
            # Update task record
            task_record.update({
                'completed_at': datetime.utcnow().isoformat(),
                'execution_time': execution_time,
                'status': 'completed',
                'result': result
            })
            
            # Emit task completed event
            await self._emit_event('task_completed', task_record)
            
            self.logger.info(f"Task {task_record['task_id']} completed successfully", 
                           execution_time=execution_time)
            
            # Clear current task and set status to idle
            self.current_task = None
            self.status = AgentStatus.IDLE
            self.last_updated = datetime.utcnow()
            
            return result
            
        except Exception as e:
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(False, execution_time)
            
            # Capture error context
            error_context = self.error_handler.capture_error(e, self.name)
            
            # Update task record
            task_record.update({
                'failed_at': datetime.utcnow().isoformat(),
                'execution_time': execution_time,
                'status': 'failed',
                'error': error_context
            })
            
            # Emit task failed event
            await self._emit_event('task_failed', task_record)
            
            self.logger.error(f"Task {task_record['task_id']} failed: {e}")
            
            # Set status to error
            self.status = AgentStatus.ERROR
            self.last_updated = datetime.utcnow()
            
            # Report error to workflow state if available
            if self._workflow_state and hasattr(self._workflow_state, 'add_error'):
                self._workflow_state.add_error(error_context)
            elif self._workflow_state and hasattr(self._workflow_state, 'state_manager'):
                self._workflow_state.state_manager.add_error(error_context)
            
            raise
    
    async def _execute_with_retry(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with retry logic."""
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1.0)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.warning(f"Retrying task {task.get('id')} (attempt {attempt})")
                    await asyncio.sleep(retry_delay * attempt)  # Exponential backoff
                
                return await self._process_task(task)
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(f"Task attempt {attempt} failed: {e}")
                else:
                    self.logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    def _update_metrics(self, success: bool, execution_time: float):
        """Update agent performance metrics."""
        if not self.config.get('enable_metrics', True):
            return
        
        if success:
            self.metrics['tasks_completed'] += 1
        else:
            self.metrics['tasks_failed'] += 1
        
        self.metrics['total_execution_time'] += execution_time
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        if total_tasks > 0:
            self.metrics['average_execution_time'] = self.metrics['total_execution_time'] / total_tasks
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for a specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        self.logger.debug(f"Added event handler for {event_type}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event_type}: {e}")
    
    def set_status(self, status: AgentStatus):
        """Set agent status and emit status change event."""
        old_status = self.status
        self.status = status
        self.last_updated = datetime.utcnow()
        
        # Emit status change event synchronously to avoid event loop issues
        event_data = {
            'old_status': old_status.value,
            'new_status': status.value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Try to emit async, fall back to sync if no event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(self._emit_event('status_changed', event_data))
            else:
                # Emit synchronously if no running loop
                for handler in self._event_handlers.get('status_changed', []):
                    if not asyncio.iscoroutinefunction(handler):
                        handler(event_data)
        except RuntimeError:
            # No event loop running, emit synchronously
            for handler in self._event_handlers.get('status_changed', []):
                if not asyncio.iscoroutinefunction(handler):
                    handler(event_data)
        
        self.logger.info(f"Agent {self.name} status changed: {old_status.value} -> {status.value}")
    
    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            **self.metrics,
            'current_status': self.status.value,
            'last_updated': self.last_updated.isoformat(),
            'created_at': self.created_at.isoformat(),
            'task_queue_length': len(self.task_queue),
            'current_task': self.current_task is not None
        }
    
    def add_task_to_queue(self, task: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM):
        """Add a task to the agent's queue."""
        task_with_priority = {
            **task,
            'priority': priority.value,
            'queued_at': datetime.utcnow().isoformat()
        }
        
        self.task_queue.append(task_with_priority)
        # Sort queue by priority (higher priority first)
        self.task_queue.sort(key=lambda t: t['priority'], reverse=True)
        
        self.logger.debug(f"Added task to queue: {task.get('id')} (priority: {priority.name})")
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task from the queue."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None
    
    def clear_task_queue(self):
        """Clear all tasks from the queue."""
        cleared_count = len(self.task_queue)
        self.task_queue.clear()
        self.logger.info(f"Cleared {cleared_count} tasks from queue")
    
    async def shutdown(self):
        """Gracefully shutdown the agent."""
        self.logger.info(f"Shutting down agent: {self.name}")
        
        # Set status to disabled
        self.set_status(AgentStatus.DISABLED)
        
        # Clear current task and queue
        self.current_task = None
        self.clear_task_queue()
        
        # Perform agent-specific cleanup
        await self._on_shutdown()
        
        self.logger.info(f"Agent {self.name} shutdown complete")
    
    async def _on_shutdown(self):
        """
        Agent-specific shutdown logic.
        Can be overridden by subclasses.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BaseAgent(name={self.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"BaseAgent(name={self.name}, status={self.status.value}, "
                f"tasks_completed={self.metrics['tasks_completed']}, "
                f"tasks_failed={self.metrics['tasks_failed']})")