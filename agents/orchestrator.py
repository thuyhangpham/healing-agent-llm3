"""
Orchestrator Agent

Central coordination agent that manages workflow execution,
agent registration, task scheduling, and system-wide monitoring.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum

from agents.base_agent import BaseAgent, AgentStatus, TaskPriority
from utils.workflow_engine import LangGraphWorkflowEngine, WorkflowState
from utils.logger import StructuredLogger
from utils.config import settings


class OrchestratorStatus(Enum):
    """Orchestrator-specific status values."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class WorkflowPhase(Enum):
    """Workflow execution phases."""
    INITIALIZATION = "initialization"
    AGENT_DISCOVERY = "agent_discovery"
    TASK_DISTRIBUTION = "task_distribution"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator agent for managing multi-agent workflows.
    
    Responsibilities:
    - Agent registration and discovery
    - Task scheduling and distribution
    - Workflow coordination using LangGraph
    - System monitoring and health checks
    - Error handling and recovery coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator agent."""
        super().__init__("orchestrator", config)
        
        # Orchestrator-specific state
        self.orchestrator_status = OrchestratorStatus.STARTING
        self.current_phase = WorkflowPhase.INITIALIZATION
        
        # Agent registry
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        
        # Workflow management
        self.workflow_engine: Optional[LangGraphWorkflowEngine] = None
        self.active_workflows: List[str] = []
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Task management
        self.task_queue: List[Dict[str, Any]] = []
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_name
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Monitoring
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
        self.task_timeout = self.config.get('task_timeout', 300)
        
        # Event subscriptions
        self._setup_agent_event_handlers()
        
        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info("Orchestrator agent initialized")
    
    def _on_initialize(self):
        """Orchestrator-specific initialization."""
        try:
            # Initialize workflow engine
            self.workflow_engine = LangGraphWorkflowEngine()
            
            # Set orchestrator status
            self.orchestrator_status = OrchestratorStatus.RUNNING
            self.current_phase = WorkflowPhase.AGENT_DISCOVERY
            
            self.logger.info("Orchestrator initialization complete")
            
            # Override base agent status to keep orchestrator status
            self.status = AgentStatus.IDLE  # Keep base agent status as IDLE
            
            # Debug logging
            self.logger.info(f"Orchestrator status set to: {self.orchestrator_status.value}")
            print(f"DEBUG: Inside _on_initialize, status is now: {self.orchestrator_status}")
        except Exception as e:
            self.logger.error(f"Error in orchestrator _on_initialize: {e}")
            raise
    
    def _setup_agent_event_handlers(self):
        """Setup event handlers for agent lifecycle events."""
        self.add_event_handler('task_completed', self._on_task_completed)
        self.add_event_handler('task_failed', self._on_task_failed)
        self.add_event_handler('error_occurred', self._on_agent_error)
        self.add_event_handler('register_healing_agent', self._on_healing_agent_register)
    
    async def _on_healing_agent_register(self, event_data: Dict[str, Any]):
        """Handle healing agent registration event."""
        healing_agent = event_data.get('agent')
        if healing_agent:
            await self.register_healing_agent(healing_agent)
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process orchestrator-specific tasks.
        
        Orchestrator tasks include:
        - register_agent: Register a new agent
        - distribute_task: Distribute a task to suitable agents
        - monitor_agents: Check agent health
        - execute_workflow: Execute a complete workflow
        """
        task_type = task.get('type', 'unknown')
        
        try:
            if task_type == 'register_agent':
                return await self._handle_agent_registration(task)
            elif task_type == 'distribute_task':
                return await self._handle_task_distribution(task)
            elif task_type == 'monitor_agents':
                return await self._handle_agent_monitoring(task)
            elif task_type == 'execute_workflow':
                return await self._handle_workflow_execution(task)
            elif task_type == 'health_check':
                return await self._handle_health_check(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to process orchestrator task {task_type}: {e}")
            raise
    
    async def _handle_agent_registration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration task."""
        agent_name = task.get('agent_name')
        agent = task.get('agent')
        capabilities = task.get('capabilities', set())
        
        if not agent_name or not agent:
            raise ValueError("Agent registration requires agent_name and agent")
        
        # Register the agent
        self.registered_agents[agent_name] = agent
        self.agent_capabilities[agent_name] = set(capabilities)
        self.agent_health[agent_name] = {
            'status': 'healthy',
            'last_check': datetime.utcnow().isoformat(),
            'tasks_completed': 0,
            'tasks_failed': 0,
            'last_activity': datetime.utcnow().isoformat()
        }
        
        # Set agent state references
        if hasattr(self, '_workflow_state') and self._workflow_state:
            agent.set_workflow_state(self._workflow_state)
        
        self.logger.info(f"Registered agent: {agent_name} with capabilities: {capabilities}")
        
        return {
            'status': 'registered',
            'agent_name': agent_name,
            'capabilities': list(capabilities),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_task_distribution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task distribution to suitable agents."""
        task_data = task.get('task_data', {})
        required_capabilities = task.get('required_capabilities', [])
        priority = task.get('priority', TaskPriority.MEDIUM)
        
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(required_capabilities)
        
        if not suitable_agents:
            raise ValueError(f"No agents available for capabilities: {required_capabilities}")
        
        # Select best agent (simple round-robin for now)
        selected_agent = self._select_best_agent(suitable_agents, task_data)
        
        # Assign task to agent
        task_id = task_data.get('id', f"task_{len(self.task_assignments)}")
        self.task_assignments[task_id] = selected_agent
        
        # Add task to agent's queue
        agent = self.registered_agents.get(selected_agent)
        if agent:
            agent.add_task_to_queue(task_data, priority)
        
        self.logger.info(f"Distributed task {task_id} to agent {selected_agent}")
        
        return {
            'status': 'distributed',
            'task_id': task_id,
            'assigned_agent': selected_agent,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_agent_monitoring(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent health monitoring."""
        health_results = {}
        
        for agent_name, agent in self.registered_agents.items():
            try:
                # Get agent metrics
                metrics = agent.get_metrics()
                
                # Update health status
                health_status = self._assess_agent_health(agent_name, metrics)
                if agent_name in self.agent_health:
                    self.agent_health[agent_name].update({
                        'status': health_status,
                        'last_check': datetime.utcnow().isoformat(),
                        'metrics': metrics
                    })
                
                health_results[agent_name] = health_status
                
            except Exception as e:
                self.logger.error(f"Failed to monitor agent {agent_name}: {e}")
                health_results[agent_name] = 'error'
        
        return {
            'status': 'completed',
            'health_results': health_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_workflow_execution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complete workflow execution."""
        workflow_definition = task.get('workflow_definition', {})
        workflow_id = task.get('workflow_id', f"workflow_{len(self.active_workflows)}")
        
        try:
            # Initialize workflow state manager
            from utils.workflow_engine import WorkflowStateManager
            workflow_state_manager = WorkflowStateManager()
            
            # Register agents with workflow
            for agent_name, agent in self.registered_agents.items():
                if hasattr(workflow_state, 'register_agent'):
                    workflow_state.register_agent(agent_name, {
                        'type': 'agent',
                        'status': agent.get_status().value,
                        'capabilities': list(self.agent_capabilities.get(agent_name, []))
                    })
                elif hasattr(workflow_state, 'state_manager'):
                    workflow_state.state_manager.register_agent(agent_name, {
                        'type': 'agent',
                        'status': agent.get_status().value,
                        'capabilities': list(self.agent_capabilities.get(agent_name, []))
                    })
            
            # Execute workflow using LangGraph
            if self.workflow_engine:
                result = await self.workflow_engine.run_workflow(workflow_state)
                
                # Record workflow execution
                start_time = workflow_state_manager.state.get('start_time', datetime.utcnow().isoformat())
                workflow_record = {
                    'workflow_id': workflow_id,
                    'status': 'completed',
                    'started_at': start_time,
                    'completed_at': datetime.utcnow().isoformat(),
                    'result': workflow_record
                }
                
                self.workflow_history.append(workflow_record)
                
                return {
                    'status': 'completed',
                    'workflow_id': workflow_id,
                    'result': workflow_record,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                raise RuntimeError("Workflow engine not initialized")
                
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                'status': 'failed',
                'workflow_id': workflow_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _handle_health_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system health check."""
        system_health = {
            'orchestrator_status': self.orchestrator_status.value,
            'current_phase': self.current_phase.value,
            'registered_agents': len(self.registered_agents),
            'active_workflows': len(self.active_workflows),
            'task_queue_length': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'system_uptime': (datetime.utcnow() - self.created_at).total_seconds()
        }
        
        return {
            'status': 'healthy',
            'system_health': system_health,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _find_suitable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents that have the required capabilities."""
        suitable_agents = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            agent = self.registered_agents.get(agent_name)
            
            # Check if agent is active and has required capabilities
            if (agent and 
                agent.get_status() == AgentStatus.IDLE and
                all(cap in capabilities for cap in required_capabilities)):
                suitable_agents.append(agent_name)
        
        return suitable_agents
    
    def _select_best_agent(self, suitable_agents: List[str], task_data: Dict[str, Any]) -> str:
        """Select the best agent for a task (simple round-robin)."""
        # For now, use simple round-robin
        # In future, could consider agent load, capabilities, past performance, etc.
        if not suitable_agents:
            raise ValueError("No suitable agents available")
        
        # Sort by current task load
        agent_loads = []
        for agent_name in suitable_agents:
            agent = self.registered_agents[agent_name]
            load = len(agent.task_queue)
            agent_loads.append((load, agent_name))
        
        # Select agent with lowest load
        agent_loads.sort()
        return agent_loads[0][1]
    
    def _assess_agent_health(self, agent_name: str, metrics: Dict[str, Any]) -> str:
        """Assess agent health based on metrics."""
        try:
            # Check agent status
            agent = self.registered_agents.get(agent_name)
            if not agent:
                return 'missing'
            
            if agent.get_status() == AgentStatus.ERROR:
                return 'unhealthy'
            
            if agent.get_status() == AgentStatus.DISABLED:
                return 'disabled'
            
            # Check error rate
            total_tasks = metrics.get('tasks_completed', 0) + metrics.get('tasks_failed', 0)
            if total_tasks > 0:
                error_rate = metrics.get('tasks_failed', 0) / total_tasks
                if error_rate >= 0.5:  # 50% or more failure rate
                    return 'unhealthy'
            
            # Check last activity
            last_updated = metrics.get('last_updated')
            if last_updated:
                last_activity = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                time_since_activity = datetime.utcnow() - last_activity.replace(tzinfo=None)
                if time_since_activity > timedelta(minutes=5):
                    return 'stale'
            
            return 'healthy'
            
        except Exception as e:
            self.logger.error(f"Error assessing agent health for {agent_name}: {e}")
            return 'error'
    
    async def _on_task_completed(self, event_data: Dict[str, Any]):
        """Handle task completion event."""
        task_id = event_data.get('task_id')
        if task_id in self.task_assignments:
            agent_name = self.task_assignments.pop(task_id)
            
            # Update agent health
            if agent_name in self.agent_health:
                self.agent_health[agent_name]['tasks_completed'] += 1
                self.agent_health[agent_name]['last_activity'] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Task {task_id} completed by agent {agent_name}")
    
    async def _on_task_failed(self, event_data: Dict[str, Any]):
        """Handle task failure event."""
        task_id = event_data.get('task_id')
        if task_id in self.task_assignments:
            agent_name = self.task_assignments.pop(task_id)
            
            # Update agent health
            if agent_name in self.agent_health:
                self.agent_health[agent_name]['tasks_failed'] += 1
                self.agent_health[agent_name]['last_activity'] = datetime.utcnow().isoformat()
            
            self.logger.warning(f"Task {task_id} failed by agent {agent_name}")
    
    async def _on_agent_error(self, event_data: Dict[str, Any]):
        """Handle agent error event."""
        agent_name = event_data.get('agent_name')
        error_context = event_data.get('error_context')
        
        if agent_name in self.agent_health:
            self.agent_health[agent_name]['status'] = 'unhealthy'
            self.agent_health[agent_name]['last_error'] = error_context
        
        self.logger.error(f"Agent {agent_name} reported error: {error_context}")
        
        # Forward to healing agent if available
        await self._forward_to_healing_agent(agent_name, error_context)
    
    async def _forward_to_healing_agent(self, agent_name: str, error_context: Dict[str, Any]):
        """Forward error to healing agent for automated repair."""
        try:
            # Check if healing agent is registered
            healing_agent = self.registered_agents.get('healing_agent')
            if not healing_agent:
                self.logger.warning("Healing agent not available for error handling")
                return
            
            # Check if auto-healing is enabled
            healing_config = self.config.get('healing_integration', {})
            if not healing_config.get('auto_error_routing', True):
                self.logger.info("Auto error routing to healing agent is disabled")
                return
            
            # Prepare error context for healing
            healing_error_context = {
                'agent_name': agent_name,
                'error_type': error_context.get('error_type', 'Unknown'),
                'error_message': error_context.get('error_message', ''),
                'traceback': error_context.get('traceback', ''),
                'file_path': error_context.get('file_path', ''),
                'function_name': error_context.get('function_name', ''),
                'line_number': error_context.get('line_number', 0),
                'timestamp': error_context.get('timestamp', datetime.now().isoformat()),
                'additional_context': error_context.get('additional_context', {}),
                'severity': error_context.get('severity', 'medium')
            }
            
            # Send to healing agent
            healing_message = {
                'type': 'error_report',
                'error_context': healing_error_context,
                'source_agent': agent_name,
                'orchestrator_request': True
            }
            
            response = await healing_agent.process_message(healing_message)
            
            if response.get('type') == 'healing_result':
                result = response.get('result', {})
                if result.get('success', False):
                    self.logger.info(f"Healing agent successfully handled error from {agent_name}")
                    
                    # Update agent health status
                    if agent_name in self.agent_health:
                        self.agent_health[agent_name]['status'] = 'healed'
                        self.agent_health[agent_name]['last_healed'] = datetime.now().isoformat()
                else:
                    self.logger.warning(f"Healing agent failed to handle error from {agent_name}: {result.get('error_message', 'Unknown')}")
            else:
                self.logger.error(f"Unexpected response from healing agent: {response}")
                
        except Exception as e:
            self.logger.error(f"Failed to forward error to healing agent: {e}")
    
    async def register_healing_agent(self, healing_agent):
        """Register healing agent and set up error handling integration."""
        try:
            self.registered_agents['healing_agent'] = healing_agent
            
            # Set up healing agent capabilities
            self.agent_capabilities['healing_agent'] = {
                'error_detection',
                'code_analysis',
                'automated_healing',
                'hot_reload',
                'metrics_collection',
                'pattern_learning'
            }
            
            # Initialize healing agent health
            self.agent_health['healing_agent'] = {
                'status': 'healthy',
                'last_activity': datetime.now().isoformat(),
                'tasks_completed': 0,
                'tasks_failed': 0,
                'healing_operations': 0
            }
            
            self.logger.info("Healing agent registered and integrated with orchestrator")
            
        except Exception as e:
            self.logger.error(f"Failed to register healing agent: {e}")
    
    async def get_healing_status(self) -> Dict[str, Any]:
        """Get current healing system status."""
        try:
            healing_agent = self.registered_agents.get('healing_agent')
            if not healing_agent:
                return {'error': 'Healing agent not registered'}
            
            # Get healing agent status
            status_message = {'type': 'get_status'}
            response = await healing_agent.process_message(status_message)
            
            if response.get('type') == 'status_response':
                healing_status = response.get('status', {})
                
                # Add orchestrator integration info
                healing_status['orchestrator_integration'] = {
                    'registered': True,
                    'auto_error_routing': self.config.get('healing_integration', {}).get('auto_error_routing', True),
                    'healing_targets': self.config.get('healing_integration', {}).get('healing_targets', []),
                    'total_agents': len(self.registered_agents) - 1  # Exclude healing agent
                }
                
                return healing_status
            else:
                return {'error': 'Failed to get healing agent status'}
                
        except Exception as e:
            self.logger.error(f"Failed to get healing status: {e}")
            return {'error': str(e)}
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        # Start health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.append(health_task)
        
        # Start task distribution task
        distribution_task = asyncio.create_task(self._task_distribution_loop())
        self._background_tasks.append(distribution_task)
        
        self.logger.info("Started orchestrator monitoring tasks")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
        self.logger.info("Stopped orchestrator monitoring tasks")
    
    async def _health_monitoring_loop(self):
        """Background loop for agent health monitoring."""
        while self.orchestrator_status in [OrchestratorStatus.RUNNING, OrchestratorStatus.PAUSED]:
            try:
                # Perform health check
                await self.execute({
                    'type': 'monitor_agents',
                    'id': f'health_check_{datetime.utcnow().timestamp()}'
                })
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _task_distribution_loop(self):
        """Background loop for task distribution."""
        while self.orchestrator_status == OrchestratorStatus.RUNNING:
            try:
                # Check for tasks in queue
                if self.task_queue and len(self.task_assignments) < self.max_concurrent_tasks:
                    task = self.task_queue.pop(0)
                    
                    try:
                        await self.execute({
                            'type': 'distribute_task',
                            'task_data': task,
                            'id': f'distribute_{datetime.utcnow().timestamp()}'
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to distribute task: {e}")
                        # Re-queue task if distribution failed
                        self.task_queue.insert(0, task)
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Task distribution error: {e}")
                await asyncio.sleep(5)
    
    def register_agent(self, agent: BaseAgent, capabilities: List[str] = None):
        """Register an agent with the orchestrator."""
        capabilities = capabilities or []
        
        # Create registration task
        registration_task = {
            'type': 'register_agent',
            'agent_name': agent.name,
            'agent': agent,
            'capabilities': capabilities,
            'id': f'register_{agent.name}_{datetime.utcnow().timestamp()}'
        }
        
        # Execute registration synchronously for immediate effect
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(self.execute(registration_task))
            else:
                # Run synchronously if no event loop
                asyncio.run(self.execute(registration_task))
        except RuntimeError:
            # No event loop, run synchronously
            import asyncio as _asyncio
            _asyncio.run(self.execute(registration_task))
    
    def submit_task(self, task: Dict[str, Any], required_capabilities: List[str] = None):
        """Submit a task for distribution to suitable agents."""
        required_capabilities = required_capabilities or []
        
        # Add task to queue
        task_with_metadata = {
            **task,
            'required_capabilities': required_capabilities or [],
            'submitted_at': datetime.utcnow().isoformat()
        }
        
        self.task_queue.append(task_with_metadata)
        self.logger.info(f"Submitted task {task.get('id')} to queue")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'orchestrator': {
                'status': self.orchestrator_status.value,
                'phase': self.current_phase.value,
                'uptime': (datetime.utcnow() - self.created_at).total_seconds(),
                'metrics': self.get_metrics()
            },
            'agents': {
                'registered': len(self.registered_agents),
                'healthy': sum(1 for h in self.agent_health.values() if h['status'] == 'healthy'),
                'unhealthy': sum(1 for h in self.agent_health.values() if h['status'] == 'unhealthy'),
                'details': self.agent_health
            },
            'workflows': {
                'active': len(self.active_workflows),
                'completed': len(self.workflow_history),
                'history': self.workflow_history[-5:]  # Last 5 workflows
            },
            'tasks': {
                'queued': len(self.task_queue),
                'assigned': len(self.task_assignments),
                'completed': len(self.completed_tasks),
                'assignments': self.task_assignments
            }
        }
    
    async def pause(self):
        """Pause orchestrator operations."""
        self.orchestrator_status = OrchestratorStatus.PAUSED
        self.logger.info("Orchestrator paused")
    
    async def resume(self):
        """Resume orchestrator operations."""
        self.orchestrator_status = OrchestratorStatus.RUNNING
        self.logger.info("Orchestrator resumed")
    
    async def _on_shutdown(self):
        """Orchestrator-specific shutdown logic."""
        self.orchestrator_status = OrchestratorStatus.STOPPING
        
        # Stop monitoring tasks
        await self.stop_monitoring()
        
        # Shutdown all registered agents
        for agent_name, agent in self.registered_agents.items():
            try:
                await agent.shutdown()
                self.logger.info(f"Shutdown agent: {agent_name}")
            except Exception as e:
                self.logger.error(f"Failed to shutdown agent {agent_name}: {e}")
        
        self.orchestrator_status = OrchestratorStatus.STOPPED
        self.logger.info("Orchestrator shutdown complete")