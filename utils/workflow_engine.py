"""
LangGraph Workflow Engine

Core workflow orchestration system using LangGraph for
multi-agent coordination and state management.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
except ImportError:
    # Fallback for older LangGraph versions
    try:
        from langgraph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from typing_extensions import TypedDict
    except ImportError:
        raise ImportError("LangGraph is required. Install with: pip install langgraph")

from utils.logger import StructuredLogger
from utils.config import settings


class WorkflowState(TypedDict):
    """Global state for the multi-agent workflow."""
    agents: Dict[str, Any]
    active_workflows: List[str]
    current_task: Optional[Dict[str, Any]]
    workflow_history: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    healing_requests: List[Dict[str, Any]]
    start_time: str
    last_update: str


class WorkflowStateManager:
    """Manager for workflow state operations."""
    
    def __init__(self):
        self.state: WorkflowState = {
            'agents': {},
            'active_workflows': [],
            'current_task': None,
            'workflow_history': [],
            'errors': [],
            'healing_requests': [],
            'start_time': datetime.utcnow().isoformat(),
            'last_update': datetime.utcnow().isoformat()
        }
    
    def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """Register an agent in the workflow state."""
        self.state['agents'][agent_name] = agent_info
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def update_agent_status(self, agent_name: str, status: str, details: Dict[str, Any] = None):
        """Update agent status in the workflow state."""
        if agent_name in self.state['agents']:
            self.state['agents'][agent_name]['status'] = status
            if details:
                self.state['agents'][agent_name].update(details)
            self.state['last_update'] = datetime.utcnow().isoformat()
    
    def add_workflow_step(self, step_type: str, details: Dict[str, Any]):
        """Add a step to the workflow history."""
        step = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': step_type,
            'details': details
        }
        self.state['workflow_history'].append(step)
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def set_current_task(self, task: Dict[str, Any]):
        """Set the current task being processed."""
        self.state['current_task'] = task
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def clear_current_task(self):
        """Clear the current task."""
        self.state['current_task'] = None
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def add_error(self, error: Dict[str, Any]):
        """Add an error to the workflow state."""
        self.state['errors'].append(error)
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def add_healing_request(self, request: Dict[str, Any]):
        """Add a healing request to the workflow state."""
        self.state['healing_requests'].append(request)
        self.state['last_update'] = datetime.utcnow().isoformat()
    
    def get_agent_status(self, agent_name: str) -> Optional[str]:
        """Get the status of a specific agent."""
        return self.state['agents'].get(agent_name, {}).get('status', 'unknown')
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agents."""
        return [name for name, info in self.state['agents'].items() 
                if info.get('status') == 'active']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary."""
        return self.state.copy()


class LangGraphWorkflowEngine:
    """Main workflow engine using LangGraph for agent orchestration."""
    
    def __init__(self):
        self.logger = StructuredLogger("workflow_engine", settings.log_level)
        self.state_manager = WorkflowStateManager()
        self.graph = None
        self.compiled_graph = None
        self.memory = MemorySaver()
        
        self.logger.info("LangGraph workflow engine initialized")
    
    def create_workflow_graph(self) -> StateGraph:
        """Create the basic workflow graph structure."""
        # Create the state graph with TypedDict
        workflow = StateGraph(WorkflowState)
        
        # Add basic workflow nodes
        workflow.add_node("start", self._start_workflow)
        workflow.add_node("execute_task", self._execute_task)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("check_agents", self._check_agents)
        workflow.add_node("healing_request", self._handle_healing_request)
        workflow.add_node("end", self._end_workflow)
        
        # Define basic edges
        workflow.add_edge("start", "execute_task")
        workflow.add_edge("execute_task", "check_agents")
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "check_agents",
            lambda state: "healing_request" if len(state.get('errors', [])) > 0 else "end",
            {
                "healing_request": "healing_request",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "healing_request", 
            lambda state: "execute_task" if len(state.get('healing_requests', [])) > 0 else "end",
            {
                "execute_task": "execute_task",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "execute_task",
            lambda state: "end" if not state.get('current_task') else "check_agents",
            {
                "end": END,
                "check_agents": "check_agents"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("start")
        
        self.graph = workflow
        self.logger.info("Created basic LangGraph workflow structure")
        return workflow
    
    def _start_workflow(self, state: WorkflowState) -> WorkflowState:
        """Start the workflow execution."""
        self.logger.info("Starting workflow execution")
        
        # Initialize workflow state
        state['active_workflows'] = ["main"]
        state['start_time'] = datetime.utcnow().isoformat()
        
        return state
    
    def _execute_task(self, state: WorkflowState) -> WorkflowState:
        """Execute a task for an agent."""
        if not state.get('current_task'):
            self.logger.warning("No current task to execute")
            return state
        
        task = state['current_task']
        agent_name = task.get('agent')
        task_type = task.get('type')
        
        self.logger.info(f"Executing task for agent: {agent_name}", task_type=task_type)
        
        # Update agent status to working
        if agent_name and agent_name in state['agents']:
            state['agents'][agent_name]['status'] = 'working'
            state['agents'][agent_name].update({
                'current_task': task_type,
                'start_time': datetime.utcnow().isoformat()
            })
        
        # Here we would actually execute the agent task
        # For now, simulate completion
        if agent_name and agent_name in state['agents']:
            state['agents'][agent_name]['status'] = 'completed'
            state['agents'][agent_name].update({
                'completed_task': task_type,
                'completion_time': datetime.utcnow().isoformat()
            })
        
        # Clear current task
        state['current_task'] = None
        
        self.logger.info(f"Task completed for agent: {agent_name}")
        return state
    
    def _check_agents(self, state: WorkflowState) -> WorkflowState:
        """Check status of all agents."""
        self.logger.info("Checking agent statuses")
        
        for agent_name, agent_info in state['agents'].items():
            if agent_info.get('status') == 'active':
                # Here we would check actual agent health
                self.logger.debug(f"Agent {agent_name} is active")
        
        return state
    
    def _handle_healing_request(self, state: WorkflowState) -> WorkflowState:
        """Handle healing requests in the workflow."""
        healing_requests = state.get('healing_requests', [])
        
        if not healing_requests:
            return state
        
        self.logger.info(f"Processing {len(healing_requests)} healing requests")
        
        for request in healing_requests:
            # Here we would process healing requests
            self.logger.info(f"Processing healing request: {request.get('id')}")
        
        # Clear processed requests
        state['healing_requests'] = []
        
        return state
    
    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        errors = state.get('errors', [])
        
        if not errors:
            return state
        
        self.logger.error(f"Handling {len(errors)} errors")
        
        for error in errors:
            # Here we would process errors
            self.logger.error(f"Processing error: {error.get('type')}")
        
        # Clear processed errors
        state['errors'] = []
        
        return state
    
    def _end_workflow(self, state: WorkflowState) -> WorkflowState:
        """End the workflow execution."""
        self.logger.info("Ending workflow execution")
        
        # Calculate workflow duration
        start_time_str = state.get('start_time')
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                duration = datetime.utcnow() - start_time.replace(tzinfo=None)
                self.logger.info(f"Workflow completed in {duration.total_seconds()} seconds")
            except Exception as e:
                self.logger.warning(f"Could not calculate workflow duration: {e}")
        
        return state
    
    def compile_workflow(self):
        """Compile the workflow graph for execution."""
        if not self.graph:
            self.graph = self.create_workflow_graph()
        
        if not self.compiled_graph:
            self.compiled_graph = self.graph.compile(checkpointer=self.memory)
            self.logger.info("Compiled LangGraph workflow")
        
        return self.compiled_graph
    
    async def run_workflow(self, initial_state: Optional[WorkflowState] = None) -> WorkflowState:
        """Run the workflow with LangGraph."""
        if not self.compiled_graph:
            self.compile_workflow()
        
        # Create initial state if not provided
        if not initial_state:
            initial_state = self.state_manager.state
        
        self.logger.info("Starting LangGraph workflow execution")
        
        try:
            # Run the workflow
            config = {"recursion_limit": 100}  # Prevent infinite loops
            result = await self.compiled_graph.ainvoke(
                initial_state,
                config=config
            )
            
            self.logger.info("Workflow execution completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return self.state_manager.to_dict()
    
    def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """Register an agent with the workflow engine."""
        self.state_manager.register_agent(agent_name, agent_info)
        self.logger.info(f"Registered agent: {agent_name}")
    
    def get_agent_status(self, agent_name: str) -> Optional[str]:
        """Get agent status from workflow state."""
        return self.state_manager.get_agent_status(agent_name)