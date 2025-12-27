"""
End-to-End Workflow Demonstration

Demonstrates the complete multi-agent workflow system including:
- Agent registration and discovery
- Task distribution and execution
- Workflow coordination using LangGraph
- Error handling and recovery
- State management and monitoring
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import OrchestratorAgent
from agents.test_agent import TestAgent
from agents.base_agent import TaskPriority
from utils.agent_registry import get_agent_registry
from utils.workflow_engine import LangGraphWorkflowEngine, WorkflowState
from utils.logger import StructuredLogger
from utils.config import settings


class WorkflowDemonstration:
    """
    Comprehensive workflow demonstration for the multi-agent system.
    
    This class demonstrates all the core functionality of the agent framework:
    1. Agent initialization and registration
    2. Task submission and distribution
    3. Workflow execution with LangGraph
    4. Error handling and recovery
    5. State management and monitoring
    """
    
    def __init__(self):
        self.logger = StructuredLogger("workflow_demo", settings.log_level)
        self.orchestrator = None
        self.test_agents = []
        self.registry = get_agent_registry()
        
    async def setup(self):
        """Initialize the demonstration environment."""
        self.logger.info("Setting up workflow demonstration")
        
        try:
            # Initialize orchestrator
            self.orchestrator = OrchestratorAgent({
                'health_check_interval': 5,
                'max_concurrent_tasks': 3,
                'task_timeout': 30
            })
            
            # Create test agents with different capabilities
            await self._create_test_agents()
            
            # Register agents with orchestrator
            await self._register_agents()
            
            # Start orchestrator monitoring
            await self.orchestrator.start_monitoring()
            
            self.logger.info("Workflow demonstration setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup demonstration: {e}")
            raise
    
    async def _create_test_agents(self):
        """Create test agents with different capabilities."""
        agent_configs = [
            {
                'name': 'echo_agent',
                'capabilities': ['echo', 'basic_processing'],
                'config': {'max_retries': 2, 'timeout': 10}
            },
            {
                'name': 'transform_agent', 
                'capabilities': ['data_transformation', 'text_processing'],
                'config': {'max_retries': 3, 'timeout': 15}
            },
            {
                'name': 'compute_agent',
                'capabilities': ['computation', 'math_operations'],
                'config': {'max_retries': 1, 'timeout': 5}
            },
            {
                'name': 'status_agent',
                'capabilities': ['status_reporting', 'monitoring'],
                'config': {'max_retries': 2, 'timeout': 8}
            }
        ]
        
        for agent_config in agent_configs:
            agent = TestAgent(agent_config['name'], agent_config['config'])
            self.test_agents.append({
                'agent': agent,
                'capabilities': agent_config['capabilities']
            })
            self.logger.info(f"Created test agent: {agent_config['name']}")
    
    async def _register_agents(self):
        """Register all test agents with the orchestrator."""
        for agent_info in self.test_agents:
            agent = agent_info['agent']
            capabilities = agent_info['capabilities']
            
            # Register with orchestrator
            self.orchestrator.register_agent(agent, capabilities)
            
            # Also register with global registry
            self.registry.register_agent(agent, capabilities)
            
            self.logger.info(f"Registered agent {agent.name} with capabilities: {capabilities}")
    
    async def demonstrate_basic_workflow(self):
        """Demonstrate basic workflow execution."""
        self.logger.info("=== Starting Basic Workflow Demonstration ===")
        
        try:
            # Submit various tasks to demonstrate different capabilities
            tasks = [
                {
                    'id': 'echo_task_1',
                    'type': 'echo',
                    'data': {'message': 'Hello from workflow demo!'},
                    'required_capabilities': ['echo']
                },
                {
                    'id': 'transform_task_1',
                    'type': 'transform',
                    'data': {
                        'text': 'sample text',
                        'number': 42,
                        'items': ['a', 'b', 'c']
                    },
                    'required_capabilities': ['data_transformation']
                },
                {
                    'id': 'compute_task_1',
                    'type': 'compute',
                    'data': {
                        'numbers': [1, 2, 3, 4, 5],
                        'operation': 'sum'
                    },
                    'required_capabilities': ['computation']
                },
                {
                    'id': 'status_task_1',
                    'type': 'status',
                    'data': {},
                    'required_capabilities': ['status_reporting']
                }
            ]
            
            # Submit tasks to orchestrator
            for task in tasks:
                self.orchestrator.submit_task(
                    task,
                    task.get('required_capabilities', [])
                )
                self.logger.info(f"Submitted task: {task['id']}")
            
            # Wait for tasks to be processed
            await asyncio.sleep(10)
            
            # Check results
            await self._check_task_results()
            
            self.logger.info("Basic workflow demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Basic workflow demonstration failed: {e}")
            raise
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling and recovery."""
        self.logger.info("=== Starting Error Handling Demonstration ===")
        
        try:
            # Submit a task that will cause an error
            error_task = {
                'id': 'error_task_1',
                'type': 'error',
                'data': {
                    'error_type': 'value',
                    'error_message': 'Intentional error for demonstration'
                },
                'required_capabilities': ['echo']  # Any agent can handle this
            }
            
            self.orchestrator.submit_task(error_task, ['echo'])
            self.logger.info("Submitted error task for demonstration")
            
            # Wait for error handling
            await asyncio.sleep(5)
            
            # Submit a normal task to verify recovery
            recovery_task = {
                'id': 'recovery_task_1',
                'type': 'echo',
                'data': {'message': 'System recovered successfully'},
                'required_capabilities': ['echo']
            }
            
            self.orchestrator.submit_task(recovery_task, ['echo'])
            self.logger.info("Submitted recovery task")
            
            # Wait for recovery
            await asyncio.sleep(5)
            
            self.logger.info("Error handling demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Error handling demonstration failed: {e}")
            raise
    
    async def demonstrate_langgraph_workflow(self):
        """Demonstrate LangGraph workflow execution."""
        self.logger.info("=== Starting LangGraph Workflow Demonstration ===")
        
        try:
            # Create a workflow state manager
            from utils.workflow_engine import WorkflowStateManager
            workflow_state_manager = WorkflowStateManager()
            
            # Register agents with workflow state
            for agent_info in self.test_agents:
                agent = agent_info['agent']
                workflow_state_manager.register_agent(agent.name, {
                    'type': 'test_agent',
                    'status': agent.get_status().value,
                    'capabilities': agent_info['capabilities']
                })
            
            # Set a current task for the workflow
            workflow_state_manager.set_current_task({
                'id': 'workflow_task_1',
                'type': 'demonstration',
                'description': 'LangGraph workflow demonstration',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Execute workflow using LangGraph
            if self.orchestrator.workflow_engine:
                # Create a simple workflow state dict for LangGraph
                workflow_state_dict = workflow_state_manager.state
                
                result = await self.orchestrator.workflow_engine.run_workflow(workflow_state_dict)
                
                self.logger.info(f"LangGraph workflow result: {result}")
            else:
                self.logger.warning("Workflow engine not available")
            
            self.logger.info("LangGraph workflow demonstration completed")
            
        except Exception as e:
            self.logger.error(f"LangGraph workflow demonstration failed: {e}")
            raise
    
    async def demonstrate_monitoring(self):
        """Demonstrate system monitoring and health checks."""
        self.logger.info("=== Starting Monitoring Demonstration ===")
        
        try:
            # Get system status
            system_status = self.orchestrator.get_system_status()
            self.logger.info(f"System status: {system_status}")
            
            # Get registry statistics
            registry_stats = self.registry.get_registry_stats()
            self.logger.info(f"Registry statistics: {registry_stats}")
            
            # Perform health check
            health_result = await self.orchestrator.execute({
                'type': 'health_check',
                'id': 'demo_health_check'
            })
            self.logger.info(f"Health check result: {health_result}")
            
            # Get agent summaries
            for agent_info in self.test_agents:
                agent = agent_info['agent']
                if hasattr(agent, 'get_test_summary'):
                    summary = agent.get_test_summary()
                    self.logger.info(f"Agent {agent.name} summary: {summary}")
            
            self.logger.info("Monitoring demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Monitoring demonstration failed: {e}")
            raise
    
    async def _check_task_results(self):
        """Check and log task execution results."""
        for agent_info in self.test_agents:
            agent = agent_info['agent']
            metrics = agent.get_metrics()
            
            if metrics['tasks_completed'] > 0 or metrics['tasks_failed'] > 0:
                self.logger.info(f"Agent {agent.name} metrics: {metrics}")
    
    async def cleanup(self):
        """Clean up demonstration resources."""
        self.logger.info("Cleaning up workflow demonstration")
        
        try:
            # Stop orchestrator monitoring
            if self.orchestrator:
                await self.orchestrator.stop_monitoring()
                
                # Shutdown all agents
                for agent_info in self.test_agents:
                    agent = agent_info['agent']
                    await agent.shutdown()
                
                # Shutdown orchestrator
                await self.orchestrator.shutdown()
            
            self.logger.info("Workflow demonstration cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def run_complete_demonstration(self):
        """Run the complete workflow demonstration."""
        self.logger.info("Starting complete workflow demonstration")
        
        try:
            # Setup
            await self.setup()
            
            # Run demonstrations
            await self.demonstrate_basic_workflow()
            await self.demonstrate_error_handling()
            await self.demonstrate_langgraph_workflow()
            await self.demonstrate_monitoring()
            
            self.logger.info("Complete workflow demonstration finished successfully")
            
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()


async def main():
    """Main function to run the workflow demonstration."""
    print("Starting ETL Sentiment Multi-Agent Workflow Demonstration")
    print("=" * 60)
    
    demo = WorkflowDemonstration()
    
    try:
        await demo.run_complete_demonstration()
        print("\nWorkflow demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
        await demo.cleanup()
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        await demo.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())