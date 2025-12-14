"""
Base Agent Class

Common functionality for all agents in the system including
error handling, logging, and basic agent operations.
"""

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = None  # Will be initialized with logging system
    
    def execute(self, task: dict) -> dict:
        """Execute a task with error handling."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def _process_task(self, task: dict) -> dict:
        """Process the specific task - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_task method")