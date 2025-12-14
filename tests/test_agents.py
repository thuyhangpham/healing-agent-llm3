"""
Test Suite for Agents

Unit tests for all agent components including
base agent functionality and individual agent implementations.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.base_agent import BaseAgent


class TestBaseAgent(unittest.TestCase):
    """Test cases for BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"test": "value"}
        self.agent = BaseAgent("test_agent", self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertEqual(self.agent.config, self.config)
    
    def test_execute_not_implemented(self):
        """Test that execute raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.agent.execute({"task": "test"})
    
    def test_process_task_not_implemented(self):
        """Test that _process_task raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.agent._process_task({"task": "test"})


if __name__ == "__main__":
    unittest.main()