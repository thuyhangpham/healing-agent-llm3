#!/usr/bin/env python3
"""
Minimal test for the healing system to verify basic functionality
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_healing_agent_basic():
    """Test basic healing agent functionality"""
    print("Testing HealingAgent Basic Functionality")
    print("=" * 50)
    
    try:
        # Import the healing agent
        from agents.healing_agent import HealingAgent, ErrorContext, ErrorSeverity
        
        print("SUCCESS: Successfully imported HealingAgent")
        
        # Create a healing agent instance
        config = {
            'auto_heal_enabled': True,
            'max_healing_attempts': 2,
            'backup_enabled': True,
            'llm': {
                'base_url': 'http://localhost:11434',
                'model': 'llama3'
            }
        }
        
        agent = HealingAgent("test_healing_agent", config)
        print("SUCCESS: Successfully created HealingAgent instance")
        
        # Test initialization
        init_result = await agent.initialize()
        print(f"SUCCESS: Initialization result: {init_result}")
        
        # Test status
        status = await agent.get_status()
        print(f"SUCCESS: Agent status: {status.get('status')}")
        print(f"SUCCESS: Agent ID: {status.get('agent_id')}")
        print(f"SUCCESS: Auto-heal enabled: {status.get('auto_heal_enabled')}")
        
        # Test error context creation
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': 'TestError',
            'error_message': 'This is a test error',
            'traceback': 'Test traceback',
            'agent_name': 'TestAgent',
            'function_name': 'test_function',
            'file_path': '/test/path.py',
            'line_number': 42
        }
        
        error_context = agent._create_error_context(error_data)
        print(f"SUCCESS: Created error context: {error_context.error_id}")
        print(f"SUCCESS: Error severity: {error_context.severity}")
        
        # Test task processing (basic task)
        task = {
            'type': 'get_metrics',
            'id': 'test_task_1'
        }
        
        result = await agent.execute(task)
        print(f"SUCCESS: Task execution result: {result.get('success')}")
        
        print("\nAll basic tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        return False

async def test_error_handling():
    """Test error handling capabilities"""
    print("\nTesting Error Handling")
    print("=" * 30)
    
    try:
        from agents.healing_agent import HealingAgent
        
        agent = HealingAgent("test_error_handler")
        await agent.initialize()
        
        # Test error report handling
        error_report = {
            'type': 'error_report',
            'error_context': {
                'timestamp': datetime.now().isoformat(),
                'error_type': 'ValueError',
                'error_message': 'Test value error',
                'traceback': 'ValueError: Test value error\n  at test.py:10',
                'agent_name': 'TestAgent',
                'function_name': 'test_func',
                'file_path': 'test.py',
                'line_number': 10
            }
        }
        
        # This should not crash even if LLM is not available
        result = await agent.process_message(error_report)
        print(f"SUCCESS: Error report processed: {result.get('type')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error handling test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("Starting Healing System Tests")
    print("=" * 40)
    
    # Test basic functionality
    basic_test = await test_healing_agent_basic()
    
    # Test error handling
    error_test = await test_error_handling()
    
    # Summary
    print("\nTest Summary")
    print("=" * 20)
    print(f"Basic Functionality: {'PASS' if basic_test else 'FAIL'}")
    print(f"Error Handling: {'PASS' if error_test else 'FAIL'}")
    
    overall_success = basic_test and error_test
    print(f"\nOverall Result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)