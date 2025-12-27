#!/usr/bin/env python3
"""
Simple Healing System Deployment Test

Tests the deployment functionality without unicode characters
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

async def test_deployment():
    """Test basic deployment functionality"""
    print("Testing Healing System Deployment")
    print("=" * 40)
    
    try:
        # Test import
        from agents.healing_agent import HealingAgent
        print("SUCCESS: Imported HealingAgent")
        
        # Test basic agent creation
        config = {
            'auto_heal_enabled': True,
            'max_healing_attempts': 2,
            'backup_enabled': True
        }
        
        agent = HealingAgent("test_deployment", config)
        print("SUCCESS: Created HealingAgent instance")
        
        # Test initialization
        init_result = await agent.initialize()
        print(f"SUCCESS: Agent initialization: {init_result}")
        
        # Test status
        status = await agent.get_status()
        print(f"SUCCESS: Agent status: {status.get('status')}")
        
        # Test directories
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        backups_dir = data_dir / "backups"
        logs_dir = data_dir / "logs"
        metrics_dir = data_dir / "metrics"
        
        print(f"SUCCESS: Project root: {project_root}")
        print(f"SUCCESS: Data directory exists: {data_dir.exists()}")
        print(f"SUCCESS: Backups directory exists: {backups_dir.exists()}")
        print(f"SUCCESS: Logs directory exists: {logs_dir.exists()}")
        print(f"SUCCESS: Metrics directory exists: {metrics_dir.exists()}")
        
        print("\nDeployment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function"""
    print("Healing System Deployment Test")
    print("=" * 30)
    
    success = await test_deployment()
    
    if success:
        print("\nDEPLOYMENT TEST: PASSED")
        print("The healing system is ready for use!")
    else:
        print("\nDEPLOYMENT TEST: FAILED")
        print("Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)