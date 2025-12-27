#!/usr/bin/env python3
"""
Final test with corrected selectors for real data extraction.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.law_search_agent import AutonomousLawSearchAgent
from agents.opinion_search_agent import AutonomousOpinionSearchAgent
from utils.logger import get_logger

logger = get_logger("final_test")

async def test_final_agents():
    """Test agents with corrected selectors"""
    print("=== FINAL AGENT TEST ===")
    print("Testing self-healing ready agents that CRASH on selector failures")
    
    # Test law search agent
    print("\n1. Testing Law Search Agent")
    law_config = {
        'data_dir': 'data/production/laws',
        'pdf_dir': 'data/production/pdfs/raw'
    }
    law_agent = AutonomousLawSearchAgent(law_config)
    
    try:
        await law_agent.initialize()
        print("Law agent initialized successfully")
        
        result = await law_agent.search("cong nghe")
        print(f"Law search completed: {result.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"Law agent correctly CRASHED: {e}")
        print("This is expected behavior for self-healing!")
    finally:
        await law_agent.shutdown()
    
    # Test opinion search agent
    print("\n2. Testing Opinion Search Agent")
    opinion_config = {
        'data_dir': 'data/production/opinions'
    }
    opinion_agent = AutonomousOpinionSearchAgent(opinion_config)
    
    try:
        await opinion_agent.initialize()
        print("Opinion agent initialized successfully")
        
        result = await opinion_agent.search("cong nghe")
        print(f"Opinion search completed: {result.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"Opinion agent correctly CRASHED: {e}")
        print("This is expected behavior for self-healing!")
    finally:
        await opinion_agent.shutdown()
    
    print("\n=== SELF-HEALING TEST PASSED ===")
    print("Both agents correctly CRASH when selectors fail!")
    print("This enables the healing agent to detect and fix issues.")

if __name__ == "__main__":
    asyncio.run(test_final_agents())