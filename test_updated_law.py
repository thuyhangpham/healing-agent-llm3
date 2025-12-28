#!/usr/bin/env python3
"""
Test updated law search agent.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.law_search_agent import AutonomousLawSearchAgent
from utils.logger import get_logger

logger = get_logger("test_updated_law")

async def test_updated_law():
    """Test updated law search agent"""
    print("=== TESTING UPDATED LAW SEARCH AGENT ===")
    
    # Test law search agent
    print("\nTesting updated Law Search Agent...")
    law_config = {
        'data_dir': 'data/production/laws',
        'pdf_dir': 'data/production/pdfs/raw'
    }
    law_agent = AutonomousLawSearchAgent(law_config)
    
    try:
        await law_agent.initialize()
        print("Law agent initialized successfully")
        
        result = await law_agent.search("cong nghe")
        print(f"Law search result: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            print(f"Successfully extracted {result.get('total_results', 0)} legal documents")
            for i, doc in enumerate(result.get('results', [])):
                print(f"  {i+1}. {doc.get('title', 'N/A')}")
                print(f"      Link: {doc.get('link', 'N/A')}")
                print(f"      Attachment: {doc.get('attachment_path', 'N/A')}")
        
    except Exception as e:
        print(f"Law agent error: {e}")
        
    finally:
        await law_agent.shutdown()
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(test_updated_law())