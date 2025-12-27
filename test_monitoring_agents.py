#!/usr/bin/env python3
"""
Test script for monitoring agents - law search and opinion search.
This script tests agents with real data extraction.
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

logger = get_logger("test_monitoring_agents")


async def test_law_search_agent():
    """Test law search agent"""
    print("\n" + "="*60)
    print("TESTING LAW SEARCH AGENT")
    print("="*60)
    
    # Configuration for law search agent
    law_config = {
        'timeout': 30,
        'rate_limit': 1.0,
        'data_dir': 'data/production/laws',
        'sources': [
            {
                'name': 'VBPL',
                'url': 'https://vbpl.vn/',
                'enabled': True
            }
        ]
    }
    
    # Initialize and test
    agent = AutonomousLawSearchAgent(law_config)
    
    try:
        # Initialize agent
        success = await agent.initialize()
        if not success:
            print("FAILED to initialize law search agent")
            return False
        
        print("Law search agent initialized successfully")
        
        # Test search
        query = "công nghệ thông tin"
        result = await agent.search(query)
        
        if result['status'] == 'success':
            print(f"Law search successful: {result['total_results']} documents found")
            print(f"Documents saved to: {law_config['data_dir']}")
            
            # Show found documents
            for i, doc in enumerate(result['results'][:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"  Title: {doc['title'][:80]}...")
                print(f"  Number: {doc['document_number']}")
                print(f"  File: {doc['filename']}")
        else:
            print(f"Law search failed: {result['error']}")
            return False
        
    except Exception as e:
        print(f"Exception in law search agent: {e}")
        return False
    finally:
        await agent.shutdown()
    
    return True


async def test_opinion_search_agent():
    """Test opinion search agent"""
    print("\n" + "="*60)
    print("TESTING OPINION SEARCH AGENT")
    print("="*60)
    
    # Configuration for opinion search agent
    opinion_config = {
        'timeout': 30,
        'rate_limit': 1.0,
        'data_dir': 'data/production/opinions',
        'use_selenium': False,  # Start with requests, can enable selenium if needed
        'sources': [
            {
                'name': 'VnExpress',
                'url': 'https://vnexpress.net/so-hoa',
                'enabled': True
            }
        ]
    }
    
    # Initialize and test
    agent = AutonomousOpinionSearchAgent(opinion_config)
    
    try:
        # Initialize agent
        success = await agent.initialize()
        if not success:
            print("FAILED to initialize opinion search agent")
            return False
        
        print("Opinion search agent initialized successfully")
        
        # Test search
        query = "trí tuệ nhân tạo"
        result = await agent.search(query)
        
        if result['status'] == 'success':
            print(f"Opinion search successful: {result['total_results']} articles found")
            print(f"Articles saved to: {opinion_config['data_dir']}")
            
            # Show found articles
            for i, article in enumerate(result['results'][:3]):  # Show first 3
                print(f"\nArticle {i+1}:")
                print(f"  Title: {article['title'][:80]}...")
                print(f"  Sapo: {article['sapo'][:100]}...")
                print(f"  File: {article['filename']}")
        else:
            print(f"Opinion search failed: {result['error']}")
            return False
        
    except Exception as e:
        print(f"Exception in opinion search agent: {e}")
        return False
    finally:
        await agent.shutdown()
    
    return True


async def main():
    """Main test function"""
    print("Starting Monitoring Squad Tests")
    print("Testing agents with real data extraction...")
    
    # Create data directories
    Path('data/production/laws').mkdir(parents=True, exist_ok=True)
    Path('data/production/opinions').mkdir(parents=True, exist_ok=True)
    
    # Test law search agent
    law_success = await test_law_search_agent()
    
    # Test opinion search agent
    opinion_success = await test_opinion_search_agent()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Law Search Agent: {'PASSED' if law_success else 'FAILED'}")
    print(f"Opinion Search Agent: {'PASSED' if opinion_success else 'FAILED'}")
    
    if law_success and opinion_success:
        print("\nALL TESTS PASSED! Monitoring Squad is ready.")
    else:
        print("\nSome tests failed. Check logs above.")
    
    return law_success and opinion_success


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(main())
    sys.exit(0 if result else 1)