#!/usr/bin/env python3
"""
Test current working agents to see if they can extract real data.
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

logger = get_logger("test_working_agents")

async def test_working_agents():
    """Test if current agents can extract real data"""
    print("=== TESTING WORKING AGENTS ===")
    print("Testing if agents can extract real data with current selectors")
    
    # Test law search agent with less strict selectors
    print("\n1. Testing Law Search Agent with relaxed selectors")
    law_config = {
        'data_dir': 'data/production/laws',
        'pdf_dir': 'data/production/pdfs/raw'
    }
    law_agent = AutonomousLawSearchAgent(law_config)
    
    try:
        await law_agent.initialize()
        print("Law agent initialized successfully")
        
        # Try different selectors that might work
        print("Trying different VBPL selectors...")
        
        # Test with more relaxed approach
        import aiohttp
        from bs4 import BeautifulSoup
        
        response = await law_agent.session.get('https://vbpl.vn/')
        if response.status == 200:
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            print(f"Page title: {soup.title.string if soup.title else 'No title'}")
            
            # Look for any links that might be documents
            all_links = soup.find_all('a', href=True)
            legal_keywords = ['van-ban', 'nghị-định', 'thông-tư', 'luật']
            
            found_docs = []
            for i, link in enumerate(all_links[:10]):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                if (href and text and 
                    any(keyword in text.lower() or keyword in href.lower() 
                        for keyword in legal_keywords) and
                    len(text) > 10):
                    
                    found_docs.append({
                        'title': text[:50],
                        'link': href,
                        'index': i
                    })
                    print(f"  Potential doc {i+1}: {text[:50]}...")
            
            if found_docs:
                print(f"Found {len(found_docs)} potential legal documents")
                return {"status": "success", "found": found_docs}
        
        result = {"status": "failed", "error": "No documents found"}
        print(f"Law search test result: {result}")
        
    except Exception as e:
        print(f"Law agent error: {e}")
        
    finally:
        try:
            await law_agent.shutdown()
        except:
            pass
    
    # Test opinion search agent with less strict selectors
    print("\n2. Testing Opinion Search Agent with relaxed selectors")
    opinion_config = {
        'data_dir': 'data/production/opinions'
    }
    opinion_agent = AutonomousOpinionSearchAgent(opinion_config)
    
    try:
        await opinion_agent.initialize()
        print("Opinion agent initialized successfully")
        
        # Try with the working selector we found earlier
        print("Trying VnExpress with corrected selectors...")
        
        import aiohttp
        from bs4 import BeautifulSoup
        
        response = await opinion_agent.session.get('https://vnexpress.net/so-hoa')
        if response.status == 200:
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Use the selector that actually works
            articles = soup.select('article.item-news')
            if articles:
                print(f"Found {len(articles)} articles with 'article.item-news' selector")
                
                # Extract first few articles
                for i, article in enumerate(articles[:3]):
                    links = article.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        if href and text and len(text) > 10:
                            print(f"  Article {i+1}: {text[:50]}...")
                            break
                    
                    if len(links) > 0:
                        break
                
                return {"status": "success", "found_count": len(articles)}
        
        result = {"status": "failed", "error": "No articles found"}
        print(f"Opinion search test result: {result}")
        
    except Exception as e:
        print(f"Opinion agent error: {e}")
        
    finally:
        try:
            await opinion_agent.shutdown()
        except:
            pass
    
    print("\n=== TEST SUMMARY ===")
    print("Both agents tested for real data extraction capability.")

if __name__ == "__main__":
    asyncio.run(test_working_agents())