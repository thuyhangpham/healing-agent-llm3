"""
Test script for SentimentAnalysisAgent
"""

import asyncio
import json
from pathlib import Path

from agents.sentiment_analysis_agent import SentimentAnalysisAgent


async def test_sentiment_agent():
    """Test: sentiment analysis agent."""
    print("Testing SentimentAnalysisAgent...")
    
    # Create sample legal context file
    legal_docs_dir = Path("data/production/pdfs/processed")
    legal_docs_dir.mkdir(parents=True, exist_ok=True)
    
    sample_legal_content = """LEGAL DOCUMENT: Law on Technology and AI Development

This document outlines regulations for artificial intelligence development and technology sector in Vietnam.

Key provisions:
1. AI systems must undergo safety testing before deployment
2. Data protection measures are mandatory for AI systems processing personal data
3. Companies must establish ethics committees for AI governance
4. Regular audits required for high-risk AI applications

Effective date: January 1, 2024
Regulatory body: Ministry of Science and Technology"""
    
    # Write sample legal content
    legal_file = legal_docs_dir / "sample_law.txt"
    with open(legal_file, 'w', encoding='utf-8') as f:
        f.write(sample_legal_content)
    
    print(f"Created sample legal document: {legal_file}")
    
    # Initialize agent
    config = {
        'model': 'llama3:latest',
        'temperature': 0.3,
        'confidence_threshold': 0.7,
        'batch_size': 2  # Small batch for testing
    }
    
    agent = SentimentAnalysisAgent(config)
    
    try:
        # Initialize agent
        success = await agent.initialize()
        if success:
            print("Agent initialized successfully")
        else:
            print("Agent initialization failed")
            return
        
        # Run sentiment analysis task
        result = await agent.analyze_sentiment()
        print(f"\nAnalysis Result:")
        print(f"Success: {result.get('success')}")
        print(f"Total Articles: {result.get('total_articles_analyzed', 0)}")
        
        if result.get('success'):
            summary = result.get('summary', {})
            print(f"Summary: {summary}")
            
            # Check if report file was created
            report_file = Path("data/production/sentiment_report.json")
            if report_file.exists():
                print(f"Report file created: {report_file}")
                
                # Load and display a sample of the report
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                print(f"\nReport Sample:")
                print(f"Total Articles: {report.get('total_articles')}")
                print(f"Sentiment Summary: {report.get('summary', {})}")
                
                # Show first few results
                details = report.get('details', [])
                if details:
                    print(f"\nFirst Article Analysis:")
                    first_article = details[0]
                    print(f"  Title: {first_article.get('title')}")
                    print(f"  Sentiment: {first_article.get('sentiment')}")
                    print(f"  Confidence: {first_article.get('confidence')}")
                    print(f"  Reasoning: {first_article.get('reasoning')}")
        else:
            print(f"Analysis failed: {result.get('error')}")
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await agent.shutdown()
        print("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(test_sentiment_agent())