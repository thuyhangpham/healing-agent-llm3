"""
Example of how to integrate SentimentAnalysisAgent with orchestrator
"""

import asyncio
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.orchestrator import OrchestratorAgent


async def example_integration():
    """Example of integrating sentiment analysis with orchestrator."""
    print("=== ETL Sentiment Analysis Integration Example ===\n")
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent({
        'health_check_interval': 60,
        'max_concurrent_tasks': 3,
        'task_timeout': 300
    })
    
    # Initialize orchestrator components
    await orchestrator._on_initialize()
    
    # Initialize sentiment analysis agent
    sentiment_agent = SentimentAnalysisAgent({
        'model': 'llama3:latest',
        'temperature': 0.3,
        'confidence_threshold': 0.7,
        'batch_size': 3
    })
    
    await sentiment_agent.initialize()
    
    try:
        # Register sentiment analysis agent with orchestrator
        orchestrator.register_agent(
            sentiment_agent,
            capabilities=['sentiment_analysis', 'llm_processing', 'legal_context_analysis']
        )
        
        # Submit sentiment analysis task
        task = {
            'id': 'sentiment_analysis_main',
            'type': 'analyze_sentiment',
            'priority': 'high',
            'description': 'Analyze sentiment of opinion articles with legal context'
        }
        
        orchestrator.submit_task(
            task,
            required_capabilities=['sentiment_analysis', 'llm_processing']
        )
        
        print(f"✅ Sentiment analysis task submitted to orchestrator")
        print(f"📊 Task ID: {task['id']}")
        print(f"🎯 Agent: {sentiment_agent.name}")
        print(f"🔧 Capabilities: sentiment_analysis, llm_processing, legal_context_analysis")
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"\n📈 System Status:")
        print(f"  - Orchestrator Status: {status['orchestrator']['status']}")
        print(f"  - Registered Agents: {status['agents']['registered']}")
        print(f"  - Tasks in Queue: {status['tasks']['queued']}")
        
        # Wait a bit for processing
        await asyncio.sleep(5)
        
        # Get updated status
        updated_status = orchestrator.get_system_status()
        print(f"\n⏱️ Updated Status:")
        print(f"  - Tasks Completed: {updated_status['tasks']['completed']}")
        print(f"  - Active Tasks: {updated_status['tasks']['assigned']}")
        
        print(f"\n✅ Integration example completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await sentiment_agent.shutdown()
        await orchestrator._on_shutdown()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(example_integration())