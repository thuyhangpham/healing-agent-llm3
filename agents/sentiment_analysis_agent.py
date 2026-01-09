"""
Sentiment Analysis Agent

Analyzes sentiment of opinion articles in the context of legal documents.
Strictly follows the architecture diagram:
- TARGET: Opinion articles from data/production/opinions/*.json
- CONTEXT: Legal laws from data/production/laws/*.json (most recent)
- MODEL: LLMClient (Llama 3 via Ollama)
"""

import sys
import os
# Add project root to sys.path so we can import 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from utils.logger import get_logger

try:
    from core.llm_client import LLMClient
except ImportError:
    LLMClient = None


class SentimentAnalysisAgent:
    """
    Agent for analyzing sentiment of opinion articles with legal document context.
    
    Architecture:
    - TARGET: Reads opinion articles from data/production/opinions/*.json
    - CONTEXT: Reads legal laws from data/production/laws/*.json (most recent)
    - MODEL: Uses LLMClient (Llama 3 via Ollama)
    - OUTPUT: Updates opinion JSON files and generates CSV summary report
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sentiment analysis agent."""
        self.config = config or {}
        self.name = "sentiment_analysis_agent"
        self.logger = get_logger(self.name)
        
        # Directories
        self.opinions_dir = Path("data/production/opinions")
        self.laws_dir = Path("data/production/laws")
        self.reports_dir = Path("data/reports")
        
        # Output file
        self.summary_csv = self.reports_dir / "sentiment_summary.csv"
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM configuration
        self.llm_client: Optional[LLMClient] = None
        self.model_name = self.config.get('model', 'llama3:latest')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 2048)
        
        # Sentiment labels
        self.sentiment_labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        
        self.logger.info("Sentiment Analysis Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize sentiment analysis agent."""
        try:
            self.logger.info("Initializing sentiment analysis agent...")
            
            # Check if input directories exist
            if not self.opinions_dir.exists():
                self.logger.warning(f"Opinions directory does not exist: {self.opinions_dir}")
                self.opinions_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.laws_dir.exists():
                self.logger.warning(f"Laws directory does not exist: {self.laws_dir}")
                self.laws_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize LLM client
            if LLMClient is not None:
                llm_config = {
                    'base_url': self.config.get('ollama_base_url', 'http://localhost:11434'),
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'timeout': self.config.get('timeout', 60)
                }
                self.llm_client = LLMClient(llm_config)
                
                if await self.llm_client.initialize():
                    self.logger.info(f"LLM client initialized with model: {self.model_name}")
                else:
                    self.logger.error("Failed to initialize LLM client")
                    return False
            else:
                self.logger.warning("LLMClient not available - sentiment analysis will use fallback")
            
            self.logger.info("Sentiment analysis agent initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analysis agent: {e}")
            return False
    
    async def analyze_sentiment(self) -> Dict[str, Any]:
        """
        Main sentiment analysis method.
        
        Returns:
            Dictionary with analysis results and summary
        """
        try:
            self.logger.info("Starting sentiment analysis task")
            
            # Load most recent law for context
            law_context = await self._load_most_recent_law()
            if not law_context:
                self.logger.warning("No law context found - proceeding without context")
                law_summary = "No legal context available"
            else:
                law_summary = law_context.get('summary', law_context.get('title', 'Legal document'))
            
            # Load opinion articles
            opinion_files = list(self.opinions_dir.glob("*.json"))
            if not opinion_files:
                self.logger.warning("No opinion articles found")
                return {
                    'success': False,
                    'error': 'No opinion articles found',
                    'total_articles': 0
                }
            
            self.logger.info(f"Found {len(opinion_files)} opinion articles to analyze")
            
            # Analyze each article
            results = []
            for article_file in opinion_files:
                try:
                    result = await self._analyze_single_article(article_file, law_summary)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to analyze article {article_file.name}: {e}")
                    continue
            
            # Generate and save CSV summary
            await self._save_summary_csv(results)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(results)
            
            self.logger.info(f"Sentiment analysis completed: {len(results)} articles analyzed")
            
            return {
                'success': True,
                'total_articles_analyzed': len(results),
                'summary': summary_stats,
                'summary_csv': str(self.summary_csv),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis task failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'total_articles': 0,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def _load_most_recent_law(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recently crawled law from data/production/laws/*.json.
        
        Returns:
            Dictionary with law data, or None if no laws found
        """
        try:
            law_files = list(self.laws_dir.glob("*.json"))
            if not law_files:
                self.logger.warning("No law files found")
                return None
            
            # Sort by modification time (most recent first)
            law_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            most_recent_law_file = law_files[0]
            
            self.logger.info(f"Loading most recent law: {most_recent_law_file.name}")
            
            with open(most_recent_law_file, 'r', encoding='utf-8') as f:
                law_data = json.load(f)
            
            # Try to load processed text if available
            # Check if there's a processed text file
            attachment_path = law_data.get('attachment_path', '')
            if attachment_path:
                processed_path = Path(attachment_path.replace('/raw/', '/processed/').replace('.pdf', '.txt'))
                if processed_path.exists():
                    with open(processed_path, 'r', encoding='utf-8') as f:
                        processed_text = f.read()
                        # Limit to first 5000 chars for context
                        if len(processed_text) > 5000:
                            processed_text = processed_text[:5000] + "...(truncated)"
                        law_data['summary'] = processed_text
                else:
                    # Use title and metadata as summary
                    law_data['summary'] = f"{law_data.get('title', '')}\nDocument: {law_data.get('document_number', 'N/A')}"
            
            return law_data
            
        except Exception as e:
            self.logger.error(f"Failed to load most recent law: {e}")
            return None
    
    async def _analyze_single_article(
        self, 
        article_file: Path, 
        law_summary: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment of a single article in the context of the law.
        
        Args:
            article_file: Path to the opinion article JSON file
            law_summary: Summary of the legal document context
            
        Returns:
            Dictionary with analysis result, or None if analysis failed
        """
        try:
            # Load article
            with open(article_file, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            article_title = article.get('title', '')
            article_sapo = article.get('sapo', '')
            
            if not article_title:
                self.logger.warning(f"Article {article_file.name} has no title, skipping")
                return None
            
            # Build prompt according to specification
            prompt = f"""CONTEXT (The Law):
{law_summary}

TARGET (The Article):
Title: {article_title}
Summary: {article_sapo}

INSTRUCTION:
Based on the provided Law Context, analyze the sentiment of the Article regarding the impact of this law on the 'Tech Industry/Innovation'.
Is the article supporting the law, criticizing it, or neutral?

RESPONSE FORMAT (JSON only):
{{
    "sentiment_label": "POSITIVE|NEGATIVE|NEUTRAL",
    "confidence": 0.0-1.0,
    "analysis_reasoning": "Brief explanation of the sentiment analysis"
}}

Labels:
- POSITIVE: Supports the law/regulation, sees it as an enabler for tech.
- NEGATIVE: Criticizes the law, sees it as a barrier/restriction.
- NEUTRAL: Informational only.

Analyze now:"""
            
            # Get LLM response
            if self.llm_client:
                response = await self.llm_client._generate_response(prompt)
                
                if response.error:
                    self.logger.warning(f"LLM error for article {article_file.name}: {response.error}")
                    # Use fallback
                    sentiment_data = self._fallback_sentiment(article_title, article_sapo)
                else:
                    # Parse LLM response
                    sentiment_data = self._parse_llm_response(response.content)
                    if not sentiment_data:
                        # Fallback if parsing fails
                        sentiment_data = self._fallback_sentiment(article_title, article_sapo)
            else:
                # Fallback if LLM not available
                sentiment_data = self._fallback_sentiment(article_title, article_sapo)
            
            # Update article JSON file
            article['sentiment_label'] = sentiment_data['sentiment_label']
            article['confidence'] = sentiment_data['confidence']
            article['analysis_reasoning'] = sentiment_data['analysis_reasoning']
            article['sentiment_analyzed_at'] = datetime.now().isoformat()
            
            # Save updated article
            with open(article_file, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Analyzed article {article_file.name}: {sentiment_data['sentiment_label']} (confidence: {sentiment_data['confidence']:.2f})")
            
            return {
                'article_id': article.get('id', article_file.stem),
                'title': article_title,
                'link': article.get('link', ''),
                'date': article.get('date', ''),
                'sentiment_label': sentiment_data['sentiment_label'],
                'confidence': sentiment_data['confidence'],
                'analysis_reasoning': sentiment_data['analysis_reasoning'],
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze article {article_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_llm_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract sentiment data.
        
        Args:
            response_content: Raw LLM response string
            
        Returns:
            Dictionary with sentiment_label, confidence, analysis_reasoning, or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # Look for JSON object in the response
            json_pattern = r'\{[^{}]*"sentiment_label"[^{}]*"analysis_reasoning"[^{}]*\}'
            matches = re.findall(json_pattern, response_content, re.DOTALL)
            
            if not matches:
                # Try simpler pattern
                json_pattern = r'\{.*?"sentiment_label".*?\}'
                matches = re.findall(json_pattern, response_content, re.DOTALL)
            
            if matches:
                json_str = matches[0]
                parsed = json.loads(json_str)
                
                sentiment_label = parsed.get('sentiment_label', 'NEUTRAL').upper()
                if sentiment_label not in self.sentiment_labels:
                    sentiment_label = 'NEUTRAL'
                
                confidence = float(parsed.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                
                analysis_reasoning = parsed.get('analysis_reasoning', 'No reasoning provided')
                
                return {
                    'sentiment_label': sentiment_label,
                    'confidence': confidence,
                    'analysis_reasoning': analysis_reasoning
                }
            else:
                # Try to extract sentiment from text
                response_lower = response_content.lower()
                if 'positive' in response_lower:
                    sentiment_label = 'POSITIVE'
                elif 'negative' in response_lower:
                    sentiment_label = 'NEGATIVE'
                else:
                    sentiment_label = 'NEUTRAL'
                
                # Try to extract confidence number
                confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', response_lower)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                confidence = max(0.0, min(1.0, confidence))
                
                return {
                    'sentiment_label': sentiment_label,
                    'confidence': confidence,
                    'analysis_reasoning': response_content[:200]  # Use first 200 chars as reasoning
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _fallback_sentiment(self, title: str, sapo: str) -> Dict[str, Any]:
        """
        Fallback sentiment analysis using simple rule-based approach.
        
        Args:
            title: Article title
            sapo: Article summary/sapo
            
        Returns:
            Dictionary with sentiment_label, confidence, analysis_reasoning
        """
        text = f"{title} {sapo}".lower()
        
        # Simple keyword-based sentiment
        positive_keywords = ['thành công', 'hiệu quả', 'phát triển', 'tiến bộ', 'tốt', 'đột phá', 'tích cực', 'hỗ trợ', 'khuyến khích']
        negative_keywords = ['thất bại', 'vấn đề', 'chậm', 'khó khăn', 'sai sót', 'giảm', 'lo ngại', 'rủi ro', 'cản trở', 'hạn chế']
        
        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)
        
        if positive_count > negative_count:
            sentiment_label = 'POSITIVE'
            confidence = min(0.7, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment_label = 'NEGATIVE'
            confidence = min(0.7, 0.5 + (negative_count * 0.1))
        else:
            sentiment_label = 'NEUTRAL'
            confidence = 0.5
        
        reasoning = f"Rule-based analysis: {positive_count} positive indicators, {negative_count} negative indicators"
        
        return {
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'analysis_reasoning': reasoning
        }
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from analysis results."""
        total = len(results)
        if total == 0:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'average_confidence': 0.0
            }
        
        positive_count = sum(1 for r in results if r.get('sentiment_label') == 'POSITIVE')
        negative_count = sum(1 for r in results if r.get('sentiment_label') == 'NEGATIVE')
        neutral_count = sum(1 for r in results if r.get('sentiment_label') == 'NEUTRAL')
        
        avg_confidence = sum(r.get('confidence', 0.0) for r in results) / total
        
        return {
            'total': total,
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'average_confidence': round(avg_confidence, 3)
        }
    
    async def _save_summary_csv(self, results: List[Dict[str, Any]]):
        """
        Save sentiment analysis summary to CSV file.
        
        Args:
            results: List of analysis results
        """
        try:
            with open(self.summary_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Article ID',
                    'Title',
                    'Link',
                    'Date',
                    'Sentiment Label',
                    'Confidence',
                    'Analysis Reasoning',
                    'Analyzed At'
                ])
                
                # Write data rows
                for result in results:
                    writer.writerow([
                        result.get('article_id', ''),
                        result.get('title', ''),
                        result.get('link', ''),
                        result.get('date', ''),
                        result.get('sentiment_label', ''),
                        result.get('confidence', 0.0),
                        result.get('analysis_reasoning', ''),
                        result.get('analyzed_at', '')
                    ])
            
            self.logger.info(f"Sentiment summary saved to: {self.summary_csv}")
            
        except Exception as e:
            self.logger.error(f"Failed to save summary CSV: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown sentiment analysis agent."""
        self.logger.info("Shutting down sentiment analysis agent")
        
        if self.llm_client:
            try:
                await self.llm_client.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down LLM client: {e}")
        
        self.logger.info("Sentiment analysis agent shutdown complete")


async def main():
    """Main entry point for standalone sentiment analysis agent."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Load configuration
    try:
        from utils.config import load_config
        config_file = project_root / "config" / "agents.yaml"
        if config_file.exists():
            full_config = load_config(str(config_file))
            agent_config = full_config.get('agents', {}).get('sentiment_analysis_agent', {})
        else:
            agent_config = {}
    except Exception as e:
        print(f"⚠️  Could not load config, using defaults: {e}")
        agent_config = {}
    
    # Initialize agent
    agent = SentimentAnalysisAgent(agent_config)
    
    try:
        # Initialize
        initialized = await agent.initialize()
        if not initialized:
            print("❌ Failed to initialize sentiment analysis agent")
            sys.exit(1)
        
        # Run analysis
        result = await agent.analyze_sentiment()
        
        # Print results
        if result.get('success'):
            print(f"\n✅ Sentiment analysis completed successfully")
            print(f"   Total articles analyzed: {result.get('total_articles_analyzed', 0)}")
            summary = result.get('summary', {})
            print(f"   Positive: {summary.get('positive', 0)}")
            print(f"   Negative: {summary.get('negative', 0)}")
            print(f"   Neutral: {summary.get('neutral', 0)}")
            print(f"   Average confidence: {summary.get('average_confidence', 0.0)}")
            print(f"   Summary CSV: {result.get('summary_csv', '')}")
        else:
            print(f"\n❌ Sentiment analysis failed: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        await agent.shutdown()
        
        # Exit
        sys.exit(0 if result.get('success') else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal error in sentiment analysis agent: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            await agent.shutdown()
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
