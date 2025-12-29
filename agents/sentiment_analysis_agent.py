"""
Sentiment Analysis Agent

Specialized agent for performing sentiment analysis on news articles and legal documents
using LLM integration with Ollama (Llama 3). Reads processed opinion articles and 
legal PDFs, analyzes sentiment with legal context, and generates comprehensive reports.
"""

import sys
import os
# Add project root to sys.path so we can import 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import aiofiles

from utils.logger import get_logger
from utils.config import settings

try:
    from core.llm_client import LLMClient
except ImportError:
    # Fallback for development
    LLMClient = None


class SentimentAnalysisAgent:
    """
    Agent for analyzing sentiment of news articles with legal document context.
    
    This agent:
    - Reads opinion articles from data/production/opinions/*.json
    - Reads legal documents from data/production/pdfs/processed/*.txt
    - Uses LLM (Llama 3) to analyze sentiment with legal context
    - Generates comprehensive sentiment reports
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sentiment analysis agent."""
        self.config = config or {}
        self.name = "sentiment_analysis_agent"
        self.logger = get_logger(self.name)
        
        # Agent-specific configuration
        self.opinions_dir = Path("data/production/opinions")
        self.legal_docs_dir = Path("data/production/pdfs/processed")
        self.output_dir = Path("data/production")
        self.output_file = self.output_dir / "sentiment_report.json"
        
        # LLM configuration
        self.llm_client: Optional[LLMClient] = None
        self.model_name = self.config.get('model', 'llama3:latest')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 2048)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.batch_size = self.config.get('batch_size', 5)
        
        # Sentiment analysis prompt template
        self.sentiment_prompt = """You are an expert sentiment analyst specializing in legal and policy analysis. 

TASK: Analyze the sentiment of the following news article regarding the provided legal document context.

LEGAL DOCUMENT CONTEXT:
{legal_context}

NEWS ARTICLE:
Title: {title}
Summary: {sapo}
Full Content: {content}

ANALYSIS REQUIREMENTS:
1. Read and understand the legal document context
2. Analyze how the news article relates to or discusses the legal context
3. Classify the sentiment as one of: Positive, Negative, or Neutral
4. Provide a confidence score between 0.0 and 1.0
5. Provide brief reasoning explaining your analysis

RESPONSE FORMAT (JSON):
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this sentiment was chosen, considering the legal context"
}}

Analyze the article's tone, attitude, and implications regarding the legal framework or policies mentioned."""

        self.logger.info("Sentiment Analysis Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize sentiment analysis agent."""
        try:
            self.logger.info("Performing sentiment analysis agent initialization")
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if input directories exist
            if not self.opinions_dir.exists():
                self.logger.warning(f"Opinions directory does not exist: {self.opinions_dir}")
            
            if not self.legal_docs_dir.exists():
                self.logger.warning(f"Legal documents directory does not exist: {self.legal_docs_dir}")
                self.legal_docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize LLM client if available
            if LLMClient is not None:
                llm_config = {
                    'base_url': settings.ollama_base_url,
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'timeout': settings.ollama_timeout
                }
                self.llm_client = LLMClient(llm_config)
                
                if await self.llm_client.initialize():
                    self.logger.info(f"LLM client initialized with model: {self.model_name}")
                else:
                    self.logger.error("Failed to initialize LLM client")
            
            self.logger.info("Sentiment analysis agent initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analysis agent: {e}")
            return False
    
    async def analyze_sentiment(self, task_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main sentiment analysis method."""
        try:
            self.logger.info("Starting sentiment analysis task")
            
            # Load opinion articles
            opinion_articles = await self._load_opinion_articles()
            if not opinion_articles:
                return {
                    'success': False,
                    'error': 'No opinion articles found to analyze',
                    'total_articles': 0,
                    'summary': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
            
            # Load legal documents for context
            legal_context = await self._load_legal_context()
            
            # Perform sentiment analysis
            analysis_results = await self._analyze_articles_with_llm(opinion_articles, legal_context)
            
            # Generate report
            report = await self._generate_sentiment_report(analysis_results)
            
            # Save report
            await self._save_report(report)
            
            self.logger.info(f"Sentiment analysis completed: {len(analysis_results)} articles analyzed")
            
            return {
                'success': True,
                'total_articles_analyzed': len(analysis_results),
                'report_file': str(self.output_file),
                'summary': report.get('summary', {}),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis task failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_articles': 0,
                'summary': {'positive': 0, 'negative': 0, 'neutral': 0},
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def _load_opinion_articles(self) -> List[Dict[str, Any]]:
        """Load opinion articles from JSON files with pre-processing filters."""
        try:
            articles = []
            filtered_count = 0
            
            if not self.opinions_dir.exists():
                self.logger.warning(f"Opinions directory not found: {self.opinions_dir}")
                return articles
            
            # Find all JSON files
            json_files = list(self.opinions_dir.glob("*.json"))
            self.logger.info(f"Found {len(json_files)} opinion article files")
            
            for file_path in json_files:
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        article = json.loads(content)
                        
                        # PRE-PROCESSING FILTER: Anti-Spam Check
                        article_link = article.get('link', '')
                        if not article_link:
                            self.logger.debug(f"Skipping article {file_path.name}: No link field")
                            filtered_count += 1
                            continue
                        
                        # Filter out ads and external links
                        if 'eclick.vn' in article_link:
                            self.logger.info(f"⏭️  Skipping Ad/External Link: {article_link[:80]}...")
                            filtered_count += 1
                            continue
                        
                        if 'vnexpress.net' not in article_link:
                            self.logger.info(f"⏭️  Skipping Ad/External Link: {article_link[:80]}...")
                            filtered_count += 1
                            continue
                        
                        # PRE-PROCESSING FILTER: Relevance Check
                        # Get keyword from metadata (query field or matched_keyword)
                        keyword = article.get('query') or article.get('matched_keyword') or article.get('matched_keywords', [])
                        
                        # Handle list of keywords
                        if isinstance(keyword, list):
                            keyword = keyword[0] if keyword else None
                        
                        if keyword:
                            keyword_lower = keyword.lower()
                            title = article.get('title', '').lower()
                            sapo = article.get('sapo', '').lower()
                            
                            # Check if keyword appears in title or sapo
                            if keyword_lower not in title and keyword_lower not in sapo:
                                self.logger.info(f"⏭️  Skipping Irrelevant Article: '{keyword}' not in title/sapo")
                                self.logger.debug(f"   Title: {article.get('title', '')[:60]}...")
                                filtered_count += 1
                                continue
                        
                        # Article passed all filters - add it
                        article['file_path'] = str(file_path)
                        articles.append(article)
                        
                except Exception as e:
                    self.logger.error(f"Failed to load article from {file_path}: {e}")
            
            self.logger.info(f"✅ Successfully loaded {len(articles)} opinion articles (filtered {filtered_count} articles)")
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to load opinion articles: {e}")
            return []
    
    async def _load_legal_context(self) -> str:
        """Load legal documents and create context string."""
        try:
            context_parts = []
            
            if not self.legal_docs_dir.exists():
                self.logger.warning(f"Legal documents directory not found: {self.legal_docs_dir}")
                return "Legal Context: Technology and AI development regulations, data protection laws, and digital transformation policies."
            
            # Find all text files
            text_files = list(self.legal_docs_dir.glob("*.txt"))
            self.logger.info(f"Found {len(text_files)} legal document files")
            
            # Create some sample legal context if no files exist
            if not text_files:
                return "Legal Context: Technology and AI development regulations, data protection laws, and digital transformation policies."
            
            for file_path in text_files[:3]:  # Limit to first 3 documents to avoid context overflow
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        
                        # Limit content length
                        if len(content) > 2000:
                            content = content[:2000] + "...(truncated)"
                        
                        context_parts.append(f"Legal Document ({file_path.name}):\n{content}\n")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load legal document from {file_path}: {e}")
            
            if context_parts:
                legal_context = "\n".join(context_parts)
                self.logger.info(f"Loaded legal context from {len(context_parts)} documents")
                return legal_context
            else:
                self.logger.warning("No legal documents found for context")
                return "Legal Context: Technology and AI development regulations, data protection laws, and digital transformation policies."
                
        except Exception as e:
            self.logger.error(f"Failed to load legal context: {e}")
            return "Error loading legal context."
    
    async def _analyze_articles_with_llm(
        self, 
        articles: List[Dict[str, Any]], 
        legal_context: str
    ) -> List[Dict[str, Any]]:
        """Analyze articles using LLM with legal context."""
        try:
            self.logger.info(f"Starting LLM analysis of {len(articles)} articles")
            
            # Initialize LLM client if not already done
            if self.llm_client is None and LLMClient is not None:
                llm_config = {
                    'base_url': settings.ollama_base_url,
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'timeout': settings.ollama_timeout
                }
                self.llm_client = LLMClient(llm_config)
                
                if not await self.llm_client.initialize():
                    self.logger.error("Failed to initialize LLM client")
                    return await self._fallback_analysis(articles)
            
            results = []
            
            # Process articles in batches to avoid overwhelming LLM
            for i in range(0, len(articles), self.batch_size):
                batch = articles[i:i + self.batch_size]
                batch_results = await self._process_article_batch(batch, legal_context)
                results.extend(batch_results)
                
                # Small delay between batches
                if i + self.batch_size < len(articles):
                    await asyncio.sleep(1)
            
            self.logger.info(f"LLM analysis completed: {len(results)} articles processed")
            return results
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return await self._fallback_analysis(articles)
    
    async def _process_article_batch(
        self, 
        batch: List[Dict[str, Any]], 
        legal_context: str
    ) -> List[Dict[str, Any]]:
        """Process a batch of articles."""
        results = []
        
        for article in batch:
            try:
                # Prepare prompt for this article
                prompt = self.sentiment_prompt.format(
                    legal_context=legal_context,
                    title=article.get('title', ''),
                    sapo=article.get('sapo', ''),
                    content=article.get('content', article.get('sapo', ''))  # Use sapo as content fallback
                )
                
                # Get LLM response
                if self.llm_client is not None:
                    response = await self.llm_client._generate_response(prompt)
                    
                    if response.error:
                        self.logger.warning(f"LLM error for article {article.get('id')}: {response.error}")
                        result = await self._fallback_single_article(article)
                    else:
                        # Parse LLM response
                        result = await self._parse_llm_response(response.content, article)
                else:
                    result = await self._fallback_single_article(article)
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze article {article.get('id')}: {e}")
                # Add fallback result
                fallback_result = await self._fallback_single_article(article)
                fallback_result['analysis_error'] = str(e)
                results.append(fallback_result)
        
        return results
    
    async def _parse_llm_response(self, response_content: str, article: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured result."""
        try:
            # Try to extract JSON from response
            json_pattern = r'\{[^{}]*"sentiment"[^{}]*\}'
            matches = re.findall(json_pattern, response_content, re.DOTALL)
            
            if matches:
                json_str = matches[0]
                parsed = json.loads(json_str)
                
                return {
                    'article_id': article.get('id'),
                    'title': article.get('title'),
                    'source': article.get('source'),
                    'date': article.get('date'),
                    'link': article.get('link'),
                    'sentiment': parsed.get('sentiment', 'neutral').lower(),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'reasoning': parsed.get('reasoning', 'No reasoning provided'),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'llm',
                    'model_used': self.model_name
                }
            else:
                # Fallback if JSON parsing fails
                self.logger.warning(f"Failed to parse JSON from LLM response for article {article.get('id')}")
                return await self._fallback_single_article(article)
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response for article {article.get('id')}: {e}")
            return await self._fallback_single_article(article)
    
    async def _fallback_analysis(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback analysis when LLM is not available."""
        self.logger.warning("Using fallback sentiment analysis (rule-based)")
        results = []
        
        for article in articles:
            result = await self._fallback_single_article(article)
            results.append(result)
        
        return results
    
    async def _fallback_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback sentiment analysis for a single article."""
        # Simple rule-based sentiment analysis
        title = article.get('title', '').lower()
        sapo = article.get('sapo', '').lower()
        text = f"{title} {sapo}"
        
        # Simple sentiment word lists (can be expanded)
        positive_words = ['thành công', 'hiệu quả', 'phát triển', 'tiến bộ', 'tốt', 'đột phá', 'tích cực', 'hiệu quả']
        negative_words = ['thất bại', 'vấn đề', 'chậm', 'khó khăn', 'sai sót', 'giảm', 'lo ngại', 'rủi ro']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.8, positive_count / (positive_count + negative_count + 1))
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.8, negative_count / (positive_count + negative_count + 1))
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        reasoning = f"Rule-based analysis: {positive_count} positive words, {negative_count} negative words found"
        
        return {
            'article_id': article.get('id'),
            'title': article.get('title'),
            'source': article.get('source'),
            'date': article.get('date'),
            'link': article.get('link'),
            'sentiment': sentiment,
            'confidence': confidence,
            'reasoning': reasoning,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_method': 'rule_based_fallback',
            'model_used': None
        }
    
    async def _generate_sentiment_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive sentiment report."""
        try:
            # Calculate summary statistics
            total_articles = len(analysis_results)
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_confidence = 0.0
            method_counts = {'llm': 0, 'rule_based_fallback': 0}
            
            for result in analysis_results:
                sentiment = result.get('sentiment', 'neutral')
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
                
                total_confidence += result.get('confidence', 0.0)
                method = result.get('analysis_method', 'unknown')
                if method in method_counts:
                    method_counts[method] += 1
            
            avg_confidence = total_confidence / total_articles if total_articles > 0 else 0.0
            
            # Generate report
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_articles': total_articles,
                'summary': {
                    'positive': sentiment_counts['positive'],
                    'negative': sentiment_counts['negative'],
                    'neutral': sentiment_counts['neutral'],
                    'average_confidence': round(avg_confidence, 3),
                    'analysis_methods': method_counts
                },
                'details': analysis_results,
                'metadata': {
                    'agent': self.name,
                    'model_used': self.model_name,
                    'confidence_threshold': self.confidence_threshold,
                    'batch_size': self.batch_size,
                    'legal_context_available': True
                }
            }
            
            self.logger.info(f"Generated sentiment report: {sentiment_counts}")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate sentiment report: {e}")
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_articles': 0,
                'summary': {'positive': 0, 'negative': 0, 'neutral': 0},
                'details': [],
                'error': str(e)
            }
    
    async def _save_report(self, report: Dict[str, Any]) -> bool:
        """Save sentiment report to JSON file."""
        try:
            # Create backup of existing report if it exists
            if self.output_file.exists():
                backup_file = self.output_file.with_suffix('.json.bak')
                import os
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(self.output_file, backup_file)
                else:
                    os.rename(self.output_file, backup_file)
            
            # Save new report
            async with aiofiles.open(self.output_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(report, indent=2, ensure_ascii=False))
            
            self.logger.info(f"Sentiment report saved to: {self.output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save sentiment report: {e}")
            return False
    
    async def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status."""
        try:
            # Check if output file exists and load it
            if self.output_file.exists():
                async with aiofiles.open(self.output_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    report = json.loads(content)
                    
                return {
                    'last_analysis': report.get('analysis_timestamp'),
                    'total_articles_analyzed': report.get('total_articles', 0),
                    'summary': report.get('summary', {}),
                    'report_file': str(self.output_file)
                }
            else:
                return {
                    'last_analysis': None,
                    'total_articles_analyzed': 0,
                    'summary': {},
                    'report_file': str(self.output_file),
                    'status': 'No analysis performed yet'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get analysis status: {e}")
            return {
                'error': str(e),
                'status': 'Error retrieving status'
            }
    
    async def analyze_single_article(self, article_id: str) -> Dict[str, Any]:
        """Analyze a single article by ID."""
        try:
            # Load article file
            article_file = self.opinions_dir / f"{article_id}.json"
            
            if not article_file.exists():
                return {
                    'success': False,
                    'error': f'Article file not found: {article_file}'
                }
            
            async with aiofiles.open(article_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                article = json.loads(content)
            
            # Load legal context
            legal_context = await self._load_legal_context()
            
            # Analyze single article
            results = await self._analyze_articles_with_llm([article], legal_context)
            
            if results:
                return {
                    'success': True,
                    'analysis': results[0]
                }
            else:
                return {
                    'success': False,
                    'error': 'Analysis failed'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze single article {article_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown sentiment analysis agent."""
        self.logger.info("Shutting down sentiment analysis agent")
        
        # Shutdown LLM client if it exists
        if self.llm_client is not None:
            try:
                await self.llm_client.shutdown()
                self.logger.info("LLM client shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down LLM client: {e}")
        
        self.logger.info("Sentiment analysis agent shutdown complete")


# Utility functions for standalone usage
async def run_sentiment_analysis(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run sentiment analysis independently.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Analysis result dictionary
    """
    agent = SentimentAnalysisAgent(config)
    
    try:
        # Initialize agent
        await agent.initialize()
        
        # Run analysis task
        result = await agent.analyze_sentiment()
        return result
        
    finally:
        # Cleanup
        await agent.shutdown()


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'model': 'llama3:latest',
            'temperature': 0.3,
            'confidence_threshold': 0.7,
            'batch_size': 3
        }
        
        result = await run_sentiment_analysis(config)
        print(f"Analysis result: {result}")
    
    asyncio.run(main())