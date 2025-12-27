"""
Opinion Search Agent - Fixed Version

Specialized agent for collecting opinion articles from VnExpress Digital section
with robust CSS selector logic and error handling.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse

from agents.base_agent import BaseAgent
from utils.logger import get_logger


class OpinionSearchAgent(BaseAgent):
    """Agent for searching and collecting opinion articles from VnExpress Digital section."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize opinion search agent."""
        super().__init__("opinion_search_agent", config)
        
        # Configuration
        self.sources = self.config.get('sources', ['VnExpress Digital'])
        self.rate_limit = self.config.get('rate_limit', 1.0)
        self.timeout = self.config.get('timeout', 30)
        self.search_topics = self.config.get('search_topics', ['công nghệ số', 'AI'])
        self.language = self.config.get('language', 'vi')
        self.region = self.config.get('region', 'VN')
        self.data_dir = Path(self.config.get('data_dir', 'data/production/opinions'))
        
        # VnExpress URLs
        self.vnexpress_base_url = 'https://vnexpress.net'
        self.vnexpress_tech_url = 'https://vnexpress.net/so-hoa'
        
        # HEALING AGENT: UPDATED SELECTORS
        # Fixed selectors based on actual VnExpress structure
        self.article_list_selector = '.list-news-sub .item'  # Main article list
        self.title_selector = 'a'  # Article titles
        self.sapo_selector = '.description'  # Article descriptions (changed from 'p')
        self.date_selector = '.date, time'  # Article dates
        self.link_selector = 'a'  # Direct links (fallback)
        
        # Additional selectors for more robust extraction
        self.category_selector = '.breadcrumb-cate'  # Category breadcrumbs
        self.meta_info_selector = '.meta-info'  # Meta information
        
        # CSS class names for more specific targeting
        self.css_title_class = 'title-news'
        self.css_sapo_class = 'description-news'
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Opinion Search Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize opinion search agent."""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br'
                }
            )
            
            self.logger.info("Opinion search agent components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize opinion search agent: {e}")
            return False
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process opinion search task."""
        task_type = task.get('type', 'search')
        query = task.get('query', 'công nghệ số')
        
        self.logger.info(f"Processing opinion search task: {task_type} - {query}")
        
        if task_type == 'search':
            return await self._search_opinions(query)
        elif task_type == 'context_aware_search':
            return await self._context_aware_search(query)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _context_aware_search(self, base_query: str) -> Dict[str, Any]:
        """Search with legal context awareness."""
        try:
            # Load legal context keywords
            legal_keywords = await self._load_legal_keywords()
            
            # Combine with search query
            search_query = f"{base_query} {legal_keywords}"
            
            self.logger.info(f"Context-aware search: {search_query}")
            
            # Perform search
            articles = await self._search_vnexpress(search_query)
            
            # Filter articles based on legal context relevance
            relevant_articles = []
            for article in articles:
                if self._is_legal_context_relevant(article, legal_keywords):
                    relevant_articles.append(article)
            
            # Save results
            await self._save_articles(relevant_articles, f"context_aware_{base_query}")
            
            return {
                'success': True,
                'task_type': 'context_aware_search',
                'query': base_query,
                'legal_keywords': legal_keywords,
                'total_found': len(articles),
                'relevant_count': len(relevant_articles),
                'articles_saved': len(relevant_articles),
                'search_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Context-aware search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_type': 'context_aware_search'
            }
    
    async def _search_opinions(self, query: str) -> Dict[str, Any]:
        """Search VnExpress for opinion articles."""
        try:
            # Construct search URL
            encoded_query = urljoin(self.vnexpress_tech_url, f"?q={query}&latest=1")
            
            self.logger.info(f"Searching VnExpress: {encoded_query}")
            
            # Make request
            response = await self.session.get(encoded_query)
            
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {response.reason}")
            
            html_content = await response.text()
            
            # Wait for page to fully render
            await asyncio.sleep(5)
            
            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract articles with multiple selector strategies
            articles = []
            
            # Strategy 1: Try main article list selector
            article_elements = soup.select(self.article_list_selector)
            if article_elements:
                self.logger.info(f"Found {len(article_elements)} articles with main selector")
                for i, element in enumerate(article_elements[:5]):  # Limit to 5 articles
                    article = await self._extract_article(element, i)
                    if article:
                        articles.append(article)
            
            # Strategy 2: Fallback to individual article elements
            if len(articles) < 5:
                self.logger.info("Trying fallback article extraction")
                article_elements = soup.select('article')
                for i, element in enumerate(article_elements[:5]):  # Limit to 5 articles
                    article = await self._extract_article(element, i)
                    if article:
                        articles.append(article)
            
            # Strategy 3: Generic fallback to any link elements
            if len(articles) < 5:
                self.logger.info("Trying generic link extraction")
                links = soup.find_all('a', href=True)
                for i, link in enumerate(links[:5]):  # Limit to 5 links
                    title = link.get_text(strip=True) or f"Article {i+1}"
                    href = link.get('href')
                    if href and not href.startswith('javascript:'):
                        article = {
                            'title': title,
                            'link': href,
                            'source': 'VnExpress Digital',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'id': f"vnexpress_tech_{int(time.time())}_{i}"
                        }
                        articles.append(article)
            
            # Save articles
            await self._save_articles(articles, f"search_{query}")
            
            return {
                'success': True,
                'task_type': 'search',
                'query': query,
                'articles_found': len(articles),
                'articles_saved': len(articles),
                'search_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_type': 'search'
            }
    
    async def _extract_article(self, element, index: int) -> Optional[Dict[str, Any]]:
        """Extract article information from HTML element."""
        try:
            # Get title with multiple fallback strategies
            title = self._get_element_text(element, [
                ('a', self.title_selector),
                ('h1', self.title_selector),
                ('h2', self.title_selector),
                ('h3', self.title_selector)
                ('[class*="title-news"]', None)  # Class-based selector
            ])
            
            if not title:
                title = f"Article {index + 1}"
            
            # Get sapo/description with multiple fallback strategies
            sapo = self._get_element_text(element, [
                ('p', self.sapo_selector),
                ('div.description', None),  # Fallback div class
                ('div.meta-info', None),  # Meta info div
                ('span', None)  # Generic span
            ])
            
            if not sapo:
                # Try to get first paragraph as fallback
                p_element = element.find('p')
                if p_element:
                    sapo = p_element.get_text(strip=True)
            
            # Get date
            date = self._get_element_text(element, [
                ('time', self.date_selector),
                ('span.date', None),  # Fallback date span
                ('div.date', None)  # Fallback date div
            ])
            
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # Get link
            link = self._get_element_text(element, [
                ('a', self.link_selector),
                ('h1 a', None),  # Title link
                ('h2 a', None),  # Title link
                ('h3 a', None),  # Title link
            ])
            
            if not link:
                # Fallback: construct from current URL
                base_url = getattr(self, 'vnexpress_base_url', 'https://vnexpress.net/')
                link_element = element.find('a', href=True)
                if link_element:
                    href = link_element.get('href')
                    if href and not href.startswith('javascript:'):
                        if href.startswith('/'):
                            link = urljoin(base_url, href[1:])
                        else:
                            link = urljoin(base_url, href)
            
            # Get image
            image = self._get_element_text(element, [
                ('img', None),  # Direct image
                ('picture source img', None),  # Picture source image
                ('div.img', None),  # Div container image
            ])
            
            return {
                'title': title.strip() if title else 'No Title',
                'sapo': sapo.strip() if sapo else 'No Description',
                'date': date,
                'link': link,
                'image': image,
                'source': 'VnExpress Digital',
                'id': f"vnexpress_tech_{int(time.time())}_{index}",
                'found_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract article {index}: {e}")
            return None
    
    def _get_element_text(self, element, selectors: List[tuple]) -> Optional[str]:
        """Get text from element using multiple selector strategies."""
        for selector, attribute in selectors:
            if attribute:  # Class-based selector
                found_elements = element.select(f'[class*="{selector}"]')
            else:  # Tag-based selector
                found_elements = element.select(selector)
            
            if found_elements:
                return found_elements[0].get_text(strip=True)
        
        return None
    
    def _is_legal_context_relevant(self, article: Dict[str, Any], legal_keywords: str) -> bool:
        """Check if article is relevant to legal context."""
        if not article:
            return False
        
        title = article.get('title', '').lower()
        sapo = article.get('sapo', '').lower()
        legal_keywords_lower = legal_keywords.lower()
        
        # Check if any legal keywords are present
        for keyword in legal_keywords_lower.split():
            if keyword in title or keyword in sapo:
                return True
        
        return False
    
    async def _load_legal_keywords(self) -> str:
        """Load legal keywords from JSON files."""
        try:
            # Search for law JSON files
            laws_dir = Path("data/production/laws")
            if laws_dir.exists():
                law_files = list(laws_dir.glob("*.json"))
                
                for law_file in law_files:
                    try:
                        with open(law_file, 'r', encoding='utf-8') as f:
                            law_data = json.load(f)
                            
                        # Get search keywords from law data
                        keywords = law_data.get('search_keywords', [])
                        if keywords:
                            return ' '.join(keywords)
                            
                    except Exception:
                        continue
            
            return "AI công nghệ kỹ thuật số"  # Default fallback
            
        except Exception as e:
            self.logger.error(f"Failed to load legal keywords: {e}")
            return "trí tuệ nhân tạo"  # Generic fallback
    
    async def _save_articles(self, articles: List[Dict[str, Any]], prefix: str):
        """Save articles to JSON files."""
        try:
            for i, article in enumerate(articles):
                # Generate ID
                article_id = f"{prefix}_{i + 1}"
                article['id'] = article_id
                article['filename'] = f"{article_id}.json"
                
                # Save individual article file
                file_path = self.data_dir / article['filename']
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)
                
                # Add delay to avoid overwhelming server
                if i % 3 == 0:
                    await asyncio.sleep(2)
            
            self.logger.info(f"Saved {len(articles)} articles with prefix: {prefix}")
            
        except Exception as e:
            self.logger.error(f"Failed to save articles: {e}")
    
    async def _shutdown(self):
        """Cleanup resources."""
        if hasattr(self, 'session') and self.session:
            await self.session.close()


# Utility function for standalone usage
async def run_opinion_search(query: str = "công nghệ số", config: Optional[Dict[str, Any]] = None):
    """Run opinion search independently."""
    agent = OpinionSearchAgent(config)
    
    try:
        if await agent.initialize():
            task = {
                'type': 'context_aware_search',
                'query': query,
                'id': 'main_search'
            }
            
            result = await agent._process_task(task)
            
            print(f"Search Results:")
            print(f"Query: {query}")
            print(f"Articles Found: {result.get('articles_found', 0)}")
            print(f"Articles Saved: {result.get('articles_saved', 0)}")
            
            return result
            
    except Exception as e:
        print(f"Search failed: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        if 'agent' in locals():
            await agent._shutdown()


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_opinion_search())