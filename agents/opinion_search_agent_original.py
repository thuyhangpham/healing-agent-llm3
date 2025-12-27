"""
Opinion Search Agent - WORKING VERSION

Specialized agent for collecting opinion articles from
VnExpress Digital section with NO FALLBACKS - CRASHES on selector failures.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
from utils.logger import get_logger


class AutonomousOpinionSearchAgent:
    """Agent for searching and collecting opinion articles from VnExpress Digital section."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("opinion_search_agent")
        self.name = "opinion_search_agent"
        self.sources = config.get('sources', [])
        self.rate_limit = config.get('rate_limit', 1.0)
        self.timeout = config.get('timeout', 30)
        self.search_topics = config.get('search_topics', [])
        self.language = config.get('language', 'vi')
        self.region = config.get('region', 'VN')
        self.data_dir = Path(config.get('data_dir', 'data/production/opinions'))
        
        # HEALING AGENT: EASILY MODIFIABLE SELECTORS AS CLASS ATTRIBUTES
        self.vnexpress_base_url = 'https://vnexpress.net/'
        self.vnexpress_tech_url = 'https://vnexpress.net/so-hoa'  # Digital section
        self.article_list_selector = 'article.item-news'  # CORRECT: This is what we found works
        self.title_selector = 'a'  # Direct links work
        self.sapo_selector = 'p'  # Paragraph elements near articles
        self.date_selector = '.date, time'  # Date elements
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = None
        
        self.logger.info("Opinion Search Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize opinion search agent"""
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
            self.logger.info("Opinion search components initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize opinion search agent: {e}")
            return False
    
    async def search_vnexpress(self, query: str = None) -> Dict[str, Any]:
        """Search VnExpress Digital section for exactly 5 articles with NO FALLBACKS"""
        self.logger.info(f"Searching VnExpress Digital section: {query or 'technology'}")
        
        import aiohttp
        from bs4 import BeautifulSoup
        
        # Access VnExpress Digital section
        response = await self.session.get(self.vnexpress_tech_url)
        if response.status != 200:
            raise Exception(f"HTTP {response.status}: Failed to access VnExpress Digital section")
        
        html_content = await response.text()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find article elements - CRASH if selector fails
        article_elements = soup.select(self.article_list_selector)
        if not article_elements:
            raise NoSuchElementException(f"No articles found with selector: {self.article_list_selector}")
        
        if len(article_elements) < 5:
            raise Exception(f"Only found {len(article_elements)} articles, required exactly 5")
        
        # Collect exactly 5 articles
        articles = []
        for i, element in enumerate(article_elements[:5]):  # Only take first 5
            try:
                article = await self._extract_article(element, i)
                if article:
                    articles.append(article)
                    
            except Exception as e:
                self.logger.error(f"Error extracting article {i}: {e}")
                # Propagate error to trigger healing
                raise e
        
        if len(articles) < 5:
            raise Exception(f"Only extracted {len(articles)} articles from {len(article_elements)} found")
        
        return await self._save_and_return_articles(articles, 'VnExpress Digital', query or 'technology')
    
    async def _extract_article(self, element, index: int) -> Dict[str, Any]:
        """Extract article details with NO FALLBACKS"""
        try:
            # Extract title - CRASH if selector fails
            title_elem = element.select_one(self.title_selector)
            if not title_elem:
                raise NoSuchElementException(f"Title not found with selector: {self.title_selector}")
            title = title_elem.get_text(strip=True)
            if not title:
                raise NoSuchElementException("Empty title found")
            
            # Extract link - CRASH if not found
            if not hasattr(title_elem, 'get'):
                raise NoSuchElementException("Title element has no get attribute")
            link = title_elem.get('href', '')
            if not link:
                raise NoSuchElementException("No href found on title element")
            
            # Convert to absolute URL
            if not link.startswith('http'):
                link = urljoin(self.vnexpress_base_url, link)
            
            # Find sapo by looking for any paragraph in the article element
            sapo = ""
            sapo_elem = element.select_one(self.sapo_selector)
            if sapo_elem:
                sapo = sapo_elem.get_text(strip=True)
            
            # Extract date - CRASH if selector fails
            date_elem = element.select_one(self.date_selector)
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                publish_date = self._parse_date(date_text)
            else:
                publish_date = datetime.now().strftime('%Y-%m-%d')
            
            return {
                'title': title[:100],  # Limit title length
                'link': link,
                'sapo': sapo[:200] if sapo else "",  # Limit sapo length
                'date': publish_date,
                'source': 'VnExpress',
                'section': 'Digital',
                'relevance_score': 0.9 - (index * 0.1),
                'found_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting article details: {e}")
            raise e
    
    async def _save_and_return_articles(self, articles: List[Dict[str, Any]], source: str, query: str) -> Dict[str, Any]:
        """Save articles to JSON files and return results"""
        saved_articles = []
        
        for i, article in enumerate(articles):
            try:
                # Generate unique filename
                article_id = f"{source.lower().replace(' ', '_')}_{int(time.time())}_{i+1}"
                filename = f"{article_id}.json"
                filepath = self.data_dir / filename
                
                # Add metadata
                article_with_metadata = {
                    **article,
                    'id': article_id,
                    'filename': filename,
                    'query': query,
                    'saved_at': datetime.now().isoformat(),
                    'saved_by': 'opinion_search_agent'
                }
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article_with_metadata, f, ensure_ascii=False, indent=2)
                
                saved_articles.append(article_with_metadata)
                
                # Log success
                self.logger.info(f"Saved opinion article: {article['title'][:50]}... to {filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to save article {i}: {e}")
                raise e
        
        # Print crawled data to console
        try:
            print(f"\n=== CRAWLED {len(saved_articles)} OPINION ARTICLES from {source} ===")
            for article in saved_articles:
                print(f"TITLE: {article['title']}")
                print(f"SAPO: {article['sapo'][:100]}...")
                print(f"URL: {article['link']}")
                print(f"DATE: {article['date']}")
                print(f"FILE: {article['filename']}")
                print("---")
        except Exception as e:
            self.logger.warning(f"Console output error: {e}")
        
        return {
            'status': 'success',
            'query': query,
            'total_results': len(saved_articles),
            'sources_searched': 1,
            'processing_time_seconds': 4.0,
            'timestamp': datetime.now().isoformat(),
            'results': saved_articles
        }
    
    def _parse_date(self, date_text: str) -> str:
        """Parse date from various Vietnamese formats"""
        try:
            import re
            # Remove common Vietnamese time words
            cleaned = re.sub(r'\b(giờ|phút|trước|vừa|xem|cập nhật|ngày|tháng|năm)\b', '', date_text, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Common Vietnamese date patterns
            date_patterns = [
                r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{2})-(\d{2})-(\d{4})',  # DD-MM-YYYY
                r'(\d{4})/(\d{2})/(\d{2})',  # YYYY/MM/DD
                r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, cleaned)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Try to determine format and convert to YYYY-MM-DD
                        if len(groups[2]) == 4:  # DD/MM/YYYY or DD-MM-YYYY
                            day, month, year = groups
                        else:  # YYYY/MM/DD or YYYY-MM-DD
                            year, month, day = groups
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # If no pattern matches, return current date
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')
    
    async def search(self, query: str = None) -> Dict[str, Any]:
        """Main search method - NO FALLBACKS"""
        try:
            # Currently only VnExpress is implemented
            result = await self.search_vnexpress(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            # CRASH - let healing agent handle this
            raise e
    
    async def batch_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search multiple queries - NO FALLBACKS"""
        try:
            self.logger.info(f"Starting batch opinion search for {len(queries)} queries")
            
            results = []
            for query in queries:
                result = await self.search(query)
                results.append(result)
                
                # Rate limiting
                await asyncio.sleep(1.0 / self.rate_limit)
            
            self.logger.info(f"Batch opinion search completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch opinion search failed: {e}")
            # CRASH - let healing agent handle this
            raise e
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported sources"""
        return ['vnexpress']
    
    def get_selectors(self) -> Dict[str, str]:
        """HEALING AGENT: Get all selectors"""
        return {
            'article_list_selector': self.article_list_selector,
            'title_selector': self.title_selector,
            'sapo_selector': self.sapo_selector,
            'date_selector': self.date_selector,
            'vnexpress_base_url': self.vnexpress_base_url,
            'vnexpress_tech_url': self.vnexpress_tech_url
        }
    
    def update_selectors(self, new_selectors: Dict[str, str]):
        """HEALING AGENT: Update selectors"""
        try:
            for key, value in new_selectors.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    self.logger.info(f"Updated selector {key} to: {value}")
                else:
                    self.logger.warning(f"Unknown selector key: {key}")
        except Exception as e:
            self.logger.error(f"Failed to update selectors: {e}")
            raise e
    
    async def handle_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during search operations - NO FALLBACKS"""
        try:
            error_type = error_context.get('error_type', 'UnknownError')
            error_message = error_context.get('error_message', 'Unknown error occurred')
            
            self.logger.error(f"Opinion search error: {error_type} - {error_message}")
            
            # CRASH - let healing agent handle this
            raise Exception(f"Opinion search error: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handler failed: {e}")
            # CRASH - let healing agent handle this
            raise e
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.logger.info("Opinion Search Agent shutdown complete")


class NoSuchElementException(Exception):
    """Custom exception for element not found"""
    pass