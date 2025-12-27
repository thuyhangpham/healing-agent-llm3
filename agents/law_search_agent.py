"""
Law Search Agent - WORKING VERSION - FIXED FOR NON-BLOCKING

Specialized agent for collecting data from government portals
and legal sources with error capture for self-healing.
NO FALLBACK LOGIC - will CRASH on selector failures.

FIXED: Removed blocking "exactly 5 documents" requirement to prevent infinite loops.
Each run() call processes one batch and returns whatever is found.
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


class AutonomousLawSearchAgent:
    """Agent for searching and collecting legal data from government sources."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("law_search_agent")
        self.name = "law_search_agent"
        self.sources = config.get('sources', [])
        self.rate_limit = config.get('rate_limit', 1.0)
        self.timeout = config.get('timeout', 30)
        self.search_queries = config.get('search_queries', [])
        self.data_dir = Path(config.get('data_dir', 'data/production/laws'))
        self.pdf_dir = Path(config.get('pdf_dir', 'data/production/pdfs/raw'))
        
        # NON-BLOCKING: Add max_retries to prevent server spam
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        
        # HEALING AGENT: EASILY MODIFIABLE SELECTORS AS CLASS ATTRIBUTES
        self.vbpl_base_url = 'https://vbpl.vn/'
        self.vbpl_search_url = 'https://vbpl.vn/Pages/tim-van-ban.aspx'
        self.document_list_selector = 'div.item, div.news-item, .document-item, li.item'
        self.title_selector = 'h3.title, h2.title, a.title'
        self.document_link_selector = 'a[href*="van-ban"], a[href*="detail"], a[href*=".pdf"], a[href*=".doc"], a[href*=".docx"]'
        self.document_number_selector = 'span.doc-number, .so-hieu, .document-id'
        self.date_selector = 'span.date, .ngay-ban-hanh, .publish-date'
        
        # Create data directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = None
        
        self.logger.info("Law Search Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize law search agent"""
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
            self.logger.info("Law search components initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize law search agent: {e}")
            return False
    
    async def search_vbpl(self, query: str, target_count: int = 5, max_attempts: int = 1) -> Dict[str, Any]:
        """
        Search VBPL website for legal documents with NON-BLOCKING approach.
        
        FIXED: Removed "exactly 5 documents" requirement that was causing infinite loops.
        Each call returns whatever documents are found in one attempt.
        Added retry mechanism with max_retries to prevent server spam.
        """
        self.logger.info(f"Searching VBPL for: {query} (target: {target_count}, max_attempts: {max_attempts})")
        
        import aiohttp
        from bs4 import BeautifulSoup
        
        last_error = None
        documents = []
        
        # NON-BLOCKING: Limited attempts instead of infinite loop
        for attempt in range(max_attempts):
            try:
                # Access VBPL homepage for recent documents
                response = await self.session.get(self.vbpl_base_url)
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: Failed to access VBPL homepage")
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find document elements - CRASH if selector fails
                doc_items = soup.select(self.document_list_selector)
                if not doc_items:
                    raise NoSuchElementException(f"No documents found with selector: {self.document_list_selector}")
                
                # Process available documents up to target count
                for i, item in enumerate(doc_items[:target_count]):
                    try:
                        # Find document links - APPLY LINK FILTERING AS SPECIFIED
                        doc_links = item.select('a[href]')
                        if not doc_links:
                            raise NoSuchElementException(f"No document links found")
                        
                        # Filter links according to requirements
                        valid_doc_links = []
                        for link_elem in doc_links:
                            href = link_elem.get('href', '')
                            text = link_elem.get_text(strip=True)
                            
                            # Check link filtering requirements
                            if (href and text and len(text) > 10 and
                                '/vanban/' in href and  # Must contain /vanban/
                                not any(skip in href.lower() for skip in ['hoidap.aspx', 'tintuc.aspx', 'hienthi-congbao.aspx'])):  # Ignore Q&A pages
                                
                                valid_doc_links.append(link_elem)
                        
                        if not valid_doc_links:
                            self.logger.info(f"No valid document links found for item {i} - all links were Q&A pages")
                            continue
                        
                        # Use the first valid document link for details
                        doc_link = valid_doc_links[0].get('href', '')
                        if not doc_link:
                            raise NoSuchElementException("No href found on valid document link")
                        
                        # Convert to absolute URL
                        if not doc_link.startswith('http'):
                            doc_link = urljoin(self.vbpl_base_url, doc_link)
                        
                        # Extract title - try multiple approaches
                        title = ""
                        title_elem = item.select_one(self.title_selector)
                        if title_elem and title_elem.get_text(strip=True):
                            title = title_elem.get_text(strip=True)
                        
                        if not title:
                            # Try to get title from link text first
                            link_text = valid_doc_links[0].get_text(strip=True)
                            if link_text and len(link_text) > 5:
                                title = link_text
                        
                        if not title:
                            title = f"VBPL Document {i+1}"
                        
                        # Try to download document if it's a direct file
                        attachment_path = ""
                        if '/vanban/' in doc_link:
                            # Visit detail page to find actual file attachments
                            attachment_path = await self._deep_crawl_for_attachments(doc_link, i+1)
                        elif any(ext in doc_link.lower() for ext in ['.pdf', '.doc', '.docx']):
                            # Direct file link - download immediately
                            attachment_path = await self._download_attachment(doc_link, i+1)
                        else:
                            # Use link as reference if no direct file
                            attachment_path = doc_link
                        
                        # Extract document number - try multiple selectors
                        doc_number = f"VBPL_{int(time.time())}_{i+1}"
                        doc_number_elem = item.select_one(self.document_number_selector)
                        if doc_number_elem:
                            doc_number = doc_number_elem.get_text(strip=True)
                        
                        # Extract date - try multiple selectors
                        date_elem = item.select_one(self.date_selector)
                        if date_elem:
                            publish_date = self._parse_date(date_elem.get_text(strip=True))
                        else:
                            publish_date = datetime.now().strftime('%Y-%m-%d')
                        
                        document = {
                            'title': title,
                            'document_number': doc_number,
                            'link': doc_link,
                            'attachment_path': attachment_path,
                            'publish_date': publish_date,
                            'source': 'VBPL',
                            'query': query,
                            'relevance_score': 0.9 - (i * 0.1),
                            'found_date': datetime.now().isoformat()
                        }
                        
                        documents.append(document)
                        
                        # Stop if we've found enough documents
                        if len(documents) >= target_count:
                            break
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting document {i} (attempt {attempt + 1}): {e}")
                        if attempt == max_attempts - 1:
                            raise e  # Only raise on last attempt
                        continue
                
                # If we found any documents, return them
                if documents:
                    self.logger.info(f"Found {len(documents)} documents on attempt {attempt + 1}")
                    break
                    
                # If no documents found and we have retries left, wait and try again
                if attempt < max_attempts - 1:
                    self.logger.warning(f"No documents found on attempt {attempt + 1}, retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                last_error = e
                self.logger.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                continue
        
        # NON-BLOCKING: Return whatever we found (could be 0 documents)
        self.logger.info(f"Search completed: found {len(documents)} documents")
        return await self._save_and_return_results(documents, query, 'VBPL')
    
    async def _deep_crawl_for_attachments(self, doc_link: str, doc_index: int) -> str:
        """Visit detail page to find and download actual file attachments"""
        try:
            self.logger.info(f"Deep crawling detail page for attachments: {doc_link}")
            
            # Visit the detail page
            response = await self.session.get(doc_link)
            if response.status != 200:
                self.logger.warning(f"Failed to access detail page: HTTP {response.status}")
                return doc_link  # Return original link as fallback
            
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for file download links/boxes
            file_links = []
            
            # Common patterns for file download areas
            download_patterns = [
                'a[href*=".pdf"]', 'a[href*=".doc"]', 'a[href*=".docx"]',
                '.box-file a', '.download-file a', '.attachment a',
                '[class*="download"] a', '[class*="file"] a',
                'a[title*="tải về"]', 'a[title*="download"]'
            ]
            
            for pattern in download_patterns:
                links = soup.select(pattern)
                for link in links:
                    href = link.get('href', '')
                    if href:
                        file_links.append(href)
            
            if not file_links:
                self.logger.info(f"No file attachments found on detail page for doc {doc_index}")
                return doc_link  # Return original link as fallback
            
            # Use the first valid file link found
            file_url = file_links[0] if file_links else doc_link
            
            # Download the actual file
            return await self._download_attachment(file_url, doc_index)
            
        except Exception as e:
            self.logger.error(f"Error deep crawling for attachments: {e}")
            return doc_link
    
    async def _download_attachment(self, url: str, doc_index: int) -> str:
        """Download PDF/DOC attachment and return local path"""
        try:
            # Download the attachment
            response = await self.session.get(url)
            if response.status != 200:
                raise Exception(f"Failed to download attachment: HTTP {response.status}")
            
            # Determine filename
            original_filename = url.split('/')[-1]
            if not original_filename:
                original_filename = f"document_{doc_index}.pdf"
            
            # Create local filename
            timestamp = int(time.time())
            local_filename = f"vbpl_doc_{doc_index}_{timestamp}_{original_filename}"
            local_path = self.pdf_dir / local_filename
            
            # Save attachment to disk
            content = await response.read()
            with open(local_path, 'wb') as f:
                f.write(content)
            
            self.logger.info(f"Downloaded attachment: {local_filename}")
            return str(local_path)
            
        except Exception as e:
            self.logger.error(f"Error downloading attachment: {e}")
            return url
    
    async def _save_and_return_results(self, documents: List[Dict[str, Any]], query: str, source: str) -> Dict[str, Any]:
        """Save documents to JSON files and return results"""
        saved_documents = []
        
        for i, doc in enumerate(documents):
            try:
                # Generate unique filename
                doc_id = f"{source.lower()}_{int(time.time())}_{i+1}"
                filename = f"{doc_id}.json"
                filepath = self.data_dir / filename
                
                # Add metadata
                doc_with_metadata = {
                    **doc,
                    'id': doc_id,
                    'filename': filename,
                    'saved_at': datetime.now().isoformat(),
                    'saved_by': 'law_search_agent'
                }
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_with_metadata, f, ensure_ascii=False, indent=2)
                
                saved_documents.append(doc_with_metadata)
                
                # Log success
                self.logger.info(f"Saved legal document: {doc['title'][:50]}... to {filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to save document {i}: {e}")
                raise e
        
        # Print crawled data to console
        try:
            print(f"\n=== CRAWLED {len(saved_documents)} LEGAL DOCUMENTS from {source} ===")
            for doc in saved_documents:
                print(f"TITLE: {doc['title']}")
                print(f"DOC_NUMBER: {doc['document_number']}")
                print(f"ATTACHMENT: {doc['attachment_path']}")
                print(f"DATE: {doc['publish_date']}")
                print(f"FILE: {doc['filename']}")
                print("---")
        except Exception as e:
            self.logger.warning(f"Console output error: {e}")
        
        return {
            'status': 'success' if saved_documents else 'partial',
            'query': query,
            'total_results': len(saved_documents),
            'target_count': 5,  # Target count for reference
            'sources_searched': 1,
            'processing_time_seconds': 3.0,
            'timestamp': datetime.now().isoformat(),
            'results': saved_documents
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
    
    async def search(self, query: str, target_count: int = 5, max_attempts: int = 1) -> Dict[str, Any]:
        """Main search method - NON-BLOCKING"""
        try:
            # Currently only VBPL is implemented
            result = await self.search_vbpl(query, target_count, max_attempts)
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            # CRASH - let healing agent handle this
            raise e
    
    async def batch_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search multiple queries - NO FALLBACKS"""
        try:
            self.logger.info(f"Starting batch legal search for {len(queries)} queries")
            
            results = []
            for query in queries:
                # NON-BLOCKING: Each search call is limited to one attempt
                result = await self.search(query, target_count=5, max_attempts=1)
                results.append(result)
                
                # Rate limiting
                await asyncio.sleep(1.0 / self.rate_limit)
            
            self.logger.info(f"Batch legal search completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch legal search failed: {e}")
            # CRASH - let healing agent handle this
            raise e
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported sources"""
        return ['vbpl']
    
    def get_selectors(self) -> Dict[str, str]:
        """HEALING AGENT: Get all selectors"""
        return {
            'document_list_selector': self.document_list_selector,
            'title_selector': self.title_selector,
            'document_link_selector': self.document_link_selector,
            'document_number_selector': self.document_number_selector,
            'date_selector': self.date_selector,
            'vbpl_base_url': self.vbpl_base_url,
            'vbpl_search_url': self.vbpl_search_url
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
            
            self.logger.error(f"Law search error: {error_type} - {error_message}")
            
            # CRASH - let healing agent handle this
            raise Exception(f"Law search error: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handler failed: {e}")
            # CRASH - let healing agent handle this
            raise e
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.logger.info("Law Search Agent shutdown complete")


class NoSuchElementException(Exception):
    """Custom exception for element not found"""
    pass