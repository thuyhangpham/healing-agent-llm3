"""
Opinion Search Agent - REWRITTEN WITH KEYWORD ROTATION

Specialized agent for collecting opinion articles from VnExpress with
smart keyword rotation, cooldown logic, and resource-efficient crawling.
"""

import sys
import os
# Add project root to sys.path so we can import 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urljoin, urlparse, quote
from utils.logger import get_logger


class AutonomousOpinionSearchAgent:
    """Agent for searching and collecting opinion articles from VnExpress Digital section."""
    
    # Maximum pages to crawl per keyword (to focus on recent news and save resources)
    # Reduced to 2 for Chaos Engineering Demo - only need recent news
    MAX_PAGES_PER_KEYWORD = 2
    # Cooldown period: reset keyword to page 1 if last_updated > 6 hours ago
    COOLDOWN_HOURS = 6
    
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
        
        # State file for pagination tracking
        self.state_file = Path("data/production/crawl_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # CHAOS DEMO: Reset state file on init to start fresh every run
        if self.state_file.exists():
            try:
                self.state_file.unlink()
                self.logger.info("🔄 CHAOS DEMO: Deleted crawl_state.json to start fresh")
                print("🔄 CHAOS DEMO: Reset crawl state - starting from page 1")
            except Exception as e:
                self.logger.warning(f"Failed to delete crawl_state.json: {e}")
        
        # DYNAMIC SOUP STATEMENT ARCHITECTURE: Selectors loaded from soup_logic.txt
        self.vnexpress_base_url = 'https://vnexpress.net/'
        self.vnexpress_tech_url = 'https://vnexpress.net/so-hoa'  # Digital section
        self.vnexpress_search_url = 'https://timkiem.vnexpress.net/'  # Search page
        
        # Soup logic file path (Dynamic Soup Statement architecture)
        self.soup_logic_file = Path("data/production/soup_logic.txt")
        
        # Fallback selectors for extraction (not for finding articles - that's dynamic)
        self.search_title_selector = 'h3 a, .title-news a, .title a, h2 a'  # Title links in search results
        self.search_sapo_selector = '.description, .sapo, .lead, p.description'  # Description/sapo in search results
        self.search_date_selector = '.date, time, .time'  # Date in search results
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = None
        
        # Processed URLs cache (loaded once per search)
        self.processed_urls: Set[str] = set()
        
        self.logger.info("Opinion Search Agent initialized with keyword rotation and Dynamic Soup Statement architecture")
    
    def _setup_driver(self):
        """
        Create and configure Chrome WebDriver with anti-detection options.
        This makes Selenium stealthy to avoid bot detection.
        
        Returns:
            webdriver.Chrome: Configured Chrome driver instance
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        
        # Basic headless options
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # CRITICAL: Anti-detection options
        # Hide automation indicators
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Set realistic User-Agent (standard Chrome on Windows 10)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Experimental options to hide automation
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Additional stealth options
        chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        chrome_options.add_argument('--disable-site-isolation-trials')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to remove webdriver property (additional stealth)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
        return driver
    
    def _load_logic(self) -> str:
        """
        Load dynamic soup selector logic from data/production/soup_logic.txt.
        
        Returns:
            String containing Python code to execute (e.g., "soup.select('article.item-news')")
            
        Raises:
            Exception: If soup_logic.txt doesn't exist or is empty
        """
        try:
            # Ensure directory exists
            self.soup_logic_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.soup_logic_file.exists():
                # Create default if missing
                default_logic = "soup.select('article.item-news')"
                with open(self.soup_logic_file, 'w', encoding='utf-8') as f:
                    f.write(default_logic)
                self.logger.warning(f"soup_logic.txt not found, created default: {default_logic}")
                print(f"⚠️  Created default soup_logic.txt: {default_logic}")
                return default_logic
            
            # Read logic from file
            with open(self.soup_logic_file, 'r', encoding='utf-8') as f:
                logic_str = f.read().strip()
            
            if not logic_str:
                raise Exception(f"soup_logic.txt is empty")
            
            self.logger.info(f"Loaded dynamic soup logic: {logic_str}")
            print(f"📄 Loaded soup logic: {logic_str}")
            return logic_str
            
        except Exception as e:
            self.logger.error(f"Failed to load soup logic: {e}")
            raise Exception(f"Failed to load soup logic from {self.soup_logic_file}: {e}")
    
    def _load_crawl_state(self) -> Dict[str, Dict[str, Any]]:
        """Load crawl state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    print(f"📂 DEBUG: Loaded crawl state: {len(state)} keywords")
                    return state
            else:
                print(f"📂 DEBUG: No crawl state file found, starting fresh")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load crawl state: {e}")
            return {}
    
    def _save_crawl_state(self, state: Dict[str, Dict[str, Any]]):
        """Save crawl state to file"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            print(f"💾 DEBUG: Saved crawl state for {len(state)} keywords")
        except Exception as e:
            self.logger.error(f"Failed to save crawl state: {e}")
    
    def _load_all_keywords_from_laws(self) -> List[str]:
        """Load ALL keywords from all law files in data/production/laws/"""
        keywords = []
        try:
            laws_dir = Path("data/production/laws")
            if laws_dir.exists():
                for law_file in laws_dir.glob("*.json"):
                    try:
                        with open(law_file, 'r', encoding='utf-8') as f:
                            law_data = json.load(f)
                            if 'search_keywords' in law_data:
                                keywords.extend(law_data['search_keywords'])
                            # Also check for alternative keyword fields
                            for key in ['keywords', 'tags', 'topics']:
                                if key in law_data and isinstance(law_data[key], list):
                                    keywords.extend(law_data[key])
                    except Exception as e:
                        self.logger.warning(f"Failed to load keywords from {law_file}: {e}")
            
            # Remove duplicates and empty strings
            keywords = list(set([k.strip() for k in keywords if k and k.strip()]))
            print(f"📋 DEBUG: Loaded {len(keywords)} unique keywords from law files: {keywords}")
            return keywords
        except Exception as e:
            self.logger.warning(f"Failed to load keywords from laws: {e}")
            return []
    
    def _select_next_keyword(self) -> Optional[str]:
        """
        Select next keyword using round-robin logic:
        - Sort by last_updated (ascending) - pick the least recently processed
        - If keyword has reached MAX_PAGES, skip it (cooldown)
        - If keyword hasn't been processed in > 6 hours, reset it to page 1
        """
        keywords = self._load_all_keywords_from_laws()
        if not keywords:
            print("⚠️  No keywords found in law files")
            return None
        
        state = self._load_crawl_state()
        now = datetime.now()
        cooldown_threshold = now - timedelta(hours=self.COOLDOWN_HOURS)
        
        # Prepare keyword candidates with metadata
        candidates = []
        for keyword in keywords:
            keyword_key = keyword  # Use keyword directly as key (simpler)
            
            if keyword_key in state:
                keyword_state = state[keyword_key]
                last_page = keyword_state.get('last_page', 1)
                last_updated_str = keyword_state.get('last_updated', '')
                status = keyword_state.get('status', 'active')
                
                # Parse last_updated timestamp
                try:
                    if 'T' in last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                    else:
                        # Try alternative format
                        last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S')
                except:
                    last_updated = datetime.min
                
                # Check if needs reset (cooldown expired)
                if last_updated < cooldown_threshold:
                    print(f"🔄 DEBUG: Keyword '{keyword}' cooldown expired (>6h), resetting to page 1")
                    state[keyword_key] = {
                        'last_page': 1,
                        'last_updated': now.isoformat(),
                        'status': 'active'
                    }
                    last_page = 1
                    last_updated = now
                
                # Skip if reached max pages (cooldown)
                if last_page > self.MAX_PAGES_PER_KEYWORD:
                    print(f"⏸️  DEBUG: Keyword '{keyword}' reached limit ({last_page} > {self.MAX_PAGES_PER_KEYWORD}), skipping")
                    continue
                
                candidates.append({
                    'keyword': keyword,
                    'last_updated': last_updated,
                    'last_page': last_page,
                    'status': status
                })
            else:
                # New keyword - add to state
                state[keyword_key] = {
                    'last_page': 1,
                    'last_updated': now.isoformat(),
                    'status': 'active'
                }
                candidates.append({
                    'keyword': keyword,
                    'last_updated': datetime.min,  # Never processed - prioritize
                    'last_page': 1,
                    'status': 'active'
                })
        
        # Save state updates
        if state:
            self._save_crawl_state(state)
        
        if not candidates:
            print("⚠️  All keywords have reached page limit. Waiting for cooldown...")
            return None
        
        # Sort by last_updated (ascending) - oldest first (round-robin)
        candidates.sort(key=lambda x: x['last_updated'])
        selected = candidates[0]
        
        print(f"\n🔄 KEYWORD ROTATION: Selected '{selected['keyword']}'")
        print(f"   Last processed: {selected['last_updated']}")
        print(f"   Current page: {selected['last_page']}")
        print(f"   Status: {selected['status']}")
        
        return selected['keyword']
    
    def _get_current_page(self, keyword: str) -> int:
        """Get current page number for keyword from state"""
        state = self._load_crawl_state()
        keyword_key = keyword
        
        if keyword_key in state and 'last_page' in state[keyword_key]:
            page = state[keyword_key]['last_page']
            print(f"📄 DEBUG: Found page {page} for keyword '{keyword}' in state")
            return page
        else:
            print(f"📄 DEBUG: No state found for keyword '{keyword}', starting at page 1")
            return 1
    
    def _update_keyword_state(self, keyword: str, page: int, status: str = 'active'):
        """Update keyword state in crawl_state.json"""
        state = self._load_crawl_state()
        keyword_key = keyword
        
        if keyword_key not in state:
            state[keyword_key] = {}
        
        state[keyword_key]['last_page'] = page
        state[keyword_key]['last_updated'] = datetime.now().isoformat()
        state[keyword_key]['status'] = status
        
        self._save_crawl_state(state)
        print(f"💾 DEBUG: Updated state for '{keyword}': page {page}, status {status}")
    
    def _get_processed_urls(self) -> Set[str]:
        """Scan data/production/opinions/ directory and collect all 'link' URLs from JSON files"""
        processed_urls = set()
        
        try:
            if not self.data_dir.exists():
                print(f"📁 DEBUG: Opinions directory does not exist: {self.data_dir}")
                return processed_urls
            
            json_files = list(self.data_dir.glob("*.json"))
            print(f"📁 DEBUG: Scanning {len(json_files)} JSON files for processed URLs...")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check for 'link' field (the URL field name in our JSON structure)
                        if 'link' in data and data['link']:
                            processed_urls.add(data['link'])
                        # Also check 'url' field for compatibility
                        elif 'url' in data and data['url']:
                            processed_urls.add(data['url'])
                except Exception as e:
                    self.logger.warning(f"Failed to read {json_file.name}: {e}")
                    continue
            
            print(f"✅ DEBUG: Found {len(processed_urls)} processed URLs in existing files")
            return processed_urls
            
        except Exception as e:
            self.logger.error(f"Error scanning processed URLs: {e}")
            return processed_urls
    
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
    
    async def search_vnexpress_search_page(self, keyword: str, page_num: int) -> Dict[str, Any]:
        """Search VnExpress using timkiem.vnexpress.net search page with keyword filtering, duplicate detection, and pagination"""
        from bs4 import BeautifulSoup
        
        print(f"\n🔍 DEBUG: Searching timkiem.vnexpress.net")
        print(f"   Keyword: {keyword}")
        print(f"   📄 Scraping Page {page_num} for keyword '{keyword}'...")
        
        # Load processed URLs to avoid duplicates
        self.processed_urls = self._get_processed_urls()
        print(f"   Already processed URLs: {len(self.processed_urls)}")
        
        # Build search URL with pagination
        encoded_keyword = quote(keyword)
        search_url = f"{self.vnexpress_search_url}?q={encoded_keyword}&latest=1&page={page_num}"
        
        print(f"   Search URL: {search_url}")
        
        # Try to use Selenium for scrolling (to load more results via lazy loading)
        article_elements = []
        html_content = None
        
        try:
            # Try Selenium first for JavaScript rendering and scrolling
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            print(f"\n🌐 DEBUG: Using Selenium for JavaScript rendering and scrolling...")
            
            # Use stealth driver setup
            driver = self._setup_driver()
            
            try:
                # Navigate to search URL
                driver.get(search_url)
                
                # Wait for body element to be present (critical for page load detection)
                print(f"   ⏳ Waiting for page to load (max 15s)...")
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Additional wait to ensure body has content (not just empty tag)
                WebDriverWait(driver, 10).until(
                    lambda d: d.find_element(By.TAG_NAME, "body").text.strip() != "" or len(d.page_source) > 1000
                )
                
                # Wait for JavaScript to finish executing
                time.sleep(3)
                
                # Scroll to load lazy-loaded content
                print(f"   📜 Scrolling to load more results...")
                for scroll_attempt in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    print(f"      Scroll {scroll_attempt + 1}/3 completed")
                
                # Get HTML after scrolling
                html_content = driver.page_source
                print(f"   ✅ Page loaded and scrolled, HTML length: {len(html_content)} characters")
                
            finally:
                driver.quit()
                
        except ImportError:
            print(f"\n⚠️  DEBUG: Selenium not available, using aiohttp (no scrolling)...")
            # Fallback to aiohttp if Selenium is not available
            response = await self.session.get(search_url)
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: Failed to access VnExpress search page")
            html_content = await response.text()
            
        except Exception as e:
            print(f"\n⚠️  DEBUG: Selenium failed ({e}), falling back to aiohttp...")
            # Fallback to aiohttp if Selenium fails
            response = await self.session.get(search_url)
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: Failed to access VnExpress search page")
            html_content = await response.text()
        
        # SAFETY CHECK: Detect blocked pages BEFORE parsing/evaluating selectors
        # If HTML is too short, this is NOT a selector issue - it's a blocked IP or network issue
        if len(html_content) < 1000:
            error_msg = f"Page load failed! HTML is too short ({len(html_content)} chars). This indicates anti-bot detection triggered or page failed to load - NOT a selector issue."
            self.logger.error(error_msg)
            print(f"\n🚨 {error_msg}")
            raise ConnectionError("Anti-bot detection triggered or page failed to load. HTML length: {} chars".format(len(html_content)))
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # DEBUG: Print raw HTML snippet to see structure
        print(f"\n📄 DEBUG: HTML structure check")
        print(f"   HTML length: {len(html_content)} characters")
        
        # DYNAMIC SOUP STATEMENT ARCHITECTURE: Load and execute logic from soup_logic.txt
        logic_str = None
        article_elements = []
        
        try:
            logic_str = self._load_logic()
            print(f"   🔧 Executing dynamic logic: {logic_str}")
            
            # Execute the logic string with soup in scope
            # The logic should be something like: soup.select('article.item-news')
            article_elements = eval(logic_str, {'soup': soup, '__builtins__': __builtins__})
            
            # Ensure result is a list
            if not isinstance(article_elements, list):
                article_elements = list(article_elements) if article_elements else []
            
            print(f"   ✅ Executed dynamic logic, found {len(article_elements)} article elements")
            
        except Exception as e:
            self.logger.error(f"Failed to execute dynamic soup logic: {e}")
            print(f"   ❌ ERROR executing dynamic logic: {e}")
            # Capture HTML snapshot for healing agent
            html_snapshot = html_content[:5000] if len(html_content) > 5000 else html_content
            logic_display = logic_str if logic_str else 'unknown'
            error_msg = f"Failed to execute dynamic soup logic '{logic_display}': {e}"
            raise Exception(error_msg)
        
        # FAIL FAST: If no articles found, raise exception to trigger healing
        if not article_elements or len(article_elements) == 0:
            logic_display = logic_str if logic_str else 'unknown'
            print(f"\n🚨 FAIL FAST: No articles found using dynamic logic: {logic_display}")
            print(f"   This indicates a broken selector - triggering healing agent...")
            # Capture HTML snapshot for healing agent (first 5000 chars)
            html_snapshot = html_content[:5000] if len(html_content) > 5000 else html_content
            error_msg = f"No articles found using dynamic logic '{logic_display}'. HTML length: {len(html_content)} chars"
            raise Exception(error_msg)
        
        print(f"\n📊 DEBUG: Found {len(article_elements)} raw article elements")
        print(f"   Processing ALL articles...")
        
        # Collect articles with duplicate detection
        articles = []
        skipped_count = 0
        skipped_reasons = {}
        duplicate_count = 0
        
        # Process ALL articles (no limit)
        for i, element in enumerate(article_elements):
            try:
                print(f"\n--- Processing article {i+1}/{len(article_elements)} ---")
                
                # Extract URL first to check for duplicates
                article_url = await self._extract_url_from_element(element)
                
                if not article_url:
                    print(f"   ❌ SKIP: Failed to extract URL")
                    skipped_count += 1
                    skipped_reasons['no_url'] = skipped_reasons.get('no_url', 0) + 1
                    continue
                
                # --- QUALITY FILTER: Strict URL filtering to exclude ads and irrelevant links ---
                article_url_lower = article_url.lower()
                
                # Blacklist: Skip ads, tracking links, and promotional content
                # Check for: eclick, click, ads, promotion, video
                blacklist_keywords = ['eclick', 'click', 'ads', 'promotion', 'video']
                is_ad = any(keyword in article_url_lower for keyword in blacklist_keywords)
                
                # Whitelist: Must be a VnExpress article URL ending with .html
                is_article = 'vnexpress.net' in article_url_lower and article_url_lower.endswith('.html')
                
                if is_ad or not is_article:
                    skip_reason = 'ad' if is_ad else 'not_article'
                    print(f"   ⏭️  SKIP: Ignored non-article/ad link: {article_url[:80]}...")
                    self.logger.info(f"⏭️ SKIP: Ignored non-article/ad link: {article_url} (reason: {skip_reason})")
                    skipped_count += 1
                    skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                    continue
                # --------------------------------------------------------------------------------
                
                # Check for duplicate
                if article_url in self.processed_urls:
                    print(f"   ⏭️  SKIP DUPLICATE: {article_url[:80]}...")
                    duplicate_count += 1
                    skipped_count += 1
                    skipped_reasons['duplicate'] = skipped_reasons.get('duplicate', 0) + 1
                    continue
                
                # Extract full article details
                article = await self._extract_article_from_search_result(element, i)
                
                if not article:
                    print(f"   ❌ SKIP: Failed to extract article data")
                    skipped_count += 1
                    skipped_reasons['extraction_failed'] = skipped_reasons.get('extraction_failed', 0) + 1
                    continue
                
                # Verify URL matches
                if article.get('link') != article_url:
                    article['link'] = article_url
                
                title = article.get('title', '')
                print(f"   📰 Title: {title[:80]}...")
                print(f"   🔗 URL: {article_url[:80]}...")
                
                # Keyword matching (check if keyword appears in title/sapo)
                title_lower = title.lower()
                sapo_lower = article.get('sapo', '').lower()
                keyword_lower = keyword.lower()
                
                if keyword_lower in title_lower or keyword_lower in sapo_lower:
                    print(f"   ✅ MATCH: Keyword '{keyword}' found in content")
                    article['matched_keyword'] = keyword
                else:
                    print(f"   ⚠️  Keyword '{keyword}' not found in content, but saving anyway")
                
                articles.append(article)
                print(f"   ✅ SAVED: Article {len(articles)} (NEW)")
                    
            except Exception as e:
                print(f"   ❌ ERROR extracting article {i+1}: {e}")
                skipped_count += 1
                skipped_reasons['extraction_error'] = skipped_reasons.get('extraction_error', 0) + 1
                self.logger.error(f"Error extracting article {i+1}: {e}")
                # Continue to next article instead of crashing
                continue
        
        print(f"\n📊 DEBUG: Collection Summary for Page {page_num}")
        print(f"   Total raw elements found: {len(article_elements)}")
        print(f"   New articles saved: {len(articles)}")
        print(f"   Duplicates skipped: {duplicate_count}")
        print(f"   Other skips: {skipped_count - duplicate_count}")
        print(f"   Skip reasons: {skipped_reasons}")
        
        # Auto-increment logic: Update page state based on results
        new_articles_saved = len(articles)
        total_found = len(article_elements)
        
        if new_articles_saved == 0 or total_found > 0:
            # If all duplicates OR we found articles (even if filtered out), move to next page
            next_page = page_num + 1
            
            # Check if next page would exceed limit
            if next_page > self.MAX_PAGES_PER_KEYWORD:
                print(f"\n📄 DEBUG: Page {page_num} reached. Next page ({next_page}) would exceed limit of {self.MAX_PAGES_PER_KEYWORD}.")
                print(f"   Saving state and sleeping this keyword. Will rotate to next keyword next run.")
                # Save state with current page (don't increment) - cooldown
                self._update_keyword_state(keyword, page_num, 'cooldown')
            else:
                print(f"\n📄 DEBUG: Auto-incrementing page: {page_num} -> {next_page}")
                print(f"   Reason: new_articles_saved={new_articles_saved}, total_found={total_found}")
                self._update_keyword_state(keyword, next_page, 'active')
        else:
            # If no articles found at all, reset to page 1
            print(f"\n📄 DEBUG: No articles found, resetting to page 1")
            self._update_keyword_state(keyword, 1, 'active')
        
        if len(articles) == 0:
            if duplicate_count > 0:
                print(f"\n⚠️  All articles were duplicates. Moving to next page.")
            else:
                print(f"\n⚠️  No articles found after filtering {len(article_elements)} results")
        
        # Return result with count of newly saved articles
        result = await self._save_and_return_articles(articles, 'VnExpress Search', keyword)
        result['new_articles_count'] = len(articles)
        result['duplicates_skipped'] = duplicate_count
        result['page_number'] = page_num
        result['keyword'] = keyword
        return result
    
    async def _extract_url_from_element(self, element) -> Optional[str]:
        """Extract URL from article element (quick check for duplicates)"""
        try:
            from bs4 import BeautifulSoup
            
            # Try multiple title selectors to find the link
            title_selectors = [
                self.search_title_selector,
                'h3 a',
                '.title-news a',
                '.title a',
                'h2 a',
                'a.title',
                'a'
            ]
            
            for selector in title_selectors:
                title_elem = element.select_one(selector)
                if title_elem:
                    link = title_elem.get('href', '')
                    if link:
                        # Convert to absolute URL
                        if not link.startswith('http'):
                            link = urljoin(self.vnexpress_base_url, link)
                        return link
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting URL: {e}")
            return None
    
    async def _extract_article_from_search_result(self, element, index: int) -> Dict[str, Any]:
        """Extract article details from search result page element"""
        try:
            # Try multiple title selectors (search page structure is different)
            title_elem = None
            title_selectors = [
                self.search_title_selector,
                'h3 a',
                '.title-news a',
                '.title a',
                'h2 a',
                'a.title',
                'a'
            ]
            
            for selector in title_selectors:
                title_elem = element.select_one(selector)
                if title_elem:
                    break
            
            if not title_elem:
                print(f"      ⚠️  No title element found with any selector")
                return None
            
            title = title_elem.get_text(strip=True)
            if not title:
                print(f"      ⚠️  Empty title found")
                return None
            
            # Extract link
            link = title_elem.get('href', '')
            if not link:
                print(f"      ⚠️  No href found on title element")
                return None
            
            # Convert to absolute URL
            if not link.startswith('http'):
                link = urljoin(self.vnexpress_base_url, link)
            
            # Find sapo/description
            sapo = ""
            sapo_selectors = [
                self.search_sapo_selector,
                '.description',
                '.sapo',
                '.lead',
                'p.description',
                'p'
            ]
            
            for selector in sapo_selectors:
                sapo_elem = element.select_one(selector)
                if sapo_elem:
                    sapo = sapo_elem.get_text(strip=True)
                    if sapo:
                        break
            
            # Extract date
            date_elem = element.select_one(self.search_date_selector)
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                publish_date = self._parse_date(date_text)
            else:
                publish_date = datetime.now().strftime('%Y-%m-%d')
            
            return {
                'title': title[:200],  # Limit title length
                'link': link,
                'sapo': sapo[:300] if sapo else "",  # Limit sapo length
                'date': publish_date,
                'source': 'VnExpress',
                'section': 'Search Results',
                'relevance_score': 0.9 - (index * 0.05),
                'found_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"      ❌ Exception in extraction: {e}")
            return None
    
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
            print(f"\n=== CRAWLED {len(saved_articles)} NEW OPINION ARTICLES from {source} ===")
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
            'new_articles_count': len(saved_articles),  # Number of newly saved articles
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
    
    async def search(self, query: str = None, target_count: int = None, max_attempts: int = 1, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Main search method - implements keyword rotation and processes ONE keyword per call.
        This method should be called by run_production.py in a loop.
        """
        try:
            # Select next keyword using round-robin logic
            keyword = self._select_next_keyword()
            
            if not keyword:
                return {
                    'status': 'no_keywords',
                    'message': 'No keywords available or all keywords in cooldown',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get current page from state
            page_num = self._get_current_page(keyword)
            
            # Check page limit BEFORE crawling
            if page_num > self.MAX_PAGES_PER_KEYWORD:
                print(f"\n{'='*80}")
                print(f"⚠️  Reached limit of {self.MAX_PAGES_PER_KEYWORD} pages for keyword '{keyword}'.")
                print(f"   Current page: {page_num} (exceeds limit of {self.MAX_PAGES_PER_KEYWORD})")
                print(f"   Sleeping this keyword. Will rotate to next keyword next run.")
                print(f"{'='*80}\n")
                
                # Save state (keep current page) - cooldown
                self._update_keyword_state(keyword, page_num, 'cooldown')
                
                # Return early - let rotation pick different keyword next time
                return {
                    'status': 'limit_reached',
                    'keyword': keyword,
                    'current_page': page_num,
                    'max_pages': self.MAX_PAGES_PER_KEYWORD,
                    'message': f"Page limit reached. Sleeping keyword. Next run will rotate to different keyword.",
                    'total_results': 0,
                    'new_articles_count': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            print(f"\n{'='*80}")
            print(f"🔍 OPINION SEARCH DEBUG: Starting search")
            print(f"   Keyword: {keyword}")
            print(f"   📄 Current page: {page_num} (from state)")
            print(f"   📊 Page limit: {self.MAX_PAGES_PER_KEYWORD} pages per keyword")
            print(f"{'='*80}\n")
            
            # Use search page with pagination - process ONE keyword for ONE page
            result = await self.search_vnexpress_search_page(keyword, page_num)
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            
            # HEALING AGENT: Report error before crashing
            # Try to load current soup logic for error context
            try:
                current_logic = self._load_logic()
            except:
                current_logic = 'unknown'
            
            error_context = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'function': 'search',
                'selector_used': current_logic,  # Dynamic soup logic from soup_logic.txt
                'soup_logic_file': str(self.soup_logic_file)
            }
            
            # This will write error to global file and then crash
            await self.handle_error(error_context)
    
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
        """Get current selectors (for compatibility - now uses dynamic logic)"""
        try:
            logic_str = self._load_logic()
            return {
                'dynamic_soup_logic': logic_str,
                'vnexpress_base_url': self.vnexpress_base_url,
                'vnexpress_tech_url': self.vnexpress_tech_url,
                'soup_logic_file': str(self.soup_logic_file)
            }
        except Exception as e:
            self.logger.warning(f"Failed to load logic for get_selectors: {e}")
            return {
                'dynamic_soup_logic': 'unknown',
                'vnexpress_base_url': self.vnexpress_base_url,
                'vnexpress_tech_url': self.vnexpress_tech_url,
                'soup_logic_file': str(self.soup_logic_file)
            }
    
    async def handle_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during search operations - NO FALLBACKS"""
        error_type = error_context.get('error_type', 'UnknownError')
        error_message = error_context.get('error_message', 'Unknown error occurred')
        
        self.logger.error(f"Opinion search error: {error_type} - {error_message}")
        
        # Write error signal for healing agent (subprocess-based)
        import traceback
        traceback_str = ''.join(traceback.format_exc())
        self.write_error_signal(error_type, error_message, traceback_str)
        
        # CRASH - let healing agent handle this
        raise Exception(f"Opinion search error: {error_type} - {error_message}")
    
    def write_error_signal(self, error_type: str, error_message: str, traceback_str: str = None):
        """Write error signal JSON file for healing agent (subprocess-based architecture)"""
        try:
            from pathlib import Path
            import json
            import traceback as tb
            
            # Get project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            error_signal_file = project_root / "data" / "production" / "error_signal.json"
            
            # Ensure directory exists
            error_signal_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to load current soup logic for error signal
            try:
                current_logic = self._load_logic()
            except:
                current_logic = 'unknown'
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': error_message,
                'agent_name': self.name,
                'function_name': 'search',
                'file_path': str(__file__),
                'selector_used': current_logic,  # Dynamic soup logic from soup_logic.txt
                'soup_logic_file': str(self.soup_logic_file),
                'severity': 'critical',
                'traceback': traceback_str or ''.join(tb.format_exc())
            }
            
            with open(error_signal_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Error signal written to: {error_signal_file}")
            print(f"\n🚨 ERROR SIGNAL CREATED: {error_signal_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write error signal: {e}")
            print(f"❌ Failed to write error signal: {e}")
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.logger.info("Opinion Search Agent shutdown complete")


class NoSuchElementException(Exception):
    """Custom exception for element not found"""
    pass


async def main():
    """
    Main entry point for one-shot opinion search agent process.
    This function runs once, processes one keyword, then exits.
    """
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
            agent_config = full_config.get('agents', {}).get('opinion_search_agent', {})
        else:
            agent_config = {}
    except Exception as e:
        print(f"⚠️  Could not load config, using defaults: {e}")
        agent_config = {}
    
    # Initialize agent
    agent = AutonomousOpinionSearchAgent(agent_config)
    
    try:
        # Initialize async components
        await agent.initialize()
        
        # Run search once (processes one keyword, one page)
        result = await agent.search()
        
        # Log result
        if result.get('status') == 'success':
            print(f"\n✅ Search completed successfully")
            print(f"   New articles: {result.get('new_articles_count', 0)}")
            print(f"   Keyword: {result.get('keyword', 'unknown')}")
            print(f"   Page: {result.get('page_number', 'unknown')}")
        elif result.get('status') == 'limit_reached':
            print(f"\n⏸️  Page limit reached for keyword")
            print(f"   Keyword: {result.get('keyword', 'unknown')}")
        elif result.get('status') == 'no_keywords':
            print(f"\n⚠️  No keywords available or all in cooldown")
        else:
            print(f"\n⚠️  Search returned: {result.get('status', 'unknown')}")
        
        # Cleanup
        await agent.shutdown()
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        # Write error signal and exit with error code
        print(f"\n❌ ERROR in opinion search agent: {e}")
        import traceback
        traceback_str = ''.join(traceback.format_exc())
        
        # Write error signal for healing agent
        agent.write_error_signal(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback_str
        )
        
        # Cleanup
        try:
            await agent.shutdown()
        except:
            pass
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
