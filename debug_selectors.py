#!/usr/bin/env python3
"""
Debug script to test actual HTML structure and fix selectors.
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin

async def debug_vnexpress():
    """Debug VnExpress HTML structure"""
    url = "https://vnexpress.net/so-hoa"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                print(f"Page title: {soup.title.string if soup.title else 'No title'}")
                
                # Find all possible article containers
                print("\n=== Finding article containers ===")
                
                selectors_to_try = [
                    'article.item-news',
                    'div.item-news',
                    'article',
                    '.item',
                    'div.news-item',
                    'li.item',
                    'div.result-item',
                    '.article-item',
                    '[class*="item"]',
                    '[class*="article"]'
                ]
                
                for selector in selectors_to_try:
                    elements = soup.select(selector)
                    if elements:
                        print(f"✅ Found {len(elements)} elements with: {selector}")
                        # Show structure of first element
                        first = elements[0]
                        print(f"   First element tag: {first.name}")
                        print(f"   First element classes: {first.get('class', [])}")
                        
                        # Try to find title and link
                        title_selectors = ['h3.title', 'h2.title', '.title a', 'a.title', 'a', 'h3', 'h2']
                        for title_sel in title_selectors:
                            title_elem = first.select_one(title_sel)
                            if title_elem:
                                title_text = title_elem.get_text(strip=True)
                                print(f"   ✅ Title found with {title_sel}: {title_text[:50]}...")
                                break
                        else:
                            print(f"   ❌ No title found")
                            
                        print("   ---")
                        break
                    else:
                        print(f"❌ No elements with: {selector}")
                
                # Also show some sample HTML structure
                print("\n=== Sample HTML structure ===")
                main_content = soup.select_one('main') or soup.select_one('.main') or soup.select_one('#main')
                if main_content:
                    print(f"Main content tag: {main_content.name}")
                    print(f"Main content classes: {main_content.get('class', [])}")
                    
                    # Show first few children
                    for i, child in enumerate(main_content.find_all(recursive=False)[:5]):
                        print(f"   Child {i}: <{child.name}> classes: {child.get('class', [])}")
                else:
                    print("No main content container found")
                    # Show body structure
                    body = soup.select_one('body')
                    if body:
                        print("Body children:")
                        for i, child in enumerate(body.find_all(recursive=False)[:10]):
                            print(f"   Child {i}: <{child.name}> classes: {child.get('class', [])}")
                
            else:
                print(f"Failed to fetch page: {response.status}")

if __name__ == "__main__":
    asyncio.run(debug_vnexpress())