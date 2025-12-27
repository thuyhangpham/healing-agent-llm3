#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup

def test_simple():
    """Simple test of VnExpress structure"""
    url = "https://vnexpress.net/so-hoa"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print("Page loaded successfully")
            
            # Try to find any article-like elements
            article_selectors = [
                'article',
                'div[class*="item"]',
                'div[class*="news"]',
                'div[class*="post"]',
                'li[class*="item"]',
                'section[class*="item"]'
            ]
            
            for selector in article_selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector: {selector}")
                    
                    # Show structure of first element
                    first = elements[0]
                    print(f"First element tag: {first.name}")
                    classes = first.get('class')
                    print(f"Classes: {classes}")
                    
                    # Look for title links inside
                    links = first.find_all('a')
                    if links:
                        for i, link in enumerate(links[:3]):
                            link_text = link.get_text(strip=True)
                            if link_text:
                                print(f"  Link {i}: {link_text[:50]}...")
                    break
            
            # Also show the overall structure
            print("\n=== Page structure ===")
            body = soup.find('body')
            if body:
                children = body.find_all(recursive=False)[:10]
                for i, child in enumerate(children):
                    print(f"  Body child {i}: <{child.name}> classes: {child.get('class')}")
        
        else:
            print(f"Failed to fetch page: {response.status_code}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simple()