#!/usr/bin/env python3
"""
Find actual selectors by examining real website structure.
"""

import requests
from bs4 import BeautifulSoup
import re

def find_vnexpress_selectors():
    """Find correct selectors for VnExpress Digital section"""
    url = "https://vnexpress.net/so-hoa"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print("=== ANALYZING VnExpress Digital Section ===")
            
            # Find any elements that contain links (likely articles)
            all_links = soup.find_all('a', href=True)
            articles_found = []
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Look for actual article links (not navigation)
                if (text and len(text) > 10 and 
                    'vnexpress.net' in href and 
                    not any(skip in href.lower() for skip in ['#', 'javascript:', 'tel:', 'mailto:'])):
                    
                    articles_found.append((text, href))
            
            print(f"Found {len(articles_found)} potential articles")
            
            # Show first 5 articles with their parent structures
            for i, (title, href) in enumerate(articles_found[:5]):
                print(f"\nArticle {i+1}:")
                print(f"  Title: {title[:80]}...")
                print(f"  URL: {href}")
                
                # Find parent element and show its structure
                link_element = soup.find('a', href=href)
                if link_element:
                    # Walk up to find a meaningful container
                    parent = link_element
                    for level in range(5):  # Check up to 5 levels up
                        parent = parent.parent
                        if not parent:
                            break
                        
                        # Check if this parent contains the article content
                        if len(parent.find_all('a')) >= 1:  # Contains at least our link
                            class_attr = parent.get('class', [])
                            tag_name = parent.name
                            print(f"  Container Level {level}: <{tag_name}> class: {class_attr}")
                            
                            # Check for description/sapo in this container
                            p_elements = parent.find_all(['p', 'div'], class_=lambda x: x and any(keyword in ' '.join(x).lower() for keyword in ['des', 'sapo', 'summary', 'lead']))
                            if p_elements:
                                print(f"    Description found: {p_elements[0].get_text(strip=True)[:50]}...")
                            
                            break
            
            # Suggest correct selectors based on findings
            print("\n=== SUGGESTED SELECTORS ===")
            print("article_list_selector: 'article' or elements containing article links")
            print("title_selector: 'a' (direct link element)")
            print("sapo_selector: Look for p/div elements near article links with description classes")
            
        else:
            print(f"Failed to fetch VnExpress: {response.status_code}")
    
    except Exception as e:
        print(f"Error analyzing VnExpress: {e}")

def find_vbpl_selectors():
    """Find correct selectors for VBPL"""
    url = "https://vbpl.vn/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print("\n=== ANALYZING VBPL Homepage ===")
            
            # Look for document links (PDF/DOC)
            all_links = soup.find_all('a', href=True)
            doc_links = []
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Look for document attachments
                if (href and any(ext in href.lower() for ext in ['.pdf', '.doc', '.docx'])):
                    doc_links.append((text, href))
            
            print(f"Found {len(doc_links)} document links")
            
            # Show first few document links
            for i, (title, href) in enumerate(doc_links[:5]):
                print(f"\nDocument {i+1}:")
                print(f"  Title: {title[:80]}...")
                print(f"  URL: {href}")
                
                # Find parent structure
                link_element = soup.find('a', href=href)
                if link_element:
                    parent = link_element.parent
                    if parent:
                        class_attr = parent.get('class', [])
                        tag_name = parent.name
                        print(f"  Container: <{tag_name}> class: {class_attr}")
            
            print("\n=== SUGGESTED SELECTORS ===")
            print("document_list_selector: Look for elements containing PDF/DOC links")
            print("document_link_selector: 'a[href*=\".pdf\"], a[href*=\".doc\"], a[href*=\".docx\"]'")
            
        else:
            print(f"Failed to fetch VBPL: {response.status_code}")
    
    except Exception as e:
        print(f"Error analyzing VBPL: {e}")

if __name__ == "__main__":
    find_vnexpress_selectors()
    find_vbpl_selectors()