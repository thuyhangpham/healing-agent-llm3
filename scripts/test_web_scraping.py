#!/usr/bin/env python3
"""
Selenium WebDriver Test

Simple test script to verify Selenium WebDriver configuration
and basic web scraping functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time


def test_selenium_basic():
    """Test basic Selenium WebDriver functionality."""
    print("Testing Selenium WebDriver...")
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        # Initialize WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        print("✓ Chrome WebDriver initialized successfully")
        
        # Test navigation
        driver.get("https://httpbin.org/html")
        print("✓ Successfully navigated to test page")
        
        # Test element finding
        title_element = driver.find_element(By.TAG_NAME, "h1")
        title_text = title_element.text
        print(f"✓ Found element with text: {title_text}")
        
        # Test page source for BeautifulSoup
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        print("✓ BeautifulSoup parsing successful")
        
        # Test finding elements with BeautifulSoup
        headings = soup.find_all(['h1', 'h2', 'h3'])
        print(f"✓ Found {len(headings)} headings with BeautifulSoup")
        
        driver.quit()
        print("✓ WebDriver closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Selenium test failed: {e}")
        return False


def test_beautifulsoup_basic():
    """Test basic BeautifulSoup functionality."""
    print("\nTesting BeautifulSoup...")
    
    try:
        # Test HTML parsing
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <div class="content">
                    <p>This is a test paragraph.</p>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Test basic parsing
        title = soup.find('title').text
        assert title == "Test Page"
        print("✓ Title parsing successful")
        
        # Test CSS selector
        content_div = soup.select_one('.content')
        assert content_div is not None
        print("✓ CSS selector successful")
        
        # Test list parsing
        list_items = soup.find_all('li')
        assert len(list_items) == 2
        print("✓ List parsing successful")
        
        return True
        
    except Exception as e:
        print(f"✗ BeautifulSoup test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=== Selenium and BeautifulSoup Test Suite ===\n")
    
    selenium_success = test_selenium_basic()
    bs4_success = test_beautifulsoup_basic()
    
    print(f"\n=== Test Results ===")
    print(f"Selenium: {'✓ PASS' if selenium_success else '✗ FAIL'}")
    print(f"BeautifulSoup: {'✓ PASS' if bs4_success else '✗ FAIL'}")
    
    if selenium_success and bs4_success:
        print("\n🎉 All tests passed! Web scraping setup is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the configuration.")
        return 1


if __name__ == "__main__":
    exit(main())