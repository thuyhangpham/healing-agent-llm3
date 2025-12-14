#!/usr/bin/env python3
"""
Ollama Integration Test

Test script to verify Ollama LLM integration and basic
code generation capabilities for healing operations.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
import requests


def test_ollama_api_connectivity():
    """Test Ollama API connectivity."""
    print("Testing Ollama API connectivity...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ API accessible, models: {[m['name'] for m in models]}")
            return True
        else:
            print(f"✗ API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False


def test_ollama_python_client():
    """Test Ollama Python client."""
    print("\nTesting Ollama Python client...")
    
    try:
        # Test basic generation
        response = ollama.generate(
            model='llama3',
            prompt='Say "Hello from Ollama!"',
            timeout=10
        )
        
        if 'response' in response:
            print(f"✓ Generation successful: {response['response'][:50]}...")
            return True
        else:
            print("✗ No response in generation")
            return False
            
    except Exception as e:
        print(f"✗ Client error: {e}")
        return False


def test_code_generation():
    """Test code generation capabilities."""
    print("\nTesting code generation capabilities...")
    
    code_prompt = """
Generate a simple Python function that adds two numbers.
Return only the code, no explanations.
"""
    
    try:
        start_time = time.time()
        response = ollama.generate(
            model='llama3',
            prompt=code_prompt,
            timeout=30
        )
        generation_time = time.time() - start_time
        
        if 'response' in response:
            code = response['response']
            print(f"✓ Code generated in {generation_time:.2f} seconds")
            print(f"Generated code:\n{code}")
            
            # Basic validation
            if 'def' in code and 'return' in code:
                print("✓ Generated code looks valid")
                return True
            else:
                print("✗ Generated code may be invalid")
                return False
        else:
            print("✗ No response in code generation")
            return False
            
    except Exception as e:
        print(f"✗ Code generation error: {e}")
        return False


def test_healing_prompt():
    """Test healing-specific prompt."""
    print("\nTesting healing prompt...")
    
    healing_prompt = """
You are an expert Python developer specializing in web scraping automation.
Analyze this error and generate a fix for the scraping code.

ERROR DETAILS:
- Error Type: AttributeError: 'NoneType' object has no attribute 'text'
- Traceback: line 15, in scrape_title
- Failed File: scraper.py

CURRENT CODE:
def scrape_title(soup):
    title_element = soup.find('h1', class_='title')
    return title_element.text

REQUIREMENTS:
1. Fix the NoneType error
2. Add error handling
3. Return only the fixed Python code

Generate the complete fixed function:
"""
    
    try:
        start_time = time.time()
        response = ollama.generate(
            model='llama3',
            prompt=healing_prompt,
            timeout=30
        )
        generation_time = time.time() - start_time
        
        if 'response' in response:
            fix_code = response['response']
            print(f"✓ Healing fix generated in {generation_time:.2f} seconds")
            print(f"Generated fix:\n{fix_code}")
            
            # Basic validation
            if 'def' in fix_code and 'try:' in fix_code:
                print("✓ Generated fix includes error handling")
                return True
            else:
                print("✗ Generated fix may be incomplete")
                return False
        else:
            print("✗ No response in healing prompt")
            return False
            
    except Exception as e:
        print(f"✗ Healing prompt error: {e}")
        return False


def main():
    """Main test function."""
    print("=== Ollama Integration Test Suite ===\n")
    
    tests = [
        ("API Connectivity", test_ollama_api_connectivity),
        ("Python Client", test_ollama_python_client),
        ("Code Generation", test_code_generation),
        ("Healing Prompt", test_healing_prompt),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n=== Test Results ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All Ollama tests passed! LLM integration is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Check Ollama configuration.")
        return 1


if __name__ == "__main__":
    exit(main())