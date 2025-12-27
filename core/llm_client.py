"""
LLM Client for Ollama Integration

This module provides integration with Ollama for local LLM inference,
specifically optimized for code analysis and generation tasks in the
self-healing system.

Features:
- HTTP client for Ollama API communication
- Prompt engineering templates for code repair
- Response parsing and code extraction
- Temperature and token management
- Error handling and retry mechanisms
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from utils.logger import get_logger


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3:latest"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    created_at: datetime
    done: bool
    total_duration: float
    prompt_eval_count: int
    eval_count: int
    error: Optional[str] = None


class LLMClient:
    """
    Client for interacting with Ollama LLM for code analysis and generation
    
    This client handles communication with the local Ollama server and provides
    specialized methods for code repair tasks in the healing system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize LLM client"""
        self.config = LLMConfig(**(config or {}))
        self.logger = get_logger("llm_client")
        self.session: Optional[aiohttp.ClientSession] = None
    
    def make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self.make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif hasattr(obj, 'value'):  # Handle Enum
            return obj.value
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        else:
            return obj
        
        # Prompt templates
    
    def make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self.make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif hasattr(obj, 'value'):  # Handle Enum
            return obj.value
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        else:
            return obj
        
        # Prompt templates
        self.code_analysis_prompt = """
You are an expert Python developer and debugging specialist. Analyze the following error and provide a detailed assessment.

ERROR CONTEXT:
- Error Type: {error_type}
- Error Message: {error_message}
- File: {file_path}:{line_number}
- Function: {function_name}
- Agent: {agent_name}

TRACEBACK:
{traceback}

ADDITIONAL CONTEXT:
{additional_context}

HTML SNAPSHOT (if applicable):
{html_snapshot}

Please analyze this error and provide:
1. Root cause analysis
2. Whether this error is repairable with code changes
3. Suggested approach for fixing
4. Confidence level (0-1)
5. Category of error (web_scraping_failure, syntax_error, logic_error, import_error, etc.)

Respond in JSON format:
{{
    "repairable": true/false,
    "root_cause": "detailed explanation",
    "error_category": "category",
    "confidence": 0.8,
    "suggested_approach": "detailed approach",
    "required_changes": ["list of required changes"]
}}
"""

        self.code_generation_prompt = """
You are an expert Python developer specializing in automated code repair. Generate a fix for the following error.

ERROR DETAILS:
- Error Type: {error_type}
- Error Message: {error_message}
- File: {file_path}:{line_number}
- Function: {function_name}

ANALYSIS:
{analysis}

CURRENT CODE (if available):
{current_code}

REQUIREMENTS:
1. Generate minimal, targeted fix that addresses the root cause
2. Maintain existing code style and patterns
3. Add appropriate error handling
4. Include comments explaining the fix
5. Ensure backward compatibility
6. Do not change the function signature unless absolutely necessary

Generate only the corrected code snippet that should replace the problematic section. 
If the entire function needs replacement, provide the complete function.
If only a specific line needs fixing, provide just that line with proper indentation.

Format your response as:
```python
# Your corrected code here
```
"""

        self.validation_prompt = """
You are a code validation specialist. Review the following fix and determine if it properly addresses the error.

ORIGINAL ERROR:
{error_context}

PROPOSED FIX:
{fix_code}

ANALYSIS:
{analysis}

Please validate the fix and provide:
1. Whether the fix addresses the root cause
2. Potential side effects or issues
3. Code quality assessment
4. Likelihood of success (0-1)

Respond in JSON format:
{{
    "fix_valid": true/false,
    "addresses_root_cause": true/false,
    "side_effects": ["list of potential side effects"],
    "code_quality_score": 0.8,
    "success_likelihood": 0.9,
    "recommendations": ["list of recommendations"]
}}
"""
    
    async def initialize(self) -> bool:
        """Initialize the LLM client"""
        try:
            self.logger.info(f"Initializing LLM client for model: {self.config.model}")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={"Content-Type": "application/json"}
            )
            
            # Test connection to Ollama
            if await self._test_connection():
                self.logger.info(f"LLM client initialized successfully with model: {self.config.model}")
                return True
            else:
                self.logger.error("Failed to connect to Ollama server")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            url = f"{self.config.base_url}/api/tags"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    
                    # Check if the configured model is available
                    if self.config.model in models:
                        self.logger.info(f"Model {self.config.model} is available")
                        return True
                    
                    # Try to auto-resolve model name
                    resolved_model = self._resolve_model_name(self.config.model, models)
                    if resolved_model != self.config.model and resolved_model in models:
                        self.logger.info(f"Auto-resolved model from {self.config.model} to {resolved_model}")
                        self.config.model = resolved_model
                        return True
                    
                    self.logger.warning(f"Model {self.config.model} not found. Available models: {models}")
                    return False
                else:
                    self.logger.error(f"Ollama server returned status {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to test Ollama connection: {e}")
            return False
    
    async def analyze_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using LLM
        
        Args:
            error_context: Dictionary containing error information
            
        Returns:
            Analysis result with repairability assessment
        """
        try:
            prompt = self.code_analysis_prompt.format(
                error_type=error_context.get('error_type', 'Unknown'),
                error_message=error_context.get('error_message', ''),
                file_path=error_context.get('file_path', ''),
                line_number=error_context.get('line_number', 0),
                function_name=error_context.get('function_name', ''),
                agent_name=error_context.get('agent_name', ''),
                traceback=error_context.get('traceback_str', ''),
                additional_context=json.dumps(self.make_serializable(error_context.get('additional_context', {})), indent=2),
                html_snapshot=error_context.get('html_snapshot', 'N/A')
            )
            
            response = await self._generate_response(prompt)
            
            if response.error:
                return {
                    'repairable': False,
                    'reason': f'LLM analysis failed: {response.error}'
                }
            
            # Parse JSON response
            try:
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse LLM analysis as JSON, using fallback")
                return {
                    'repairable': True,
                    'error_category': 'unknown',
                    'confidence': 0.5,
                    'suggested_approach': response.content[:200]
                }
                
        except Exception as e:
            self.logger.error(f"Error analysis failed: {e}")
            return {
                'repairable': False,
                'reason': f'Analysis failed: {e}'
            }
    
    async def generate_fix(self, error_context: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate code fix using LLM
        
        Args:
            error_context: Error context information
            analysis: Previous analysis results
            
        Returns:
            Generated fix code or None if failed
        """
        try:
            # Get current code if available
            current_code = await self._get_current_code(error_context.get('file_path', ''))
            
            prompt = self.code_generation_prompt.format(
                error_type=error_context.get('error_type', 'Unknown'),
                error_message=error_context.get('error_message', ''),
                file_path=error_context.get('file_path', ''),
                line_number=error_context.get('line_number', 0),
                function_name=error_context.get('function_name', ''),
                analysis=json.dumps(self.make_serializable(analysis), indent=2),
                current_code=current_code
            )
            
            response = await self._generate_response(prompt)
            
            if response.error:
                self.logger.error(f"Fix generation failed: {response.error}")
                return None
            
            # Extract code from response
            fix_code = self._extract_code(response.content)
            
            if fix_code:
                self.logger.info(f"Generated fix code ({len(fix_code)} characters)")
                return fix_code
            else:
                self.logger.warning("No code found in LLM response")
                return None
                
        except Exception as e:
            self.logger.error(f"Fix generation failed: {e}")
            return None
    
    async def validate_fix(self, error_context: Dict[str, Any], fix_code: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a generated fix using LLM
        
        Args:
            error_context: Original error context
            fix_code: Generated fix code
            analysis: Previous analysis results
            
        Returns:
            Validation result
        """
        try:
            # Convert error_context to JSON-serializable format
            serializable_context = self.make_serializable(error_context)
            
            prompt = self.validation_prompt.format(
                error_context=json.dumps(serializable_context, indent=2),
                fix_code=fix_code,
                analysis=json.dumps(self.make_serializable(analysis), indent=2)
            )
            
            response = await self._generate_response(prompt)
            
            if response.error:
                return {
                    'fix_valid': False,
                    'reason': f'Validation failed: {response.error}'
                }
            
            # Parse JSON response
            try:
                validation = json.loads(response.content)
                return validation
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse validation as JSON, using fallback")
                return {
                    'fix_valid': True,
                    'success_likelihood': 0.7,
                    'recommendations': ['Manual review recommended']
                }
                
        except Exception as e:
            self.logger.error(f"Fix validation failed: {e}")
            return {
                'fix_valid': False,
                'reason': f'Validation failed: {e}'
            }
    
    async def _generate_response(self, prompt: str) -> LLMResponse:
        """Generate response from LLM with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                payload = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                }
                
                url = f"{self.config.base_url}/api/generate"
                
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return LLMResponse(
                            content=data.get('response', ''),
                            model=data.get('model', self.config.model),
                            created_at=datetime.now(),
                            done=data.get('done', True),
                            total_duration=data.get('total_duration', 0) / 1e9,  # Convert nanoseconds to seconds
                            prompt_eval_count=data.get('prompt_eval_count', 0),
                            eval_count=data.get('eval_count', 0)
                        )
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
                        self.logger.warning(f"LLM request failed (attempt {attempt + 1}): {last_error}")
                        
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                self.logger.warning(f"LLM request timeout (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"LLM request error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        return LLMResponse(
            content='',
            model=self.config.model,
            created_at=datetime.now(),
            done=False,
            total_duration=0,
            prompt_eval_count=0,
            eval_count=0,
            error=last_error
        )
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from LLM response"""
        # Look for code blocks
        code_block_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for any code block
        generic_code_pattern = r'```\n?(.*?)\n?```'
        matches = re.findall(generic_code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to extract single lines that look like code
        lines = response.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip obvious non-code lines
            if (line.startswith('#') or 
                line.startswith('Here') or 
                line.startswith('The') or 
                line.startswith('This') or
                not line or
                line.lower() in ['fix:', 'solution:', 'answer:']):
                continue
            
            # Include lines that look like code
            if (any(char in line for char in ['=', '(', ')', '[', ']', '{', '}', '.', ':']) or
                line.startswith(('def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except', 'with '))):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    async def _get_current_code(self, file_path: str) -> str:
        """Get current code from file for context"""
        try:
            if not file_path:
                return ""
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Limit content size to avoid token limits
            if len(content) > 2000:
                # Try to get relevant function or class
                lines = content.split('\n')
                return '\n'.join(lines[:50]) + "\n... (truncated)"
            
            return content
            
        except Exception as e:
            self.logger.warning(f"Failed to read current code from {file_path}: {e}")
            return ""
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            url = f"{self.config.base_url}/api/show"
            payload = {"name": self.config.model}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            url = f"{self.config.base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
                else:
                    return []
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def _resolve_model_name(self, requested_model: str, available_models: List[str]) -> str:
        """
        Resolve model name to an available model.
        
        Args:
            requested_model: The requested model name
            available_models: List of available models from Ollama
            
        Returns:
            Resolved model name or original if no match found
        """
        # Exact match
        if requested_model in available_models:
            return requested_model
        
        # Try to find partial matches for common patterns
        requested_lower = requested_model.lower()
        
        # Handle llama3 variations
        if 'llama3' in requested_lower:
            llama_models = [m for m in available_models if 'llama3' in m.lower()]
            if llama_models:
                # Prefer :latest tag if available
                latest_models = [m for m in llama_models if ':latest' in m]
                if latest_models:
                    return latest_models[0]
                # Otherwise return first match
                return llama_models[0]
        
        # Handle other common model patterns
        model_patterns = {
            'codellama': ['codellama'],
            'mistral': ['mistral'],
            'gemma': ['gemma'],
            'qwen': ['qwen'],
        }
        
        for pattern, possible_matches in model_patterns.items():
            if pattern in requested_lower:
                matching_models = [m for m in available_models 
                                if any(match in m.lower() for match in possible_matches)]
                if matching_models:
                    return matching_models[0]
        
        # No match found
        return requested_model
    
    async def shutdown(self):
        """Shutdown the LLM client"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.logger.info("LLM client shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during LLM client shutdown: {e}")