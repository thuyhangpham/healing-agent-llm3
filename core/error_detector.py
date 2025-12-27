"""
Error Detection and Analysis System

This module provides comprehensive error detection and analysis capabilities
for the self-healing system, with specialized focus on web scraping failures
and runtime errors.

Features:
- HTML/CSS change detection algorithms
- Error categorization and prioritization
- Failure pattern analysis and learning
- Web scraping error simulation for testing
- Error similarity detection and clustering
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter

import aiohttp
from bs4 import BeautifulSoup
from utils.logger import get_logger


class ErrorCategory(Enum):
    """Error categories for classification"""
    WEB_SCRAPING_FAILURE = "web_scraping_failure"
    HTML_STRUCTURE_CHANGED = "html_structure_changed"
    CSS_SELECTOR_FAILURE = "css_selector_failure"
    JAVASCRIPT_ERROR = "javascript_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ErrorPattern:
    """Pattern of recurring errors"""
    pattern_id: str
    error_type: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    affected_agents: List[str]
    common_context: Dict[str, Any]
    suggested_fixes: List[str]
    success_rate: float = 0.0


@dataclass
class HTMLDiff:
    """Difference between HTML snapshots"""
    added_elements: List[str]
    removed_elements: List[str]
    modified_elements: List[str]
    css_changes: List[str]
    structural_changes: List[str]
    similarity_score: float


@dataclass
class ErrorAnalysis:
    """Complete analysis of an error"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    repairable: bool
    confidence: float
    root_cause: str
    suggested_approach: str
    html_diff: Optional[HTMLDiff] = None
    similar_errors: List[str] = None
    pattern_matches: List[str] = None
    required_changes: List[str] = None
    estimated_fix_time: float = 0.0


class ErrorDetector:
    """
    Error detection and analysis system for self-healing
    
    This class provides comprehensive error analysis capabilities including
    HTML/CSS change detection, error categorization, pattern recognition,
    and repairability assessment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize error detector"""
        self.config = config or {}
        self.logger = get_logger("error_detector")
        
        # Error patterns storage
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.html_snapshots: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.pattern_min_frequency = self.config.get('pattern_min_frequency', 3)
        self.html_diff_threshold = self.config.get('html_diff_threshold', 0.8)
        self.max_history_size = self.config.get('max_history_size', 10000)
        
        # Error classification patterns
        self.error_patterns_regex = {
            ErrorCategory.WEB_SCRAPING_FAILURE: [
                r'no element found',
                r'element not found',
                r'failed to locate',
                r'selector.*not found',
                r'css.*selector.*failed'
            ],
            ErrorCategory.HTML_STRUCTURE_CHANGED: [
                r'html.*structure.*changed',
                r'dom.*changed',
                r'page.*layout.*changed',
                r'element.*structure.*modified'
            ],
            ErrorCategory.CSS_SELECTOR_FAILURE: [
                r'css.*selector',
                r'invalid selector',
                r'selector.*syntax',
                r'css.*path.*not.*found'
            ],
            ErrorCategory.NETWORK_ERROR: [
                r'connection.*refused',
                r'network.*unreachable',
                r'dns.*resolution.*failed',
                r'connection.*timeout'
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                r'timeout',
                r'timed.*out',
                r'operation.*timed'
            ],
            ErrorCategory.AUTHENTICATION_ERROR: [
                r'unauthorized',
                r'authentication.*failed',
                r'access.*denied',
                r'401|403'
            ],
            ErrorCategory.RATE_LIMIT_ERROR: [
                r'rate.*limit',
                r'too.*many.*requests',
                r'429',
                r'throttled'
            ],
            ErrorCategory.SYNTAX_ERROR: [
                r'syntax.*error',
                r'invalid.*syntax',
                r'parse.*error'
            ],
            ErrorCategory.IMPORT_ERROR: [
                r'import.*error',
                r'module.*not.*found',
                r'cannot.*import'
            ],
            ErrorCategory.TYPE_ERROR: [
                r'type.*error',
                r'expected.*type',
                r'invalid.*type'
            ],
            ErrorCategory.ATTRIBUTE_ERROR: [
                r'attribute.*error',
                r'has.*no.*attribute',
                r'object.*has.*no.*attribute'
            ],
            ErrorCategory.KEY_ERROR: [
                r'key.*error',
                r'key.*not.*found',
                r'missing.*key'
            ],
            ErrorCategory.INDEX_ERROR: [
                r'index.*error',
                r'index.*out.*of.*range',
                r'list.*index'
            ]
        }
        
        # Initialize storage
        self._load_patterns()
        
        self.logger.info("ErrorDetector initialized")
    
    async def initialize(self) -> bool:
        """Initialize the error detector"""
        try:
            # Load existing patterns and history
            await self._load_data()
            
            # Start background tasks for pattern learning
            asyncio.create_task(self._pattern_learning_loop())
            
            self.logger.info("ErrorDetector initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ErrorDetector: {e}")
            return False
    
    async def analyze_error(self, error_context) -> Dict[str, Any]:
        """
        Analyze an error and provide comprehensive assessment
        
        Args:
            error_context: ErrorContext object with error information
            
        Returns:
            Analysis result with repairability assessment
        """
        try:
            start_time = time.time()
            
            # Extract error information
            error_message = error_context.error_message
            traceback_str = error_context.traceback_str
            error_type = error_context.error_type
            
            # Categorize error
            category = self._categorize_error(error_message, traceback_str, error_type)
            
            # Determine severity
            severity = self._determine_severity(category, error_context)
            
            # Analyze HTML/CSS changes if available
            html_diff = None
            if error_context.html_snapshot:
                html_diff = await self._analyze_html_changes(error_context)
            
            # Find similar errors
            similar_errors = self._find_similar_errors(error_context)
            
            # Match against known patterns
            pattern_matches = self._match_patterns(error_context)
            
            # Assess repairability
            repairable, confidence = self._assess_repairability(
                category, severity, html_diff, pattern_matches
            )
            
            # Generate root cause analysis
            root_cause = self._analyze_root_cause(
                category, error_context, html_diff, pattern_matches
            )
            
            # Suggest approach
            suggested_approach = self._suggest_approach(
                category, repairable, root_cause, pattern_matches
            )
            
            # Estimate fix time
            estimated_fix_time = self._estimate_fix_time(category, severity, repairable)
            
            # Create analysis result
            analysis = ErrorAnalysis(
                error_id=error_context.error_id,
                category=category,
                severity=severity,
                repairable=repairable,
                confidence=confidence,
                root_cause=root_cause,
                suggested_approach=suggested_approach,
                html_diff=html_diff,
                similar_errors=similar_errors,
                pattern_matches=pattern_matches,
                required_changes=self._get_required_changes(category, root_cause),
                estimated_fix_time=estimated_fix_time
            )
            
            # Store in history
            await self._store_error_analysis(error_context, analysis)
            
            # Update patterns
            await self._update_patterns(error_context, analysis)
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Error analysis completed in {analysis_time:.3f}s")
            
            return asdict(analysis)
            
        except Exception as e:
            self.logger.error(f"Error analysis failed: {e}")
            return {
                'repairable': False,
                'reason': f'Analysis failed: {e}',
                'category': ErrorCategory.UNKNOWN_ERROR.value,
                'severity': ErrorSeverity.MEDIUM.value,
                'confidence': 0.0
            }
    
    def _categorize_error(self, error_message: str, traceback_str: str, error_type: str) -> ErrorCategory:
        """Categorize error based on message and traceback"""
        combined_text = f"{error_message} {traceback_str} {error_type}".lower()
        
        # Check each category
        for category, patterns in self.error_patterns_regex.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return category
        
        # Additional heuristics
        if 'scraping' in combined_text or 'beautifulsoup' in combined_text:
            return ErrorCategory.WEB_SCRAPING_FAILURE
        
        if 'html' in combined_text or 'css' in combined_text:
            return ErrorCategory.HTML_STRUCTURE_CHANGED
        
        if 'selector' in combined_text:
            return ErrorCategory.CSS_SELECTOR_FAILURE
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(self, category: ErrorCategory, error_context) -> ErrorSeverity:
        """Determine error severity based on category and context"""
        # Critical categories
        if category in [ErrorCategory.AUTHENTICATION_ERROR, ErrorCategory.RATE_LIMIT_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity categories
        if category in [ErrorCategory.WEB_SCRAPING_FAILURE, ErrorCategory.HTML_STRUCTURE_CHANGED]:
            return ErrorSeverity.HIGH
        
        # Medium severity categories
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.TIMEOUT_ERROR, 
                       ErrorCategory.CSS_SELECTOR_FAILURE]:
            return ErrorSeverity.MEDIUM
        
        # Check error context for severity indicators
        error_message = error_context.error_message.lower()
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'emergency']):
            return ErrorSeverity.CRITICAL
        
        if any(keyword in error_message for keyword in ['warning', 'deprecated']):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    async def _analyze_html_changes(self, error_context) -> Optional[HTMLDiff]:
        """Analyze HTML changes between snapshots"""
        try:
            current_html = error_context.html_snapshot
            if not current_html:
                return None
            
            # Get previous snapshot for comparison
            url = error_context.additional_context.get('url', 'unknown')
            previous_snapshot = self.html_snapshots.get(url)
            
            if not previous_snapshot:
                # Store current snapshot for future comparison
                self.html_snapshots[url] = {
                    'html': current_html,
                    'timestamp': error_context.timestamp,
                    'hash': hashlib.md5(current_html.encode()).hexdigest()
                }
                return None
            
            # Compare HTML
            previous_html = previous_snapshot['html']
            similarity = SequenceMatcher(None, previous_html, current_html).ratio()
            
            if similarity > self.html_diff_threshold:
                return None  # No significant changes
            
            # Parse HTML for detailed comparison
            current_soup = BeautifulSoup(current_html, 'html.parser')
            previous_soup = BeautifulSoup(previous_html, 'html.parser')
            
            # Find differences
            added_elements = []
            removed_elements = []
            modified_elements = []
            
            # Simple element comparison (can be enhanced)
            current_elements = {tag.name for tag in current_soup.find_all()}
            previous_elements = {tag.name for tag in previous_soup.find_all()}
            
            added_elements = list(current_elements - previous_elements)
            removed_elements = list(previous_elements - current_elements)
            
            # CSS changes detection
            css_changes = []
            current_css = self._extract_css_info(current_soup)
            previous_css = self._extract_css_info(previous_soup)
            
            if current_css != previous_css:
                css_changes = ["CSS structure changed"]
            
            # Structural changes
            structural_changes = []
            if len(current_soup.find_all()) != len(previous_soup.find_all()):
                structural_changes.append(f"Element count changed: {len(previous_soup.find_all())} → {len(current_soup.find_all())}")
            
            # Update stored snapshot
            self.html_snapshots[url] = {
                'html': current_html,
                'timestamp': error_context.timestamp,
                'hash': hashlib.md5(current_html.encode()).hexdigest()
            }
            
            return HTMLDiff(
                added_elements=added_elements,
                removed_elements=removed_elements,
                modified_elements=modified_elements,
                css_changes=css_changes,
                structural_changes=structural_changes,
                similarity_score=similarity
            )
            
        except Exception as e:
            self.logger.warning(f"HTML analysis failed: {e}")
            return None
    
    def _extract_css_info(self, soup) -> Dict[str, Any]:
        """Extract CSS information from BeautifulSoup object"""
        css_info = {}
        
        # Extract style tags
        style_tags = soup.find_all('style')
        css_info['style_tags'] = len(style_tags)
        
        # Extract inline styles
        elements_with_style = soup.find_all(attrs={'style': True})
        css_info['inline_styles'] = len(elements_with_style)
        
        # Extract class names
        classes = set()
        for tag in soup.find_all(class_=True):
            classes.update(tag['class'])
        css_info['classes'] = list(classes)
        
        return css_info
    
    def _find_similar_errors(self, error_context) -> List[str]:
        """Find similar errors in history"""
        similar_errors = []
        
        for historical_error in self.error_history[-100:]:  # Check last 100 errors
            similarity = self._calculate_error_similarity(error_context, historical_error)
            if similarity > self.similarity_threshold:
                similar_errors.append(historical_error['error_id'])
        
        return similar_errors[:10]  # Return top 10 similar errors
    
    def _calculate_error_similarity(self, error_context, historical_error: Dict[str, Any]) -> float:
        """Calculate similarity between current error and historical error"""
        # Message similarity
        message_similarity = SequenceMatcher(
            None, 
            error_context.error_message.lower(), 
            historical_error.get('error_message', '').lower()
        ).ratio()
        
        # Type similarity
        type_similarity = 1.0 if error_context.error_type == historical_error.get('error_type') else 0.0
        
        # Agent similarity
        agent_similarity = 1.0 if error_context.agent_name == historical_error.get('agent_name') else 0.0
        
        # Weighted average
        return (message_similarity * 0.6 + type_similarity * 0.3 + agent_similarity * 0.1)
    
    def _match_patterns(self, error_context) -> List[str]:
        """Match error against known patterns"""
        pattern_matches = []
        
        for pattern_id, pattern in self.error_patterns.items():
            if self._matches_pattern(error_context, pattern):
                pattern_matches.append(pattern_id)
        
        return pattern_matches
    
    def _matches_pattern(self, error_context, pattern: ErrorPattern) -> bool:
        """Check if error matches a pattern"""
        # Check error type
        if error_context.error_type != pattern.error_type:
            return False
        
        # Check agent
        if error_context.agent_name not in pattern.affected_agents:
            return False
        
        # Check context similarity (simplified)
        context_similarity = SequenceMatcher(
            None,
            str(error_context.additional_context),
            str(pattern.common_context)
        ).ratio()
        
        return context_similarity > 0.5
    
    def _assess_repairability(self, category: ErrorCategory, severity: ErrorSeverity, 
                            html_diff: Optional[HTMLDiff], pattern_matches: List[str]) -> Tuple[bool, float]:
        """Assess if error is repairable and confidence level"""
        # Highly repairable categories
        if category in [ErrorCategory.CSS_SELECTOR_FAILURE, ErrorCategory.WEB_SCRAPING_FAILURE]:
            return True, 0.9
        
        # Moderately repairable categories
        if category in [ErrorCategory.HTML_STRUCTURE_CHANGED, ErrorCategory.SYNTAX_ERROR]:
            return True, 0.7
        
        # Low repairability categories
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.AUTHENTICATION_ERROR]:
            return False, 0.3
        
        # Pattern-based assessment
        if pattern_matches:
            pattern = self.error_patterns[pattern_matches[0]]
            if pattern.success_rate > 0.8:
                return True, pattern.success_rate
        
        # HTML changes indicate repairability
        if html_diff and html_diff.similarity_score < 0.5:
            return True, 0.8
        
        # Default assessment
        return False, 0.5
    
    def _analyze_root_cause(self, category: ErrorCategory, error_context, 
                           html_diff: Optional[HTMLDiff], pattern_matches: List[str]) -> str:
        """Analyze root cause of error"""
        if html_diff:
            return f"HTML structure changed (similarity: {html_diff.similarity_score:.2f})"
        
        if pattern_matches:
            pattern = self.error_patterns[pattern_matches[0]]
            return f"Recurring pattern: {pattern.pattern_id}"
        
        # Category-specific root causes
        root_causes = {
            ErrorCategory.CSS_SELECTOR_FAILURE: "CSS selectors no longer match page elements",
            ErrorCategory.WEB_SCRAPING_FAILURE: "Web page structure or content changed",
            ErrorCategory.HTML_STRUCTURE_CHANGED: "HTML DOM structure modified",
            ErrorCategory.NETWORK_ERROR: "Network connectivity issues",
            ErrorCategory.TIMEOUT_ERROR: "Operation exceeded time limit",
            ErrorCategory.AUTHENTICATION_ERROR: "Authentication credentials invalid or expired",
            ErrorCategory.RATE_LIMIT_ERROR: "API rate limit exceeded",
            ErrorCategory.SYNTAX_ERROR: "Code syntax error detected",
            ErrorCategory.IMPORT_ERROR: "Module import failure",
            ErrorCategory.TYPE_ERROR: "Data type mismatch",
            ErrorCategory.ATTRIBUTE_ERROR: "Object attribute not found",
            ErrorCategory.KEY_ERROR: "Dictionary key not found",
            ErrorCategory.INDEX_ERROR: "List index out of range"
        }
        
        return root_causes.get(category, "Unknown root cause")
    
    def _suggest_approach(self, category: ErrorCategory, repairable: bool, 
                         root_cause: str, pattern_matches: List[str]) -> str:
        """Suggest approach for fixing the error"""
        if not repairable:
            return "Manual intervention required - error not automatically repairable"
        
        if pattern_matches:
            pattern = self.error_patterns[pattern_matches[0]]
            if pattern.suggested_fixes:
                return f"Apply known fix pattern: {pattern.suggested_fixes[0]}"
        
        approaches = {
            ErrorCategory.CSS_SELECTOR_FAILURE: "Update CSS selectors to match current page structure",
            ErrorCategory.WEB_SCRAPING_FAILURE: "Modify scraping logic to handle new page structure",
            ErrorCategory.HTML_STRUCTURE_CHANGED: "Adapt parsing logic to new HTML structure",
            ErrorCategory.SYNTAX_ERROR: "Fix syntax errors in code",
            ErrorCategory.IMPORT_ERROR: "Update import statements or install missing modules",
            ErrorCategory.TYPE_ERROR: "Add type conversion or validation",
            ErrorCategory.ATTRIBUTE_ERROR: "Add attribute existence checks or use getattr()",
            ErrorCategory.KEY_ERROR: "Add key existence checks or use dict.get()",
            ErrorCategory.INDEX_ERROR: "Add bounds checking or use try/except"
        }
        
        return approaches.get(category, "Analyze error context and generate appropriate fix")
    
    def _get_required_changes(self, category: ErrorCategory, root_cause: str) -> List[str]:
        """Get list of required changes to fix the error"""
        changes = {
            ErrorCategory.CSS_SELECTOR_FAILURE: ["Update CSS selectors", "Test selector validity"],
            ErrorCategory.WEB_SCRAPING_FAILURE: ["Modify parsing logic", "Update element selection"],
            ErrorCategory.HTML_STRUCTURE_CHANGED: ["Adapt DOM traversal", "Update element locators"],
            ErrorCategory.SYNTAX_ERROR: ["Fix syntax", "Validate code structure"],
            ErrorCategory.IMPORT_ERROR: ["Update imports", "Install dependencies"],
            ErrorCategory.TYPE_ERROR: ["Add type conversion", "Implement type checking"],
            ErrorCategory.ATTRIBUTE_ERROR: ["Add attribute checks", "Use safe attribute access"],
            ErrorCategory.KEY_ERROR: ["Add key validation", "Use default values"],
            ErrorCategory.INDEX_ERROR: ["Add bounds checking", "Use safe indexing"]
        }
        
        return changes.get(category, ["Analyze and implement appropriate fix"])
    
    def _estimate_fix_time(self, category: ErrorCategory, severity: ErrorSeverity, repairable: bool) -> float:
        """Estimate time required to fix error (in seconds)"""
        if not repairable:
            return 300.0  # 5 minutes for manual intervention
        
        base_times = {
            ErrorCategory.CSS_SELECTOR_FAILURE: 30.0,
            ErrorCategory.WEB_SCRAPING_FAILURE: 45.0,
            ErrorCategory.HTML_STRUCTURE_CHANGED: 60.0,
            ErrorCategory.SYNTAX_ERROR: 20.0,
            ErrorCategory.IMPORT_ERROR: 25.0,
            ErrorCategory.TYPE_ERROR: 35.0,
            ErrorCategory.ATTRIBUTE_ERROR: 30.0,
            ErrorCategory.KEY_ERROR: 25.0,
            ErrorCategory.INDEX_ERROR: 25.0
        }
        
        base_time = base_times.get(category, 60.0)
        
        # Adjust for severity
        severity_multiplier = {
            ErrorSeverity.LOW: 0.5,
            ErrorSeverity.MEDIUM: 1.0,
            ErrorSeverity.HIGH: 1.5,
            ErrorSeverity.CRITICAL: 2.0
        }
        
        return base_time * severity_multiplier.get(severity, 1.0)
    
    async def _store_error_analysis(self, error_context, analysis: ErrorAnalysis):
        """Store error analysis in history"""
        try:
            error_record = {
                'error_id': error_context.error_id,
                'timestamp': error_context.timestamp.isoformat(),
                'error_type': error_context.error_type,
                'error_message': error_context.error_message,
                'agent_name': error_context.agent_name,
                'category': analysis.category.value,
                'severity': analysis.severity.value,
                'repairable': analysis.repairable,
                'confidence': analysis.confidence,
                'root_cause': analysis.root_cause,
                'suggested_approach': analysis.suggested_approach
            }
            
            self.error_history.append(error_record)
            
            # Limit history size
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # Save to disk periodically
            if len(self.error_history) % 100 == 0:
                await self._save_data()
                
        except Exception as e:
            self.logger.error(f"Failed to store error analysis: {e}")
    
    async def _update_patterns(self, error_context, analysis: ErrorAnalysis):
        """Update error patterns based on new analysis"""
        try:
            # Create pattern key
            pattern_key = f"{analysis.category.value}:{error_context.agent_name}:{error_context.error_type}"
            
            if pattern_key not in self.error_patterns:
                self.error_patterns[pattern_key] = ErrorPattern(
                    pattern_id=pattern_key,
                    error_type=error_context.error_type,
                    frequency=0,
                    first_seen=error_context.timestamp,
                    last_seen=error_context.timestamp,
                    affected_agents=[error_context.agent_name],
                    common_context=error_context.additional_context,
                    suggested_fixes=[analysis.suggested_approach],
                    success_rate=0.0
                )
            
            # Update pattern
            pattern = self.error_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_seen = error_context.timestamp
            
            if error_context.agent_name not in pattern.affected_agents:
                pattern.affected_agents.append(error_context.agent_name)
            
            # Update success rate based on repairability
            if analysis.repairable:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + 1.0) / pattern.frequency
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1)) / pattern.frequency
            
        except Exception as e:
            self.logger.error(f"Failed to update patterns: {e}")
    
    async def _pattern_learning_loop(self):
        """Background task for pattern learning and optimization"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze patterns for optimization
                await self._optimize_patterns()
                
                # Clean old data
                await self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Pattern learning loop error: {e}")
    
    async def _optimize_patterns(self):
        """Optimize error patterns based on recent data"""
        try:
            # Remove patterns with low frequency
            patterns_to_remove = []
            for pattern_id, pattern in self.error_patterns.items():
                if pattern.frequency < self.pattern_min_frequency:
                    age_days = (datetime.now() - pattern.first_seen).days
                    if age_days > 7:  # Remove old patterns with low frequency
                        patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                del self.error_patterns[pattern_id]
                self.logger.info(f"Removed low-frequency pattern: {pattern_id}")
            
            # Merge similar patterns
            await self._merge_similar_patterns()
            
        except Exception as e:
            self.logger.error(f"Pattern optimization failed: {e}")
    
    async def _merge_similar_patterns(self):
        """Merge similar error patterns"""
        # Implementation for pattern merging
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # Clean old error history
            self.error_history = [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > cutoff_date
            ]
            
            # Clean old HTML snapshots
            urls_to_remove = []
            for url, snapshot in self.html_snapshots.items():
                snapshot_date = datetime.fromisoformat(snapshot['timestamp'])
                if snapshot_date < cutoff_date:
                    urls_to_remove.append(url)
            
            for url in urls_to_remove:
                del self.html_snapshots[url]
            
            self.logger.info("Cleaned up old data")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    async def _load_data(self):
        """Load existing patterns and history from disk"""
        try:
            # Load patterns
            patterns_file = Path("data/error_patterns.json")
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    for pattern_id, pattern_data in patterns_data.items():
                        pattern_data['first_seen'] = datetime.fromisoformat(pattern_data['first_seen'])
                        pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                        self.error_patterns[pattern_id] = ErrorPattern(**pattern_data)
            
            # Load history
            history_file = Path("data/error_history.json")
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.error_history = json.load(f)
            
            # Load HTML snapshots
            snapshots_file = Path("data/html_snapshots.json")
            if snapshots_file.exists():
                with open(snapshots_file, 'r', encoding='utf-8') as f:
                    snapshots_data = json.load(f)
                    for url, snapshot in snapshots_data.items():
                        snapshot['timestamp'] = datetime.fromisoformat(snapshot['timestamp'])
                        self.html_snapshots[url] = snapshot
            
            self.logger.info(f"Loaded {len(self.error_patterns)} patterns and {len(self.error_history)} error records")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing data: {e}")
    
    async def _save_data(self):
        """Save patterns and history to disk"""
        try:
            # Ensure data directory exists
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Save patterns
            patterns_data = {}
            for pattern_id, pattern in self.error_patterns.items():
                pattern_dict = asdict(pattern)
                pattern_dict['first_seen'] = pattern.first_seen.isoformat()
                pattern_dict['last_seen'] = pattern.last_seen.isoformat()
                patterns_data[pattern_id] = pattern_dict
            
            with open(data_dir / "error_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save history
            with open(data_dir / "error_history.json", 'w', encoding='utf-8') as f:
                json.dump(self.error_history, f, indent=2)
            
            # Save HTML snapshots
            snapshots_data = {}
            for url, snapshot in self.html_snapshots.items():
                snapshot_dict = snapshot.copy()
                snapshot_dict['timestamp'] = snapshot['timestamp'].isoformat()
                snapshots_data[url] = snapshot_dict
            
            with open(data_dir / "html_snapshots.json", 'w', encoding='utf-8') as f:
                json.dump(snapshots_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
    
    def _load_patterns(self):
        """Load initial error patterns"""
        # Can be extended to load predefined patterns
        pass
    
    async def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all error patterns"""
        return {
            pattern_id: asdict(pattern) 
            for pattern_id, pattern in self.error_patterns.items()
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get error detection statistics"""
        total_errors = len(self.error_history)
        category_counts = Counter(error.get('category', 'unknown') for error in self.error_history)
        severity_counts = Counter(error.get('severity', 1) for error in self.error_history)
        
        repairable_count = sum(1 for error in self.error_history if error.get('repairable', False))
        repairable_rate = repairable_count / total_errors if total_errors > 0 else 0
        
        return {
            'total_errors': total_errors,
            'total_patterns': len(self.error_patterns),
            'category_distribution': dict(category_counts),
            'severity_distribution': dict(severity_counts),
            'repairable_rate': repairable_rate,
            'html_snapshots': len(self.html_snapshots)
        }
    
    async def shutdown(self):
        """Shutdown the error detector"""
        try:
            # Save data before shutdown
            await self._save_data()
            self.logger.info("ErrorDetector shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")