"""
Code Validator

Validation functionality for generated code including
syntax checking and basic logic validation.
"""

import ast
import asyncio
import importlib
import inspect
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger


class ValidationLevel(Enum):
    """Levels of code validation"""
    SYNTAX_ONLY = "syntax_only"
    IMPORT_CHECK = "import_check"
    BASIC_EXECUTION = "basic_execution"
    LOGIC_VALIDATION = "logic_validation"
    SECURITY_CHECK = "security_check"
    FULL_VALIDATION = "full_validation"


class ValidationStatus(Enum):
    """Validation status"""
    PENDING = "pending"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """Issue found during validation"""
    issue_type: str
    severity: str  # error, warning, info
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of code validation"""
    status: ValidationStatus
    is_valid: bool
    syntax_valid: bool
    imports_valid: bool
    execution_valid: bool
    security_valid: bool
    issues: List[ValidationIssue]
    validation_time: float
    test_results: Optional[Dict[str, Any]] = None
    suggestions: List[str] = None
    confidence_score: float = 0.0


class CodeValidator:
    """
    Comprehensive code validator for generated healing fixes
    
    This validator performs multiple levels of validation including syntax checking,
    import validation, basic execution testing, logic validation, and security checks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize code validator"""
        self.config = config or {}
        self.logger = get_logger("code_validator")
        
        # Configuration
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'full_validation'))
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.enable_logic_validation = self.config.get('enable_logic_validation', True)
        self.temp_dir = Path(self.config.get('temp_dir', 'data/temp/validation'))
        
        # Security patterns
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'open\s*\([\'"]\s*[\/]',  # File system access
            r'subprocess\.',
            r'os\.system',
            r'shutil\.',
            r'pickle\.loads?',
            r'marshal\.loads?',
        ]
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("CodeValidator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the validator"""
        try:
            # Test basic functionality
            test_code = "def test_function(): return True"
            result = await self.validate_code(test_code, "test.py")
            
            if result.status == ValidationStatus.PASSED:
                self.logger.info("CodeValidator initialization completed")
                return True
            else:
                self.logger.error("CodeValidator self-test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CodeValidator: {e}")
            return False
    
    async def validate_code(self, code: str, file_path: Optional[str] = None, 
                          context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate generated code comprehensively
        
        Args:
            code: Code to validate
            file_path: Original file path for context
            context: Additional context for validation
            
        Returns:
            ValidationResult with comprehensive validation results
        """
        start_time = time.time()
        issues = []
        
        try:
            self.logger.info(f"Starting code validation for {file_path or 'unknown file'}")
            
            # Step 1: Syntax validation
            syntax_valid, syntax_issues = await self._validate_syntax(code)
            issues.extend(syntax_issues)
            
            if not syntax_valid:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    syntax_valid=False,
                    imports_valid=False,
                    execution_valid=False,
                    security_valid=False,
                    issues=issues,
                    validation_time=time.time() - start_time,
                    confidence_score=0.0
                )
            
            # Step 2: Import validation
            imports_valid = True
            import_issues = []
            if self.validation_level.value in ['import_check', 'basic_execution', 'logic_validation', 'security_check', 'full_validation']:
                imports_valid, import_issues = await self._validate_imports(code)
                issues.extend(import_issues)
            
            # Step 3: Security validation
            security_valid = True
            security_issues = []
            if self.enable_security_checks and self.validation_level.value in ['security_check', 'full_validation']:
                security_valid, security_issues = await self._validate_security(code)
                issues.extend(security_issues)
            
            # Step 4: Basic execution validation
            execution_valid = True
            execution_issues = []
            test_results = None
            if self.validation_level.value in ['basic_execution', 'logic_validation', 'full_validation']:
                execution_valid, execution_issues, test_results = await self._validate_execution(code, context)
                issues.extend(execution_issues)
            
            # Step 5: Logic validation
            logic_valid = True
            logic_issues = []
            if self.enable_logic_validation and self.validation_level.value in ['logic_validation', 'full_validation']:
                logic_valid, logic_issues = await self._validate_logic(code, context)
                issues.extend(logic_issues)
            
            # Calculate overall validity
            is_valid = syntax_valid and imports_valid and execution_valid and security_valid and logic_valid
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                syntax_valid, imports_valid, execution_valid, security_valid, logic_valid
            )
            
            # Determine status
            if is_valid:
                status = ValidationStatus.PASSED
            elif syntax_valid and imports_valid:
                status = ValidationStatus.WARNING  # Passed basic checks but has issues
            else:
                status = ValidationStatus.FAILED
            
            # Generate suggestions
            suggestions = self._generate_suggestions(issues)
            
            validation_time = time.time() - start_time
            
            self.logger.info(f"Validation completed in {validation_time:.3f}s - Status: {status.value}")
            
            return ValidationResult(
                status=status,
                is_valid=is_valid,
                syntax_valid=syntax_valid,
                imports_valid=imports_valid,
                execution_valid=execution_valid,
                security_valid=security_valid,
                issues=issues,
                validation_time=validation_time,
                test_results=test_results,
                suggestions=suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                syntax_valid=False,
                imports_valid=False,
                execution_valid=False,
                security_valid=False,
                issues=[ValidationIssue(
                    issue_type="validation_error",
                    severity="error",
                    message=f"Validation failed: {e}"
                )],
                validation_time=time.time() - start_time,
                confidence_score=0.0
            )
    
    async def _validate_syntax(self, code: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate Python syntax"""
        issues = []
        
        try:
            ast.parse(code)
            return True, issues
            
        except SyntaxError as e:
            issues.append(ValidationIssue(
                issue_type="syntax_error",
                severity="error",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column_number=e.offset,
                code_snippet=self._get_code_snippet(code, e.lineno),
                suggestion="Fix the syntax error and try again"
            ))
            return False, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="parse_error",
                severity="error",
                message=f"Parse error: {e}",
                suggestion="Check for invalid Python syntax"
            ))
            return False, issues
    
    async def _validate_imports(self, code: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate that all imports are available"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if not await self._is_import_available(module_name):
                            issues.append(ValidationIssue(
                                issue_type="import_error",
                                severity="error",
                                message=f"Module '{module_name}' not available",
                                line_number=node.lineno,
                                suggestion=f"Install module '{module_name}' or check import spelling"
                            ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if not await self._is_import_available(node.module):
                            issues.append(ValidationIssue(
                                issue_type="import_error",
                                severity="error",
                                message=f"Module '{node.module}' not available",
                                line_number=node.lineno,
                                suggestion=f"Install module '{node.module}' or check import spelling"
                            ))
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="import_validation_error",
                severity="error",
                message=f"Import validation failed: {e}",
                suggestion="Check import statements for syntax errors"
            ))
            return False, issues
    
    async def _validate_security(self, code: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate code for security issues"""
        issues = []
        
        try:
            import re
            
            for pattern in self.dangerous_patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append(ValidationIssue(
                        issue_type="security_warning",
                        severity="warning",
                        message=f"Potentially dangerous code pattern detected: {match.group()}",
                        line_number=line_num,
                        code_snippet=self._get_code_snippet(code, line_num),
                        suggestion="Review this code for security implications"
                    ))
            
            # Check for hardcoded secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append(ValidationIssue(
                        issue_type="security_warning",
                        severity="warning",
                        message=f"Potential hardcoded secret detected",
                        line_number=line_num,
                        code_snippet=self._get_code_snippet(code, line_num),
                        suggestion="Use environment variables or configuration files for secrets"
                    ))
            
            return len([i for i in issues if i.severity == "error"]) == 0, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="security_validation_error",
                severity="error",
                message=f"Security validation failed: {e}",
                suggestion="Manual security review recommended"
            ))
            return False, issues
    
    async def _validate_execution(self, code: str, context: Optional[Dict[str, Any]]) -> Tuple[bool, List[ValidationIssue], Optional[Dict[str, Any]]]:
        """Validate code by attempting to execute it"""
        issues = []
        test_results = None
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.temp_dir) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Try to compile the code
                import py_compile
                py_compile.compile(temp_file, doraise=True)
                
                # Try to execute in a controlled environment
                test_results = await self._execute_code_safely(temp_file, context)
                
                if test_results.get('execution_success', False):
                    return True, issues, test_results
                else:
                    issues.append(ValidationIssue(
                        issue_type="execution_error",
                        severity="error",
                        message=f"Code execution failed: {test_results.get('error', 'Unknown error')}",
                        suggestion="Debug the code logic and fix execution errors"
                    ))
                    return False, issues, test_results
                    
            finally:
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_file)
                except:
                    pass
                    
        except py_compile.PyCompileError as e:
            issues.append(ValidationIssue(
                issue_type="compilation_error",
                severity="error",
                message=f"Code compilation failed: {e}",
                suggestion="Fix syntax and logic errors"
            ))
            return False, issues, test_results
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="execution_validation_error",
                severity="error",
                message=f"Execution validation failed: {e}",
                suggestion="Check code for runtime errors"
            ))
            return False, issues, test_results
    
    async def _validate_logic(self, code: str, context: Optional[Dict[str, Any]]) -> Tuple[bool, List[ValidationIssue]]:
        """Validate code logic and structure"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for common logic issues
            for node in ast.walk(tree):
                # Check for infinite loops
                if isinstance(node, ast.While):
                    if (isinstance(node.test, ast.NameConstant) and 
                        node.test.value is True):
                        issues.append(ValidationIssue(
                            issue_type="logic_warning",
                            severity="warning",
                            message="Potential infinite loop detected",
                            line_number=node.lineno,
                            suggestion="Add break condition or timeout"
                        ))
                
                # Check for unhandled exceptions
                elif isinstance(node, ast.Try):
                    if not node.handlers:
                        issues.append(ValidationIssue(
                            issue_type="logic_warning",
                            severity="warning",
                            message="Try block without exception handlers",
                            line_number=node.lineno,
                            suggestion="Add appropriate exception handlers"
                        ))
                
                # Check for bare except
                elif isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        issues.append(ValidationIssue(
                            issue_type="logic_warning",
                            severity="warning",
                            message="Bare except clause detected",
                            line_number=node.lineno,
                            suggestion="Specify exception types for better error handling"
                        ))
            
            # Check function complexity
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            for func in functions:
                complexity = self._calculate_cyclomatic_complexity(func)
                if complexity > 10:
                    issues.append(ValidationIssue(
                        issue_type="complexity_warning",
                        severity="warning",
                        message=f"Function '{func.name}' has high complexity ({complexity})",
                        line_number=func.lineno,
                        suggestion="Consider breaking down the function into smaller functions"
                    ))
            
            return len([i for i in issues if i.severity == "error"]) == 0, issues
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="logic_validation_error",
                severity="error",
                message=f"Logic validation failed: {e}",
                suggestion="Manual code review recommended"
            ))
            return False, issues
    
    async def _is_import_available(self, module_name: str) -> bool:
        """Check if a module is available for import"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except Exception:
            # Other import errors might mean the module exists but has issues
            return True
    
    def _get_code_snippet(self, code: str, line_number: int, context_lines: int = 2) -> str:
        """Get code snippet around a specific line"""
        lines = code.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        snippet_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            snippet_lines.append(f"{prefix}{lines[i]}")
        
        return '\n'.join(snippet_lines)
    
    def _calculate_cyclomatic_complexity(self, func_node) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_confidence_score(self, syntax_valid: bool, imports_valid: bool, 
                                 execution_valid: bool, security_valid: bool, 
                                 logic_valid: bool) -> float:
        """Calculate overall confidence score"""
        weights = {
            'syntax': 0.3,
            'imports': 0.2,
            'execution': 0.25,
            'security': 0.15,
            'logic': 0.1
        }
        
        score = 0.0
        if syntax_valid:
            score += weights['syntax']
        if imports_valid:
            score += weights['imports']
        if execution_valid:
            score += weights['execution']
        if security_valid:
            score += weights['security']
        if logic_valid:
            score += weights['logic']
        
        return score
    
    def _generate_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate suggestions based on validation issues"""
        suggestions = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate suggestions for each issue type
        if 'syntax_error' in issue_types:
            suggestions.append("Fix syntax errors before proceeding")
        
        if 'import_error' in issue_types:
            suggestions.append("Install missing dependencies or fix import statements")
        
        if 'security_warning' in issue_types:
            suggestions.append("Review and address security concerns")
        
        if 'logic_warning' in issue_types:
            suggestions.append("Improve code logic and structure")
        
        if 'complexity_warning' in issue_types:
            suggestions.append("Consider refactoring complex functions")
        
        return suggestions
    
    async def _execute_code_safely(self, file_path: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute code in a safe environment"""
        try:
            # Create a safe namespace for execution
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                }
            }
            
            # Add context variables if provided
            if context:
                safe_globals.update(context)
            
            # Execute the code
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Use exec with timeout
            exec_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: exec(code, safe_globals)
                ),
                timeout=self.timeout_seconds
            )
            
            return {
                'execution_success': True,
                'result': exec_result,
                'namespace': safe_globals
            }
            
        except asyncio.TimeoutError:
            return {
                'execution_success': False,
                'error': 'Code execution timed out'
            }
        except Exception as e:
            return {
                'execution_success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        # This would track validation history if implemented
        return {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0,
            'common_issues': [],
            'validation_level': self.validation_level.value
        }
    
    async def shutdown(self):
        """Shutdown the validator"""
        try:
            # Clean up temporary files
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*.py"):
                    try:
                        temp_file.unlink()
                    except:
                        pass
            
            self.logger.info("CodeValidator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")