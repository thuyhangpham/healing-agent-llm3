"""
Code Patching and Hot-Reload System

This module provides hot-reload capabilities for applying code fixes
without system downtime, including backup creation, validation,
and rollback mechanisms.

Features:
- Importlib.reload() for runtime module replacement
- Code validation and syntax checking
- Backup creation (.bak files) before patching
- Safe patch application with rollback capability
- Concurrent operation handling during patching
- Atomic file operations for safety
"""

import asyncio
import ast
import importlib
import inspect
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import traceback

from utils.logger import get_logger


class PatchStatus(Enum):
    """Status of code patching operations"""
    PENDING = "pending"
    VALIDATING = "validating"
    APPLYING = "applying"
    RELOADING = "reloading"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ValidationLevel(Enum):
    """Code validation levels"""
    SYNTAX_ONLY = "syntax_only"
    IMPORT_CHECK = "import_check"
    BASIC_EXECUTION = "basic_execution"
    FULL_TEST = "full_test"


@dataclass
class PatchOperation:
    """Record of a patch operation"""
    patch_id: str
    file_path: str
    original_code: str
    patched_code: str
    timestamp: datetime
    status: PatchStatus
    backup_path: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False
    modules_reloaded: List[str] = None
    test_results: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    syntax_valid: bool
    imports_valid: bool
    execution_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    validation_time: float


class CodePatcher:
    """
    Code patching and hot-reload system for self-healing
    
    This class provides safe code patching capabilities with validation,
    backup creation, hot-reload functionality, and rollback mechanisms.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize code patcher"""
        self.config = config or {}
        self.logger = get_logger("code_patcher")
        
        # Patching state
        self.active_patches: Dict[str, PatchOperation] = {}
        self.patch_history: List[PatchOperation] = []
        self.locked_modules: Set[str] = set()
        self.reload_lock = threading.Lock()
        
        # Configuration
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'basic_execution'))
        self.auto_rollback = self.config.get('auto_rollback', True)
        self.max_patch_size = self.config.get('max_patch_size', 100000)  # 100KB
        self.test_timeout = self.config.get('test_timeout', 30)  # seconds
        self.backup_dir = Path(self.config.get('backup_dir', 'data/backups'))
        self.temp_dir = Path(self.config.get('temp_dir', 'data/temp'))
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Module tracking
        self.loaded_modules: Dict[str, Any] = {}
        self.module_dependencies: Dict[str, Set[str]] = {}
        
        self.logger.info("CodePatcher initialized")
    
    async def initialize(self) -> bool:
        """Initialize the code patcher"""
        try:
            # Track currently loaded modules
            await self._scan_loaded_modules()
            
            # Clean up old temporary files
            await self._cleanup_temp_files()
            
            self.logger.info("CodePatcher initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CodePatcher: {e}")
            return False
    
    async def apply_fix(self, file_path: str, fix_code: str) -> bool:
        """
        Apply a code fix to a file with hot-reload
        
        Args:
            file_path: Path to the file to patch
            fix_code: Code to apply as fix
            
        Returns:
            True if patch was applied successfully
        """
        patch_id = f"patch_{int(time.time())}_{hash(fix_code) % 10000:04d}"
        
        try:
            self.logger.info(f"Applying patch {patch_id} to {file_path}")
            
            # Validate file path and code
            if not await self._validate_inputs(file_path, fix_code):
                return False
            
            # Read original code
            original_code = await self._read_file_safe(file_path)
            if original_code is None:
                return False
            
            # Create patch operation
            patch_op = PatchOperation(
                patch_id=patch_id,
                file_path=file_path,
                original_code=original_code,
                patched_code=fix_code,
                timestamp=datetime.now(),
                status=PatchStatus.PENDING,
                modules_reloaded=[]
            )
            
            self.active_patches[patch_id] = patch_op
            
            # Step 1: Validate the patch
            patch_op.status = PatchStatus.VALIDATING
            validation_result = await self._validate_patch(fix_code, file_path)
            patch_op.validation_results = asdict(validation_result)
            
            if not validation_result.is_valid:
                patch_op.status = PatchStatus.FAILED
                patch_op.error_message = f"Validation failed: {'; '.join(validation_result.errors)}"
                self.logger.error(f"Patch {patch_id} validation failed")
                return False
            
            # Step 2: Create backup
            backup_path = None
            if self.backup_enabled:
                backup_path = await self._create_backup(file_path, patch_id)
                patch_op.backup_path = backup_path
            
            # Step 3: Apply patch
            patch_op.status = PatchStatus.APPLYING
            if not await self._apply_patch_to_file(file_path, fix_code):
                if self.auto_rollback and backup_path:
                    await self._rollback_file(file_path, backup_path)
                    patch_op.rollback_performed = True
                patch_op.status = PatchStatus.FAILED
                patch_op.error_message = "Failed to apply patch to file"
                return False
            
            # Step 4: Reload modules
            patch_op.status = PatchStatus.RELOADING
            reloaded_modules = await self._reload_modules(file_path)
            patch_op.modules_reloaded = reloaded_modules
            
            # Step 5: Test the patch
            patch_op.status = PatchStatus.TESTING
            test_results = await self._test_patch(file_path, reloaded_modules)
            patch_op.test_results = test_results
            
            if not test_results.get('success', False):
                self.logger.warning(f"Patch {patch_id} tests failed, rolling back")
                if self.auto_rollback and backup_path:
                    await self._rollback_file(file_path, backup_path)
                    await self._reload_modules(file_path)  # Reload original
                    patch_op.rollback_performed = True
                patch_op.status = PatchStatus.ROLLED_BACK
                patch_op.error_message = f"Tests failed: {test_results.get('error', 'Unknown error')}"
                return False
            
            # Success!
            patch_op.status = PatchStatus.COMPLETED
            self.patch_history.append(patch_op)
            
            # Clean up active patch
            if patch_id in self.active_patches:
                del self.active_patches[patch_id]
            
            self.logger.info(f"Patch {patch_id} applied successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Patch {patch_id} failed with exception: {e}")
            if patch_id in self.active_patches:
                patch_op = self.active_patches[patch_id]
                patch_op.status = PatchStatus.FAILED
                patch_op.error_message = str(e)
                
                # Auto-rollback if enabled
                if self.auto_rollback and patch_op.backup_path:
                    await self._rollback_file(file_path, patch_op.backup_path)
                    patch_op.rollback_performed = True
            
            return False
    
    async def validate_fix(self, error_context) -> bool:
        """
        Validate that a fix works by testing the error scenario
        
        Args:
            error_context: Original error context
            
        Returns:
            True if fix is validated
        """
        try:
            file_path = error_context.file_path
            function_name = error_context.function_name
            
            if not file_path or not function_name:
                self.logger.warning("Cannot validate fix: missing file path or function name")
                return False
            
            # Get the module and function
            module_name = self._get_module_name_from_path(file_path)
            if not module_name:
                return False
            
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, function_name):
                    self.logger.warning(f"Function {function_name} not found in module {module_name}")
                    return False
                
                # Try to call the function with test data
                func = getattr(module, function_name)
                
                # Create test parameters based on error context
                test_args = self._create_test_args(error_context)
                
                # Test the function
                if inspect.iscoroutinefunction(func):
                    result = await func(*test_args['args'], **test_args['kwargs'])
                else:
                    result = func(*test_args['args'], **test_args['kwargs'])
                
                self.logger.info(f"Function {function_name} validated successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Function validation failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fix validation failed: {e}")
            return False
    
    async def _validate_inputs(self, file_path: str, fix_code: str) -> bool:
        """Validate inputs for patching"""
        # Check file path
        if not file_path or not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False
        
        # Check file size
        if len(fix_code.encode()) > self.max_patch_size:
            self.logger.error(f"Patch too large: {len(fix_code)} bytes")
            return False
        
        # Check file extension
        if not file_path.endswith('.py'):
            self.logger.error(f"Only Python files can be patched: {file_path}")
            return False
        
        return True
    
    async def _read_file_safe(self, file_path: str) -> Optional[str]:
        """Safely read file contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    async def _validate_patch(self, fix_code: str, file_path: str) -> ValidationResult:
        """Validate patch code"""
        start_time = time.time()
        errors = []
        warnings = []
        suggestions = []
        
        # Syntax validation
        syntax_valid = True
        try:
            ast.parse(fix_code)
        except SyntaxError as e:
            syntax_valid = False
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            syntax_valid = False
            errors.append(f"Parse error: {e}")
        
        # Import validation
        imports_valid = True
        if syntax_valid and self.validation_level.value in ['import_check', 'basic_execution', 'full_test']:
            try:
                # Extract imports and check if they're available
                tree = ast.parse(fix_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            try:
                                importlib.import_module(alias.name)
                            except ImportError:
                                imports_valid = False
                                errors.append(f"Import not found: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            try:
                                importlib.import_module(node.module)
                            except ImportError:
                                imports_valid = False
                                errors.append(f"Import not found: {node.module}")
            except Exception as e:
                warnings.append(f"Could not validate imports: {e}")
        
        # Basic execution validation
        execution_valid = True
        if (syntax_valid and imports_valid and 
            self.validation_level.value in ['basic_execution', 'full_test']):
            
            try:
                # Create a temporary file and try to execute it
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(fix_code)
                    temp_file = f.name
                
                try:
                    # Execute in a subprocess for safety
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, '-m', 'py_compile', temp_file],
                        capture_output=True,
                        text=True,
                        timeout=self.test_timeout
                    )
                    
                    if result.returncode != 0:
                        execution_valid = False
                        errors.append(f"Compilation failed: {result.stderr}")
                        
                finally:
                    os.unlink(temp_file)
                    
            except Exception as e:
                execution_valid = False
                errors.append(f"Execution validation failed: {e}")
        
        # Generate suggestions
        if not syntax_valid:
            suggestions.append("Check for syntax errors and indentation")
        if not imports_valid:
            suggestions.append("Verify all imports are available")
        if not execution_valid:
            suggestions.append("Test the code logic and dependencies")
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=syntax_valid and imports_valid and execution_valid,
            syntax_valid=syntax_valid,
            imports_valid=imports_valid,
            execution_valid=execution_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            validation_time=validation_time
        )
    
    async def _create_backup(self, file_path: str, patch_id: str) -> Optional[str]:
        """Create backup of file before patching"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            backup_filename = f"{filename}.{timestamp}.{patch_id}.bak"
            backup_path = self.backup_dir / backup_filename
            
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    async def _apply_patch_to_file(self, file_path: str, fix_code: str) -> bool:
        """Apply patch to file atomically"""
        try:
            # Write to temporary file first
            temp_file = self.temp_dir / f"temp_{int(time.time())}.py"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(fix_code)
            
            # Atomic move
            shutil.move(str(temp_file), file_path)
            
            self.logger.info(f"Applied patch to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply patch: {e}")
            return False
    
    async def _reload_modules(self, file_path: str) -> List[str]:
        """Reload modules affected by the patch"""
        reloaded_modules = []
        
        with self.reload_lock:
            try:
                module_name = self._get_module_name_from_path(file_path)
                if not module_name:
                    return reloaded_modules
                
                # Get the module if it's loaded
                if module_name in sys.modules:
                    # Reload the main module
                    module = importlib.reload(sys.modules[module_name])
                    reloaded_modules.append(module_name)
                    
                    # Find and reload dependent modules
                    dependents = self._find_dependent_modules(module_name)
                    for dependent in dependents:
                        if dependent in sys.modules:
                            importlib.reload(sys.modules[dependent])
                            reloaded_modules.append(dependent)
                
                self.logger.info(f"Reloaded modules: {reloaded_modules}")
                return reloaded_modules
                
            except Exception as e:
                self.logger.error(f"Failed to reload modules: {e}")
                return reloaded_modules
    
    def _get_module_name_from_path(self, file_path: str) -> Optional[str]:
        """Convert file path to module name"""
        try:
            # Get absolute path and make it relative to project root
            abs_path = os.path.abspath(file_path)
            project_root = os.path.abspath(os.getcwd())
            
            if abs_path.startswith(project_root):
                rel_path = os.path.relpath(abs_path, project_root)
                # Convert path separators to dots and remove .py extension
                module_name = rel_path.replace(os.sep, '.')
                if module_name.endswith('.py'):
                    module_name = module_name[:-3]
                return module_name
            
            return None
            
        except Exception:
            return None
    
    def _find_dependent_modules(self, module_name: str) -> List[str]:
        """Find modules that depend on the given module"""
        dependents = []
        
        try:
            for name, module in sys.modules.items():
                if hasattr(module, '__file__') and module.__file__:
                    # Check if the module imports the target module
                    try:
                        with open(module.__file__, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if f"import {module_name}" in content or f"from {module_name}" in content:
                                dependents.append(name)
                    except Exception:
                        continue
                        
        except Exception as e:
            self.logger.warning(f"Failed to find dependent modules: {e}")
        
        return dependents
    
    async def _test_patch(self, file_path: str, reloaded_modules: List[str]) -> Dict[str, Any]:
        """Test the applied patch"""
        try:
            # Basic test: try to import all reloaded modules
            for module_name in reloaded_modules:
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"Failed to import {module_name}: {e}"
                    }
            
            # If we have specific test functions, run them
            module_name = self._get_module_name_from_path(file_path)
            if module_name and module_name in sys.modules:
                module = sys.modules[module_name]
                
                # Look for test functions
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (attr_name.startswith('test_') and 
                        callable(attr) and 
                        not attr_name.startswith('__')):
                        
                        try:
                            if inspect.iscoroutinefunction(attr):
                                await attr()
                            else:
                                attr()
                        except Exception as e:
                            return {
                                'success': False,
                                'error': f"Test {attr_name} failed: {e}"
                            }
            
            return {'success': True, 'message': 'All tests passed'}
            
        except Exception as e:
            return {'success': False, 'error': f"Testing failed: {e}"}
    
    async def _rollback_file(self, file_path: str, backup_path: str) -> bool:
        """Rollback file to backup"""
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                self.logger.info(f"Rolled back {file_path} to {backup_path}")
                return True
            else:
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _create_test_args(self, error_context) -> Dict[str, Any]:
        """Create test arguments based on error context"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated test data generation
        return {
            'args': [],
            'kwargs': {}
        }
    
    async def _scan_loaded_modules(self):
        """Scan currently loaded modules"""
        try:
            for name, module in sys.modules.items():
                if hasattr(module, '__file__') and module.__file__:
                    self.loaded_modules[name] = module
                    
        except Exception as e:
            self.logger.warning(f"Failed to scan loaded modules: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up old temporary files"""
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("temp_*.py"):
                    # Remove files older than 1 hour
                    if time.time() - temp_file.stat().st_mtime > 3600:
                        temp_file.unlink()
                        
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    async def get_patch_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get patch operation history"""
        history = self.patch_history[-limit:] if limit > 0 else self.patch_history
        return [asdict(patch) for patch in history]
    
    async def get_active_patches(self) -> List[Dict[str, Any]]:
        """Get currently active patches"""
        return [asdict(patch) for patch in self.active_patches.values()]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get patching statistics"""
        total_patches = len(self.patch_history)
        successful_patches = sum(1 for p in self.patch_history if p.status == PatchStatus.COMPLETED)
        failed_patches = sum(1 for p in self.patch_history if p.status == PatchStatus.FAILED)
        rolled_back_patches = sum(1 for p in self.patch_history if p.status == PatchStatus.ROLLED_BACK)
        
        success_rate = successful_patches / total_patches if total_patches > 0 else 0
        
        return {
            'total_patches': total_patches,
            'successful_patches': successful_patches,
            'failed_patches': failed_patches,
            'rolled_back_patches': rolled_back_patches,
            'success_rate': success_rate,
            'active_patches': len(self.active_patches),
            'backups_created': len(list(self.backup_dir.glob("*.bak"))),
            'loaded_modules': len(self.loaded_modules)
        }
    
    async def manual_rollback(self, patch_id: str) -> bool:
        """Manually rollback a specific patch"""
        try:
            if patch_id not in self.active_patches:
                # Check history
                for patch in self.patch_history:
                    if patch.patch_id == patch_id and patch.backup_path:
                        return await self._rollback_file(patch.file_path, patch.backup_path)
                return False
            
            patch = self.active_patches[patch_id]
            if patch.backup_path:
                success = await self._rollback_file(patch.file_path, patch.backup_path)
                if success:
                    patch.rollback_performed = True
                    patch.status = PatchStatus.ROLLED_BACK
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Manual rollback failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the code patcher"""
        try:
            # Wait for active patches to complete
            if self.active_patches:
                self.logger.info(f"Waiting for {len(self.active_patches)} active patches to complete...")
                await asyncio.sleep(5)
            
            # Clean up temporary files
            await self._cleanup_temp_files()
            
            self.logger.info("CodePatcher shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")