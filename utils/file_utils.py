"""
File Utilities

Common file operations for the system including backup creation,
file reading/writing, and path management.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional


class FileUtils:
    """Utility class for common file operations."""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists, create if it doesn't."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def create_backup(file_path: str) -> str:
        """Create backup of file with .bak extension."""
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read file content as string."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def write_file(file_path: str, content: str) -> None:
        """Write content to file."""
        # Ensure directory exists
        FileUtils.ensure_directory(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """Read JSON file as dictionary."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]) -> None:
        """Write dictionary to JSON file."""
        # Ensure directory exists
        FileUtils.ensure_directory(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if file exists."""
        return os.path.isfile(file_path)
    
    @staticmethod
    def get_relative_path(file_path: str, base_path: str) -> str:
        """Get relative path from base path."""
        return os.path.relpath(file_path, base_path)


# Convenience functions for backward compatibility
def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    FileUtils.ensure_directory(path)


def safe_write(file_path: str, content: str) -> None:
    """Write content to file safely."""
    FileUtils.write_file(file_path, content)


def safe_read(file_path: str) -> str:
    """Read file content safely."""
    return FileUtils.read_file(file_path)