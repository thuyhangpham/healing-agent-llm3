"""
Global State Management System

Centralized state management for the multi-agent system.
Provides persistent state storage, validation, and change tracking.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading

# Local logger to avoid circular import
def _get_logger():
    return logging.getLogger("global_state")

# Simple settings fallback
class _Settings:
    def __init__(self):
        self.log_level = "INFO"

def _get_settings():
    try:
        from utils.config import settings
        return settings
    except ImportError:
        return _Settings()

settings = _get_settings()


class StateChangeType(Enum):
    """Types of state changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BULK_UPDATE = "bulk_update"


@dataclass
class StateChange:
    """Represents a state change event."""
    timestamp: datetime
    change_type: StateChangeType
    key: str
    old_value: Any
    new_value: Any
    agent_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StateValidationError(Exception):
    """Exception raised for state validation errors."""
    pass


class StateValidator:
    """Validates state changes based on defined rules."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.logger = StructuredLogger("state_validator", settings.log_level)
    
    def add_validation_rule(self, key: str, rule: Callable[[Any], bool]):
        """Add a validation rule for a specific state key."""
        if key not in self.validation_rules:
            self.validation_rules[key] = []
        self.validation_rules[key].append(rule)
        self.logger.debug(f"Added validation rule for key: {key}")
    
    def validate_state_change(self, key: str, value: Any) -> bool:
        """Validate a state change against all rules for the key."""
        if key not in self.validation_rules:
            return True  # No rules defined for this key
        
        for rule in self.validation_rules[key]:
            try:
                if not rule(value):
                    raise StateValidationError(f"Validation failed for key '{key}' with value '{value}'")
            except Exception as e:
                self.logger.error(f"State validation error for key '{key}': {e}")
                raise StateValidationError(f"Validation error: {e}")
        
        return True
    
    def validate_full_state(self, state: Dict[str, Any]) -> List[str]:
        """Validate the entire state dictionary."""
        errors = []
        
        for key, value in state.items():
            try:
                self.validate_state_change(key, value)
            except StateValidationError as e:
                errors.append(f"{key}: {str(e)}")
        
        return errors


class GlobalStateManager:
    """
    Centralized global state management system.
    
    Features:
    - Thread-safe state operations
    - State change tracking and history
    - Validation system
    - Persistent storage (optional)
    - Event notifications
    - Performance monitoring
    """
    
    def __init__(self, enable_persistence: bool = False, persistence_file: Optional[str] = None):
        self.logger = StructuredLogger("global_state", settings.log_level)
        
        # Core state storage
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Change tracking
        self._change_history: List[StateChange] = []
        self._max_history_size = 1000
        
        # Event system
        self._event_listeners: Dict[str, List[Callable]] = {}
        
        # Validation
        self.validator = StateValidator()
        self._setup_default_validators()
        
        # Persistence
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file or "data/global_state.json"
        
        # Performance metrics
        self._metrics = {
            'total_changes': 0,
            'changes_per_second': 0.0,
            'last_change_time': None,
            'validation_errors': 0,
            'persistence_errors': 0
        }
        
        # Load initial state
        if enable_persistence:
            self._load_state()
        
        self.logger.info("Global state manager initialized")
    
    def _setup_default_validators(self):
        """Setup default validation rules."""
        # Agent status validation
        valid_agent_statuses = {'initializing', 'idle', 'working', 'error', 'completed', 'disabled'}
        self.validator.add_validation_rule(
            'agent_status', 
            lambda x: x in valid_agent_statuses
        )
        
        # Task priority validation
        valid_priorities = {1, 2, 3, 4}  # LOW, MEDIUM, HIGH, CRITICAL
        self.validator.add_validation_rule(
            'task_priority',
            lambda x: x in valid_priorities
        )
        
        # Workflow status validation
        valid_workflow_statuses = {'starting', 'running', 'paused', 'stopping', 'stopped'}
        self.validator.add_validation_rule(
            'workflow_status',
            lambda x: x in valid_workflow_statuses
        )
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the global state."""
        with self._lock:
            return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any, agent_name: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a value in the global state with validation and tracking."""
        try:
            # Validate the change
            self.validator.validate_state_change(key, value)
            
            with self._lock:
                old_value = self._state.get(key)
                
                # Record the change
                change = StateChange(
                    timestamp=datetime.utcnow(),
                    change_type=StateChangeType.UPDATE if key in self._state else StateChangeType.CREATE,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    agent_name=agent_name,
                    metadata=metadata
                )
                
                # Update state
                self._state[key] = value
                self._change_history.append(change)
                
                # Trim history if needed
                if len(self._change_history) > self._max_history_size:
                    self._change_history = self._change_history[-self._max_history_size:]
                
                # Update metrics
                self._update_metrics()
                
                # Persist if enabled
                if self.enable_persistence:
                    self._save_state_async()
                
                # Emit events
                self._emit_state_change_event(change)
                
                self.logger.debug(f"State updated: {key} = {value}", agent=agent_name)
                
                return True
                
        except StateValidationError as e:
            self.logger.warning(f"State validation failed: {e}")
            self._metrics['validation_errors'] += 1
            return False
        except Exception as e:
            self.logger.error(f"Failed to set state {key}: {e}")
            return False
    
    def delete_state(self, key: str, agent_name: Optional[str] = None) -> bool:
        """Delete a key from the global state."""
        try:
            with self._lock:
                if key not in self._state:
                    return False
                
                old_value = self._state[key]
                
                # Record the change
                change = StateChange(
                    timestamp=datetime.utcnow(),
                    change_type=StateChangeType.DELETE,
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    agent_name=agent_name
                )
                
                # Delete from state
                del self._state[key]
                self._change_history.append(change)
                
                # Update metrics
                self._update_metrics()
                
                # Persist if enabled
                if self.enable_persistence:
                    self._save_state_async()
                
                # Emit events
                self._emit_state_change_event(change)
                
                self.logger.debug(f"State deleted: {key}", agent=agent_name)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete state {key}: {e}")
            return False
    
    def bulk_update_state(self, updates: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, bool]:
        """Update multiple state values atomically."""
        results = {}
        
        try:
            # Validate all changes first
            for key, value in updates.items():
                self.validator.validate_state_change(key, value)
            
            with self._lock:
                old_values = {key: self._state.get(key) for key in updates.keys()}
                
                # Record bulk change
                change = StateChange(
                    timestamp=datetime.utcnow(),
                    change_type=StateChangeType.BULK_UPDATE,
                    key="bulk_update",
                    old_value=old_values,
                    new_value=updates,
                    agent_name=agent_name,
                    metadata={'updated_keys': list(updates.keys())}
                )
                
                # Apply all updates
                for key, value in updates.items():
                    self._state[key] = value
                    results[key] = True
                
                self._change_history.append(change)
                
                # Update metrics
                self._update_metrics()
                
                # Persist if enabled
                if self.enable_persistence:
                    self._save_state_async()
                
                # Emit events
                self._emit_state_change_event(change)
                
                self.logger.info(f"Bulk state update: {len(updates)} keys", agent=agent_name)
                
        except StateValidationError as e:
            self.logger.warning(f"Bulk state validation failed: {e}")
            self._metrics['validation_errors'] += 1
            results = {key: False for key in updates.keys()}
        except Exception as e:
            self.logger.error(f"Failed bulk state update: {e}")
            results = {key: False for key in updates.keys()}
        
        return results
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get a copy of the entire state."""
        with self._lock:
            return self._state.copy()
    
    def get_change_history(self, key: Optional[str] = None, 
                        since: Optional[datetime] = None,
                        limit: Optional[int] = None) -> List[StateChange]:
        """Get state change history with optional filtering."""
        with self._lock:
            history = self._change_history.copy()
            
            # Filter by key
            if key:
                history = [change for change in history if change.key == key]
            
            # Filter by time
            if since:
                history = [change for change in history if change.timestamp >= since]
            
            # Limit results
            if limit:
                history = history[-limit:]
            
            return history
    
    def add_event_listener(self, event_type: str, listener: Callable[[StateChange], None]):
        """Add an event listener for state changes."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(listener)
        self.logger.debug(f"Added event listener for: {event_type}")
    
    def remove_event_listener(self, event_type: str, listener: Callable[[StateChange], None]):
        """Remove an event listener."""
        if event_type in self._event_listeners:
            try:
                self._event_listeners[event_type].remove(listener)
                self.logger.debug(f"Removed event listener for: {event_type}")
            except ValueError:
                pass  # Listener not found
    
    def _emit_state_change_event(self, change: StateChange):
        """Emit state change events to listeners."""
        event_type = change.change_type.value
        
        # Emit to specific event type listeners
        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    listener(change)
                except Exception as e:
                    self.logger.error(f"Event listener error: {e}")
        
        # Also emit to general 'any' listeners if they exist
        if 'any' in self._event_listeners:
            for listener in self._event_listeners['any']:
                try:
                    listener(change)
                except Exception as e:
                    self.logger.error(f"Event listener error: {e}")
    
    def _update_metrics(self):
        """Update performance metrics."""
        self._metrics['total_changes'] += 1
        self._metrics['last_change_time'] = datetime.utcnow()
        
        # Calculate changes per second
        if len(self._change_history) > 1:
            time_span = self._change_history[-1].timestamp - self._change_history[0].timestamp
            if time_span.total_seconds() > 0:
                self._metrics['changes_per_second'] = len(self._change_history) / time_span.total_seconds()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager performance metrics."""
        with self._lock:
            return {
                **self._metrics,
                'state_size': len(self._state),
                'history_size': len(self._change_history),
                'event_listeners': sum(len(listeners) for listeners in self._event_listeners.values())
            }
    
    def validate_state(self) -> List[str]:
        """Validate the entire current state."""
        with self._lock:
            return self.validator.validate_full_state(self._state)
    
    def _save_state_async(self):
        """Save state to persistent storage asynchronously."""
        try:
            asyncio.create_task(self._save_state())
        except RuntimeError:
            # No event loop, save synchronously
            self._save_state()
    
    def _save_state(self) -> bool:
        """Save state to persistent storage."""
        if not self.enable_persistence:
            return True
        
        try:
            import os
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            
            state_data = {
                'state': self._state,
                'metadata': {
                    'saved_at': datetime.utcnow().isoformat(),
                    'total_changes': self._metrics['total_changes'],
                    'version': '1.0'
                }
            }
            
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.debug(f"State saved to: {self.persistence_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            self._metrics['persistence_errors'] += 1
            return False
    
    def _load_state(self) -> bool:
        """Load state from persistent storage."""
        if not self.enable_persistence:
            return True
        
        try:
            import os
            if not os.path.exists(self.persistence_file):
                self.logger.info("No existing state file found, starting fresh")
                return True
            
            with open(self.persistence_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            with self._lock:
                self._state = state_data.get('state', {})
                
                # Load metadata
                metadata = state_data.get('metadata', {})
                saved_at = metadata.get('saved_at')
                if saved_at:
                    self.logger.info(f"Loaded state from: {saved_at}")
                
                self.logger.info(f"Loaded {len(self._state)} state keys")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            self._metrics['persistence_errors'] += 1
            return False
    
    def clear_state(self, agent_name: Optional[str] = None) -> bool:
        """Clear all state (with optional agent tracking)."""
        try:
            with self._lock:
                old_state = self._state.copy()
                self._state.clear()
                
                # Record the clear operation
                change = StateChange(
                    timestamp=datetime.utcnow(),
                    change_type=StateChangeType.BULK_UPDATE,
                    key="clear_state",
                    old_value=old_state,
                    new_value={},
                    agent_name=agent_name
                )
                
                self._change_history.append(change)
                
                # Persist if enabled
                if self.enable_persistence:
                    self._save_state_async()
                
                # Emit events
                self._emit_state_change_event(change)
                
                self.logger.info("State cleared", agent=agent_name)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear state: {e}")
            return False
    
    def create_snapshot(self, name: str) -> str:
        """Create a named snapshot of the current state."""
        snapshot_id = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with self._lock:
                snapshot_data = {
                    'snapshot_id': snapshot_id,
                    'name': name,
                    'created_at': datetime.utcnow().isoformat(),
                    'state': self._state.copy(),
                    'change_count': len(self._change_history)
                }
                
                import os
                snapshots_dir = "data/snapshots"
                os.makedirs(snapshots_dir, exist_ok=True)
                
                snapshot_file = fr"{snapshots_dir}/{snapshot_id}.json"
                with open(snapshot_file, 'w', encoding='utf-8') as f:
                    json.dump(snapshot_data, snapshot_data, indent=2, default=str)
                
                self.logger.info(f"Created snapshot: {snapshot_id}")
                return snapshot_id
                
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            return ""
    
    def __str__(self) -> str:
        """String representation of state manager."""
        return f"GlobalStateManager(keys={len(self._state)}, changes={len(self._change_history)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"GlobalStateManager(keys={len(self._state)}, "
                f"changes={len(self._change_history)}, "
                f"listeners={sum(len(l) for l in self._event_listeners.values())})")


# Global state manager instance
_global_state_manager: Optional[GlobalStateManager] = None


def get_global_state_manager() -> GlobalStateManager:
    """Get the singleton global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = GlobalStateManager(
            enable_persistence=settings.get('enable_state_persistence', False),
            persistence_file=settings.get('state_persistence_file', 'data/global_state.json')
        )
    return _global_state_manager


def get_state(key: str, default: Any = None) -> Any:
    """Convenience function to get global state."""
    return get_global_state_manager().get_state(key, default)


def set_state(key: str, value: Any, agent_name: Optional[str] = None) -> bool:
    """Convenience function to set global state."""
    return get_global_state_manager().set_state(key, value, agent_name)


def bulk_set_state(updates: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, bool]:
    """Convenience function for bulk state updates."""
    return get_global_state_manager().bulk_update_state(updates, agent_name)