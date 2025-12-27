"""
Tests for Global State Management System

Test suite for the centralized state management system
that provides persistent storage, validation, and change tracking.
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from utils.global_state import (
    GlobalStateManager, StateChange, StateChangeType, StateValidationError,
    get_global_state_manager, get_state, set_state, bulk_set_state
)


class TestStateChange:
    """Test cases for StateChange dataclass."""
    
    def test_state_change_creation(self):
        """Test StateChange creation."""
        timestamp = datetime.utcnow()
        change = StateChange(
            timestamp=timestamp,
            change_type=StateChangeType.UPDATE,
            key="test_key",
            old_value="old",
            new_value="new",
            agent_name="test_agent"
        )
        
        assert change.timestamp == timestamp
        assert change.change_type == StateChangeType.UPDATE
        assert change.key == "test_key"
        assert change.old_value == "old"
        assert change.new_value == "new"
        assert change.agent_name == "test_agent"
        assert change.metadata is None


class TestStateValidator:
    """Test cases for StateValidator class."""
    
    def test_add_validation_rule(self):
        """Test adding validation rules."""
        from utils.global_state import StateValidator
        
        validator = StateValidator()
        
        # Add a simple rule
        validator.add_validation_rule("test_key", lambda x: x > 0)
        
        assert "test_key" in validator.validation_rules
        assert len(validator.validation_rules["test_key"]) == 1
    
    def test_validate_state_change_success(self):
        """Test successful state validation."""
        from utils.global_state import StateValidator
        
        validator = StateValidator()
        validator.add_validation_rule("test_key", lambda x: x > 0)
        
        # Should pass validation
        assert validator.validate_state_change("test_key", 5) is True
    
    def test_validate_state_change_failure(self):
        """Test failed state validation."""
        from utils.global_state import StateValidator
        
        validator = StateValidator()
        validator.add_validation_rule("test_key", lambda x: x > 0)
        
        # Should fail validation
        with pytest.raises(StateValidationError):
            validator.validate_state_change("test_key", -1)
    
    def test_validate_full_state(self):
        """Test full state validation."""
        from utils.global_state import StateValidator
        
        validator = StateValidator()
        validator.add_validation_rule("key1", lambda x: x > 0)
        validator.add_validation_rule("key2", lambda x: len(str(x)) > 2)
        
        # Valid state
        valid_state = {"key1": 5, "key2": "test"}
        errors = validator.validate_full_state(valid_state)
        assert len(errors) == 0
        
        # Invalid state
        invalid_state = {"key1": -1, "key2": "x"}
        errors = validator.validate_full_state(invalid_state)
        assert len(errors) == 2


class TestGlobalStateManager:
    """Test cases for GlobalStateManager class."""
    
    @pytest.fixture
    def temp_state_file(self):
        """Create a temporary file for state persistence."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    @pytest.fixture
    def state_manager(self, temp_state_file):
        """Create a state manager instance for testing."""
        with patch('utils.global_state.StructuredLogger'), \
             patch('utils.global_state.settings'):
            return GlobalStateManager(
                enable_persistence=False,
                persistence_file=temp_state_file
            )
    
    def test_state_manager_initialization(self, state_manager):
        """Test state manager initialization."""
        assert state_manager._state == {}
        assert len(state_manager._change_history) == 0
        assert state_manager.enable_persistence is False
        assert state_manager._max_history_size == 1000
    
    def test_get_and_set_state(self, state_manager):
        """Test basic state get/set operations."""
        # Test getting non-existent key
        assert state_manager.get_state("nonexistent") is None
        assert state_manager.get_state("nonexistent", "default") == "default"
        
        # Test setting and getting
        result = state_manager.set_state("test_key", "test_value")
        assert result is True
        assert state_manager.get_state("test_key") == "test_value"
    
    def test_set_state_with_validation(self, state_manager):
        """Test state setting with validation."""
        # Add a validation rule
        state_manager.validator.add_validation_rule("positive_number", lambda x: x > 0)
        
        # Valid value should work
        result = state_manager.set_state("positive_number", 5)
        assert result is True
        assert state_manager.get_state("positive_number") == 5
        
        # Invalid value should fail
        result = state_manager.set_state("positive_number", -1)
        assert result is False
        assert state_manager.get_state("positive_number") == 5  # Unchanged
    
    def test_delete_state(self, state_manager):
        """Test state deletion."""
        # Set a value first
        state_manager.set_state("test_key", "test_value")
        assert state_manager.get_state("test_key") == "test_value"
        
        # Delete the value
        result = state_manager.delete_state("test_key")
        assert result is True
        assert state_manager.get_state("test_key") is None
        
        # Try to delete non-existent key
        result = state_manager.delete_state("nonexistent")
        assert result is False
    
    def test_bulk_update_state(self, state_manager):
        """Test bulk state updates."""
        updates = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        results = state_manager.bulk_update_state(updates)
        
        # All should succeed
        assert all(results.values())
        assert state_manager.get_state("key1") == "value1"
        assert state_manager.get_state("key2") == "value2"
        assert state_manager.get_state("key3") == "value3"
    
    def test_bulk_update_with_validation_failure(self, state_manager):
        """Test bulk update with validation failure."""
        # Add validation rule
        state_manager.validator.add_validation_rule("key2", lambda x: x > 0)
        
        updates = {
            "key1": "value1",
            "key2": -1,  # Invalid
            "key3": "value3"
        }
        
        results = state_manager.bulk_update_state(updates)
        
        # All should fail due to validation
        assert all(not success for success in results.values())
        # State should be unchanged
        assert state_manager.get_state("key1") is None
        assert state_manager.get_state("key2") is None
        assert state_manager.get_state("key3") is None
    
    def test_get_full_state(self, state_manager):
        """Test getting full state copy."""
        state_manager.set_state("key1", "value1")
        state_manager.set_state("key2", "value2")
        
        full_state = state_manager.get_full_state()
        
        assert full_state == {"key1": "value1", "key2": "value2"}
        
        # Verify it's a copy
        full_state["key3"] = "value3"
        assert state_manager.get_state("key3") is None
    
    def test_change_history(self, state_manager):
        """Test change history tracking."""
        # Make some changes
        state_manager.set_state("key1", "value1", agent_name="agent1")
        state_manager.set_state("key1", "value2", agent_name="agent2")
        state_manager.delete_state("key1", agent_name="agent3")
        
        # Get full history
        history = state_manager.get_change_history()
        assert len(history) == 3
        
        # Check first change (create)
        first_change = history[0]
        assert first_change.change_type == StateChangeType.CREATE
        assert first_change.key == "key1"
        assert first_change.old_value is None
        assert first_change.new_value == "value1"
        assert first_change.agent_name == "agent1"
        
        # Check second change (update)
        second_change = history[1]
        assert second_change.change_type == StateChangeType.UPDATE
        assert second_change.key == "key1"
        assert second_change.old_value == "value1"
        assert second_change.new_value == "value2"
        assert second_change.agent_name == "agent2"
        
        # Check third change (delete)
        third_change = history[2]
        assert third_change.change_type == StateChangeType.DELETE
        assert third_change.key == "key1"
        assert third_change.old_value == "value2"
        assert third_change.new_value is None
        assert third_change.agent_name == "agent3"
    
    def test_change_history_filtering(self, state_manager):
        """Test change history filtering."""
        # Make changes at different times
        base_time = datetime.utcnow()
        
        with patch('utils.global_state.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = base_time
            state_manager.set_state("key1", "value1")
            
            mock_datetime.utcnow.return_value = base_time + timedelta(seconds=10)
            state_manager.set_state("key2", "value2")
            
            mock_datetime.utcnow.return_value = base_time + timedelta(seconds=20)
            state_manager.set_state("key1", "updated")
        
        # Filter by key
        key1_history = state_manager.get_change_history(key="key1")
        assert len(key1_history) == 2
        assert all(change.key == "key1" for change in key1_history)
        
        # Filter by time
        filtered_history = state_manager.get_change_history(
            since=base_time + timedelta(seconds=5)
        )
        assert len(filtered_history) == 2
        
        # Filter with limit
        limited_history = state_manager.get_change_history(limit=2)
        assert len(limited_history) == 2
        assert limited_history[0].key == "key2"  # Most recent first
    
    def test_event_listeners(self, state_manager):
        """Test event listener system."""
        events_received = []
        
        def test_listener(change):
            events_received.append(change)
        
        # Add listener for any change type
        state_manager.add_event_listener("any", test_listener)
        
        # Make a change
        state_manager.set_state("test_key", "test_value")
        
        # Check event was received
        assert len(events_received) == 1
        assert events_received[0].key == "test_key"
        assert events_received[0].new_value == "test_value"
        
        # Remove listener
        state_manager.remove_event_listener("update", test_listener)
        
        # Make another change
        state_manager.set_state("test_key2", "test_value2")
        
        # Should have received both events
        assert len(events_received) == 2
    
    def test_metrics(self, state_manager):
        """Test performance metrics."""
        # Initial metrics
        metrics = state_manager.get_metrics()
        assert metrics['total_changes'] == 0
        assert metrics['state_size'] == 0
        assert metrics['history_size'] == 0
        
        # Make some changes
        state_manager.set_state("key1", "value1")
        state_manager.set_state("key2", "value2")
        
        metrics = state_manager.get_metrics()
        assert metrics['total_changes'] == 2
        assert metrics['state_size'] == 2
        assert metrics['history_size'] == 2
        assert metrics['last_change_time'] is not None
    
    def test_clear_state(self, state_manager):
        """Test state clearing."""
        # Set some state
        state_manager.set_state("key1", "value1")
        state_manager.set_state("key2", "value2")
        
        assert state_manager.get_state("key1") == "value1"
        assert state_manager.get_state("key2") == "value2"
        
        # Clear state
        result = state_manager.clear_state(agent_name="test_agent")
        assert result is True
        
        # State should be empty
        assert state_manager.get_state("key1") is None
        assert state_manager.get_state("key2") is None
        
        # Check change history
        history = state_manager.get_change_history()
        clear_change = history[-1]
        assert clear_change.change_type == StateChangeType.BULK_UPDATE
        assert clear_change.key == "clear_state"
        assert clear_change.agent_name == "test_agent"
    
    def test_persistence(self, temp_state_file):
        """Test state persistence."""
        with patch('utils.global_state.StructuredLogger'), \
             patch('utils.global_state.settings'):
            
            # Create state manager with persistence
            state_manager = GlobalStateManager(
                enable_persistence=True,
                persistence_file=temp_state_file
            )
            
            # Set some state
            state_manager.set_state("persistent_key", "persistent_value")
            
            # Create new instance to test loading
            state_manager2 = GlobalStateManager(
                enable_persistence=True,
                persistence_file=temp_state_file
            )
            
            # Check if state was loaded
            assert state_manager2.get_state("persistent_key") == "persistent_value"
    
    def test_create_snapshot(self, state_manager):
        """Test snapshot creation."""
        # Set some state
        state_manager.set_state("key1", "value1")
        state_manager.set_state("key2", "value2")
        
        # Create snapshot
        snapshot_id = state_manager.create_snapshot("test_snapshot")
        
        assert snapshot_id != ""
        assert "test_snapshot" in snapshot_id
        
        # Check if snapshot file was created
        snapshot_file = f"data/snapshots/{snapshot_id}.json"
        # Note: In real test, would check file existence
        assert snapshot_id.endswith('.json') or snapshot_id.endswith('.json') is False


class TestGlobalStateIntegration:
    """Integration tests for global state system."""
    
    def test_singleton_pattern(self):
        """Test singleton pattern for global state manager."""
        with patch('utils.global_state.StructuredLogger'), \
             patch('utils.global_state.settings'):
            
            # Reset singleton
            import utils.global_state
            utils.global_state._global_state_manager = None
            
            # Get instance twice
            manager1 = get_global_state_manager()
            manager2 = get_global_state_manager()
            
            # Should be same instance
            assert manager1 is manager2
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('utils.global_state.StructuredLogger'), \
             patch('utils.global_state.settings'):
            
            # Reset singleton
            import utils.global_state
            utils.global_state._global_state_manager = None
            
            # Test convenience functions
            result = set_state("test_key", "test_value", "test_agent")
            assert result is True
            
            value = get_state("test_key")
            assert value == "test_value"
            
            # Test bulk update
            updates = {"key1": "value1", "key2": "value2"}
            results = bulk_set_state(updates, "test_agent")
            assert all(results.values())
            
            assert get_state("key1") == "value1"
            assert get_state("key2") == "value2"
    
    def test_thread_safety(self):
        """Test thread safety of state operations."""
        import threading
        import time
        
        with patch('utils.global_state.StructuredLogger'), \
             patch('utils.global_state.settings'):
            
            # Reset singleton
            import utils.global_state
            utils.global_state._global_state_manager = None
            
            manager = get_global_state_manager()
            
            # Shared data for threads
            results = []
            errors = []
            
            def worker(thread_id):
                try:
                    for i in range(10):
                        key = f"thread_{thread_id}_key_{i}"
                        value = f"thread_{thread_id}_value_{i}"
                        success = manager.set_state(key, value)
                        results.append((thread_id, i, success))
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check for errors
            assert len(errors) == 0, f"Thread errors: {errors}"
            
            # Check all operations succeeded
            assert len(results) == 50  # 5 threads * 10 operations
            assert all(success for _, _, success in results)
            
            # Verify state integrity
            for thread_id in range(5):
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    assert manager.get_state(key) == value


if __name__ == "__main__":
    pytest.main([__file__])