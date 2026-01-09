#!/usr/bin/env python3
"""
Metrics Monitor for Self-Healing System

Tracks system reliability metrics including:
- Cycle success/failure rates
- Crash events
- Healing events and recovery times
- Article collection statistics
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading


class MetricsMonitor:
    """
    Thread-safe metrics monitor for tracking Self-Healing System reliability.
    
    Tracks:
    - System uptime and cycle statistics
    - Crash events
    - Healing events with recovery durations
    - Total articles collected
    """
    
    def __init__(self, metrics_file: Optional[Path] = None):
        """
        Initialize the metrics monitor.
        
        Args:
            metrics_file: Path to JSON file storing metrics. Defaults to 
                        data/metrics/system_stats.json
        """
        if metrics_file is None:
            project_root = Path(__file__).parent.parent
            metrics_file = project_root / "data" / "metrics" / "system_stats.json"
        
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for file operations
        self._lock = threading.Lock()
        
        # Initialize metrics file if it doesn't exist
        if not self.metrics_file.exists():
            self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        """Initialize the metrics file with default structure."""
        initial_data = {
            'start_time': datetime.now().isoformat(),
            'total_cycles': 0,
            'successful_cycles': 0,
            'crashes': [],
            'heal_events': [],
            'total_articles_collected': 0
        }
        self._write_metrics(initial_data)
    
    def _read_metrics(self) -> Dict[str, Any]:
        """Read metrics from JSON file (thread-safe)."""
        with self._lock:
            try:
                if not self.metrics_file.exists():
                    self._initialize_metrics_file()
                
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # If file is corrupted, reinitialize
                print(f"⚠️  Warning: Failed to read metrics file: {e}. Reinitializing...")
                self._initialize_metrics_file()
                return self._read_metrics()
    
    def _write_metrics(self, data: Dict[str, Any]):
        """Write metrics to JSON file (thread-safe)."""
        with self._lock:
            try:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = self.metrics_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_file.replace(self.metrics_file)
            except Exception as e:
                print(f"❌ Error writing metrics file: {e}")
                raise
    
    def log_cycle(self, status: str, articles_count: int = 0):
        """
        Log a cycle completion.
        
        Args:
            status: Cycle status - 'success' or 'failed'
            articles_count: Number of articles collected in this cycle
        """
        metrics = self._read_metrics()
        
        # Increment total cycles
        metrics['total_cycles'] = metrics.get('total_cycles', 0) + 1
        
        # Increment successful cycles if status is 'success'
        if status.lower() == 'success':
            metrics['successful_cycles'] = metrics.get('successful_cycles', 0) + 1
        
        # Add articles to total
        if articles_count > 0:
            metrics['total_articles_collected'] = metrics.get('total_articles_collected', 0) + articles_count
        
        self._write_metrics(metrics)
    
    def log_crash(self, error_msg: str):
        """
        Log a crash event.
        
        Args:
            error_msg: Error message describing the crash
        """
        metrics = self._read_metrics()
        
        crash_event = {
            'timestamp': datetime.now().isoformat(),
            'error_message': error_msg
        }
        
        crashes = metrics.get('crashes', [])
        crashes.append(crash_event)
        metrics['crashes'] = crashes
        
        self._write_metrics(metrics)
    
    def log_heal_success(self, duration: float):
        """
        Log a successful healing event.
        
        Args:
            duration: Duration in seconds it took to fix the issue
        """
        metrics = self._read_metrics()
        
        heal_event = {
            'timestamp': datetime.now().isoformat(),
            'duration_to_fix': duration
        }
        
        heal_events = metrics.get('heal_events', [])
        heal_events.append(heal_event)
        metrics['heal_events'] = heal_events
        
        self._write_metrics(metrics)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate system reliability metrics.
        
        Returns:
            Dictionary containing:
            - uptime_percent: Percentage of successful cycles
            - mttr_seconds: Mean Time To Recovery (average healing duration)
            - recovery_rate_percent: Percentage of crashes that were recovered
            - total_cycles: Total number of cycles
            - successful_cycles: Number of successful cycles
            - total_crashes: Total number of crashes
            - total_heals: Total number of successful heals
            - total_articles: Total articles collected
            - start_time: When monitoring started
        """
        metrics = self._read_metrics()
        
        total_cycles = metrics.get('total_cycles', 0)
        successful_cycles = metrics.get('successful_cycles', 0)
        crashes = metrics.get('crashes', [])
        heal_events = metrics.get('heal_events', [])
        
        # Calculate uptime percentage
        if total_cycles > 0:
            uptime_percent = (successful_cycles / total_cycles) * 100
        else:
            uptime_percent = 0.0
        
        # Calculate MTTR (Mean Time To Recovery)
        if heal_events:
            durations = [event.get('duration_to_fix', 0) for event in heal_events]
            mttr_seconds = sum(durations) / len(durations)
        else:
            mttr_seconds = 0.0
        
        # Calculate recovery rate
        total_crashes = len(crashes)
        total_heals = len(heal_events)
        
        if total_crashes > 0:
            recovery_rate_percent = (total_heals / total_crashes) * 100
        else:
            recovery_rate_percent = 0.0
        
        return {
            'uptime_percent': round(uptime_percent, 2),
            'mttr_seconds': round(mttr_seconds, 2),
            'recovery_rate_percent': round(recovery_rate_percent, 2),
            'total_cycles': total_cycles,
            'successful_cycles': successful_cycles,
            'total_crashes': total_crashes,
            'total_heals': total_heals,
            'total_articles': metrics.get('total_articles_collected', 0),
            'start_time': metrics.get('start_time', 'unknown')
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get raw statistics from the metrics file.
        
        Returns:
            Dictionary with all raw metrics data
        """
        return self._read_metrics()
    
    def reset(self):
        """Reset all metrics (useful for testing or starting fresh)."""
        self._initialize_metrics_file()
    
    def print_summary(self):
        """Print a formatted summary of current metrics."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("SYSTEM RELIABILITY METRICS SUMMARY")
        print("="*80)
        print(f"Start Time: {metrics['start_time']}")
        print(f"Total Cycles: {metrics['total_cycles']}")
        print(f"Successful Cycles: {metrics['successful_cycles']}")
        print(f"Total Crashes: {metrics['total_crashes']}")
        print(f"Total Heals: {metrics['total_heals']}")
        print(f"Total Articles Collected: {metrics['total_articles']}")
        print("-"*80)
        print(f"Uptime: {metrics['uptime_percent']}%")
        print(f"MTTR (Mean Time To Recovery): {metrics['mttr_seconds']} seconds")
        print(f"Recovery Rate: {metrics['recovery_rate_percent']}%")
        print("="*80 + "\n")


# Convenience function for quick access
def get_metrics_monitor(metrics_file: Optional[Path] = None) -> MetricsMonitor:
    """
    Get or create a MetricsMonitor instance.
    
    Args:
        metrics_file: Optional path to metrics file
        
    Returns:
        MetricsMonitor instance
    """
    return MetricsMonitor(metrics_file)


if __name__ == "__main__":
    # Example usage and testing
    monitor = MetricsMonitor()
    
    # Simulate some events
    print("Testing Metrics Monitor...")
    
    # Log some cycles
    monitor.log_cycle('success', articles_count=5)
    monitor.log_cycle('success', articles_count=3)
    monitor.log_cycle('failed', articles_count=0)
    
    # Log a crash
    monitor.log_crash("Selector not found: article.item-news")
    
    # Log a successful heal
    monitor.log_heal_success(duration=12.5)
    
    # Log another cycle
    monitor.log_cycle('success', articles_count=7)
    
    # Print summary
    monitor.print_summary()
    
    # Show calculated metrics
    metrics = monitor.calculate_metrics()
    print("\nCalculated Metrics:")
    print(json.dumps(metrics, indent=2))

