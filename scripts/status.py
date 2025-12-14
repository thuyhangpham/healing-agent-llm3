#!/usr/bin/env python3
"""
System Status Checker

CLI utility for checking system status, agent health,
and healing metrics.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import settings
from utils.logger import StructuredLogger


def check_system_status():
    """Check overall system status."""
    print("=== ETL Sentiment System Status ===")
    print(f"Version: {settings.app_version}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Data Directory: {settings.data_dir}")
    print()
    
    # Check Ollama connectivity
    print("=== Ollama Status ===")
    try:
        import requests
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama connected at {settings.ollama_base_url}")
            print(f"✓ Available models: {[m['name'] for m in models]}")
        else:
            print(f"✗ Ollama connection failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Ollama connection error: {e}")
    print()
    
    # Check directories
    print("=== Directory Status ===")
    directories = [
        settings.data_dir,
        "data/collected_data",
        "data/logs", 
        "data/metrics"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (missing)")
    print()


def check_agent_status():
    """Check status of individual agents."""
    print("=== Agent Status ===")
    # TODO: Implement actual agent status checking
    print("Agent status checking not yet implemented")
    print()


def check_healing_metrics():
    """Check healing operation metrics."""
    print("=== Healing Metrics ===")
    metrics_file = settings.metrics_file
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            successful = metrics.get('successful_repairs', 0)
            failed = metrics.get('failed_repairs', 0)
            total = successful + failed
            
            if total > 0:
                success_rate = (successful / total) * 100
                avg_mttr = metrics.get('total_mttr', 0) / successful if successful > 0 else 0
                
                print(f"Total Repairs: {total}")
                print(f"Successful: {successful}")
                print(f"Failed: {failed}")
                print(f"Success Rate: {success_rate:.1f}%")
                print(f"Average MTTR: {avg_mttr:.2f} seconds")
            else:
                print("No healing operations recorded yet")
        except Exception as e:
            print(f"Error reading metrics: {e}")
    else:
        print("No metrics file found")
    print()


def show_recent_errors():
    """Show recent system errors."""
    print("=== Recent Errors ===")
    log_file = settings.log_file
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Show last 10 error lines
            error_lines = [line for line in lines[-10:] if '"level":"ERROR"' in line]
            
            if error_lines:
                for line in error_lines:
                    try:
                        log_entry = json.loads(line.strip())
                        print(f"{log_entry['timestamp']}: {log_entry['message']}")
                    except:
                        print(line.strip())
            else:
                print("No recent errors found")
        except Exception as e:
            print(f"Error reading log file: {e}")
    else:
        print("No log file found")
    print()


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Sentiment System Status")
    parser.add_argument("--system", action="store_true", help="Show system status")
    parser.add_argument("--agents", action="store_true", help="Show agent status")
    parser.add_argument("--healing", action="store_true", help="Show healing metrics")
    parser.add_argument("--errors", action="store_true", help="Show recent errors")
    parser.add_argument("--all", action="store_true", help="Show all status information")
    
    args = parser.parse_args()
    
    if args.all or not any([args.system, args.agents, args.healing, args.errors]):
        check_system_status()
        check_agent_status()
        check_healing_metrics()
        show_recent_errors()
    else:
        if args.system:
            check_system_status()
        if args.agents:
            check_agent_status()
        if args.healing:
            check_healing_metrics()
        if args.errors:
            show_recent_errors()


if __name__ == "__main__":
    main()