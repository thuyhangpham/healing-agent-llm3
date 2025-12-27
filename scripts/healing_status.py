#!/usr/bin/env python3
"""
Healing System Status Monitor

This script provides real-time monitoring of the self-healing system,
including metrics, performance, and health status.

Features:
- Real-time healing metrics dashboard
- Performance target tracking
- Error pattern analysis
- System health monitoring
- Export capabilities
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logger import get_logger
from utils.config import load_config


class HealingSystemMonitor:
    """Monitor for healing system status and metrics"""
    
    def __init__(self):
        self.logger = get_logger("healing_monitor")
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.data_dir = project_root / "data"
        
        # Load configuration
        self.healing_config = load_config(str(self.config_dir / "healing.yaml"))
        self.agents_config = load_config(str(self.config_dir / "agents.yaml"))
        
        # Performance targets
        self.mttr_target = self.healing_config.get("healing_agent", {}).get("mttr_target_seconds", 60)
        self.success_rate_target = self.healing_config.get("healing_agent", {}).get("success_rate_target", 0.8)
        
        # Initialize components
        self.healing_agent = None
        self.metrics = None
        
    async def initialize(self) -> bool:
        """Initialize monitoring components"""
        try:
            # Import healing components
            from agents.healing_agent import HealingAgent
            from core.healing_metrics import HealingMetrics
            
            # Initialize healing agent
            healing_config = self.healing_config.get("healing_agent", {})
            self.healing_agent = HealingAgent("healing_agent", healing_config)
            await self.healing_agent.initialize()
            
            # Initialize metrics
            metrics_config = self.healing_config.get("metrics", {})
            self.metrics = HealingMetrics(metrics_config)
            await self.metrics.initialize()
            
            self.logger.info("Healing system monitor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitor: {e}")
            return False
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        print("🔍 Starting Healing System Monitoring...")
        print(f"📊 Update Interval: {interval} seconds")
        print(f"🎯 MTTR Target: {self.mttr_target}s")
        print(f"🎯 Success Rate Target: {self.success_rate_target * 100:.1f}%")
        print("\n" + "="*80)
        
        try:
            while True:
                await self._display_status_dashboard()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            await self._export_final_report()
    
    async def _display_status_dashboard(self):
        """Display comprehensive status dashboard"""
        # Clear screen (platform dependent)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀 SELF-HEALING SYSTEM DASHBOARD")
        print("="*80)
        print(f"🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Get healing agent status
        healing_status = await self.healing_agent.get_status()
        
        # System Status Section
        print("\n📊 SYSTEM STATUS")
        print("-"*40)
        print(f"🤖 Healing Agent: {healing_status.get('status', 'Unknown')}")
        print(f"🔄 Auto-Heal: {'✅ Enabled' if healing_status.get('auto_heal_enabled', False) else '❌ Disabled'}")
        print(f"📈 Active Operations: {healing_status.get('active_operations', 0)}")
        print(f"📋 Total Healing Ops: {healing_status.get('total_healing_operations', 0)}")
        
        # Performance Metrics
        print("\n📈 PERFORMANCE METRICS")
        print("-"*40)
        success_rate = healing_status.get('success_rate', 0)
        avg_mttr = healing_status.get('average_mttr', 0)
        
        # MTTR Status
        mttr_status = "✅" if avg_mttr <= self.mttr_target else "❌"
        mttr_percentage = (avg_mttr / self.mttr_target * 100) if self.mttr_target > 0 else 0
        
        print(f"📊 MTTR: {avg_mttr:.1f}s {mttr_status} (Target: {self.mttr_target}s)")
        if avg_mttr > 0:
            print(f"   📈 MTTR Performance: {mttr_percentage:.1f}% of target")
        
        # Success Rate Status
        sr_status = "✅" if success_rate >= self.success_rate_target else "❌"
        sr_percentage = success_rate * 100
        
        print(f"🎯 Success Rate: {sr_percentage:.1f}% {sr_status} (Target: {self.success_rate_target * 100:.1f}%)")
        
        # Recent Results
        recent_results = healing_status.get('recent_results', [])
        if recent_results:
            print(f"\n📋 RECENT HEALING OPERATIONS (Last {len(recent_results)})")
            print("-"*40)
            for i, result in enumerate(recent_results[-5:], 1):  # Show last 5
                status_icon = "✅" if result.get('success', False) else "❌"
                time_str = f"{result.get('time_to_repair', 0):.1f}s"
                print(f"  {i}. {status_icon} {result.get('error_id', 'Unknown')} - {time_str}")
                print(f"     {result.get('fix_description', 'No description')[:60]}...")
        
        # Error Patterns
        error_patterns = healing_status.get('error_patterns', {})
        if error_patterns:
            print(f"\n🔍 ERROR PATTERNS (Top 5)")
            print("-"*40)
            sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            for i, (pattern, count) in enumerate(sorted_patterns[:5], 1):
                print(f"  {i}. {pattern}: {count} occurrences")
        
        # System Health
        print("\n🏥 SYSTEM HEALTH")
        print("-"*40)
        await self._display_system_health()
        
        # Performance Targets Summary
        print("\n🎯 PERFORMANCE TARGETS")
        print("-"*40)
        
        # MTTR Target
        mttr_met = avg_mttr <= self.mttr_target if avg_mttr > 0 else False
        mttr_icon = "✅" if mttr_met else "❌"
        print(f"📊 MTTR Target: {mttr_icon} {avg_mttr:.1f}s / {self.mttr_target}s")
        
        # Success Rate Target
        sr_met = success_rate >= self.success_rate_target
        sr_icon = "✅" if sr_met else "❌"
        print(f"🎯 Success Rate: {sr_icon} {sr_percentage:.1f}% / {self.success_rate_target * 100:.1f}%")
        
        # Overall System Status
        overall_status = "🟢 HEALTHY" if (mttr_met and sr_met) else "🟡 NEEDS ATTENTION"
        if not mttr_met and not sr_met:
            overall_status = "🔴 CRITICAL"
        
        print(f"\n🏥 OVERALL STATUS: {overall_status}")
        
        # Footer
        print("\n" + "="*80)
        print("Press Ctrl+C to stop monitoring")
        print("Commands: [E]xport data, [H]ealth check, [Q]uit")
    
    async def _display_system_health(self):
        """Display detailed system health"""
        health_checks = [
            ("Configuration Files", self._check_config_health),
            ("Data Directories", self._check_data_health),
            ("Dependencies", self._check_dependency_health),
            ("Services", self._check_service_health)
        ]
        
        for check_name, check_func in health_checks:
            try:
                status = await check_func()
                icon = "✅" if status else "❌"
                print(f"  {icon} {check_name}")
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
    
    async def _check_config_health(self) -> bool:
        """Check configuration health"""
        required_files = [
            self.config_dir / "healing.yaml",
            self.config_dir / "agents.yaml"
        ]
        return all(file.exists() for file in required_files)
    
    async def _check_data_health(self) -> bool:
        """Check data directories health"""
        required_dirs = [
            self.data_dir / "backups",
            self.data_dir / "metrics",
            self.data_dir / "logs",
            self.data_dir / "exports"
        ]
        return all(dir.exists() for dir in required_dirs)
    
    async def _check_dependency_health(self) -> bool:
        """Check dependency health"""
        try:
            import aiohttp
            import yaml
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            return False
    
    async def _check_service_health(self) -> bool:
        """Check service health"""
        try:
            # Check if healing agent is responsive
            status = await self.healing_agent.get_status()
            return status is not None
        except Exception:
            return False
    
    async def _export_final_report(self):
        """Export final monitoring report"""
        try:
            print("\n📊 Generating final report...")
            
            # Get comprehensive metrics
            healing_status = await self.healing_agent.get_status()
            performance_report = await self.metrics.get_performance_report(7)  # Last 7 days
            
            # Create report
            report = {
                "report_generated": datetime.now().isoformat(),
                "monitoring_period": "Last 7 days",
                "healing_status": healing_status,
                "performance_report": performance_report,
                "system_health": {
                    "config_health": await self._check_config_health(),
                    "data_health": await self._check_data_health(),
                    "dependency_health": await self._check_dependency_health(),
                    "service_health": await self._check_service_health()
                }
            }
            
            # Export to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.data_dir / "exports" / f"healing_report_{timestamp}.json"
            
            report_file.parent.mkdir(exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Report exported to: {report_file}")
            
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
    
    async def run_interactive_mode(self):
        """Run interactive monitoring mode"""
        print("\n🎮 INTERACTIVE MODE")
        print("="*40)
        print("Available commands:")
        print("  status  - Show current status")
        print("  metrics - Show detailed metrics")
        print("  export  - Export data")
        print("  health  - Run health checks")
        print("  patterns - Show error patterns")
        print("  quit    - Exit monitor")
        print("="*40)
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "q":
                    break
                elif command == "status":
                    await self._display_status_dashboard()
                elif command == "metrics":
                    await self._show_detailed_metrics()
                elif command == "export":
                    await self._export_data()
                elif command == "health":
                    await self._run_health_checks()
                elif command == "patterns":
                    await self._show_error_patterns()
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    async def _show_detailed_metrics(self):
        """Show detailed metrics"""
        try:
            report = await self.metrics.get_performance_report(7)
            
            print("\n📊 DETAILED METRICS (Last 7 Days)")
            print("-"*50)
            
            if 'error' in report:
                print(f"❌ {report['error']}")
                return
            
            print(f"📈 Total Events: {report.get('total_events', 0)}")
            print(f"✅ Successful: {report.get('successful_events', 0)}")
            print(f"❌ Failed: {report.get('total_events', 0) - report.get('successful_events', 0)}")
            print(f"📊 MTTR: {report.get('mttr', 0):.1f}s")
            print(f"🎯 Success Rate: {report.get('success_rate', 0)*100:.1f}%")
            
            # Category breakdown
            category_breakdown = report.get('category_breakdown', {})
            if category_breakdown:
                print(f"\n📋 BREAKDOWN BY CATEGORY:")
                for category, stats in category_breakdown.items():
                    print(f"  {category}: {stats.get('success_rate', 0)*100:.1f}% ({stats.get('total', 0)} events)")
            
        except Exception as e:
            print(f"❌ Failed to get metrics: {e}")
    
    async def _export_data(self):
        """Export healing data"""
        try:
            print("📤 Exporting healing data...")
            
            # Export JSON
            json_file = await self.metrics.export_research_data("json", 7)
            if json_file:
                print(f"✅ JSON exported: {json_file}")
            
            # Export CSV
            csv_file = await self.metrics.export_research_data("csv", 7)
            if csv_file:
                print(f"✅ CSV exported: {csv_file}")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    async def _run_health_checks(self):
        """Run comprehensive health checks"""
        print("\n🏥 RUNNING HEALTH CHECKS")
        print("-"*40)
        
        health_checks = [
            ("Configuration", self._check_config_health),
            ("Data Directories", self._check_data_health),
            ("Dependencies", self._check_dependency_health),
            ("Services", self._check_service_health)
        ]
        
        all_healthy = True
        
        for check_name, check_func in health_checks:
            try:
                status = await check_func()
                icon = "✅" if status else "❌"
                print(f"  {icon} {check_name}")
                if not status:
                    all_healthy = False
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
                all_healthy = False
        
        overall_status = "🟢 HEALTHY" if all_healthy else "🔴 UNHEALTHY"
        print(f"\n🏥 OVERALL HEALTH: {overall_status}")
    
    async def _show_error_patterns(self):
        """Show error patterns analysis"""
        try:
            healing_status = await self.healing_agent.get_status()
            error_patterns = healing_status.get('error_patterns', {})
            
            print("\n🔍 ERROR PATTERNS ANALYSIS")
            print("-"*50)
            
            if not error_patterns:
                print("No error patterns detected yet.")
                return
            
            # Sort by frequency
            sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            
            print("Top Error Patterns:")
            for i, (pattern, count) in enumerate(sorted_patterns[:10], 1):
                bar_length = min(count * 2, 20)  # Scale bar
                bar = "█" * bar_length
                print(f"  {i:2d}. {pattern:<30} {count:>3} {bar}")
                
        except Exception as e:
            print(f"❌ Failed to analyze patterns: {e}")


async def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Healing System Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--single", action="store_true", help="Show single status update")
    
    args = parser.parse_args()
    
    monitor = HealingSystemMonitor()
    
    if not await monitor.initialize():
        print("❌ Failed to initialize monitor")
        return 1
    
    try:
        if args.single:
            await monitor._display_status_dashboard()
        elif args.interactive:
            await monitor.run_interactive_mode()
        else:
            await monitor.start_monitoring(args.interval)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))