#!/usr/bin/env python3
"""
Chaos Test Script for Self-Healing Multi-Agent System

This script demonstrates the paper's results by:
1. Launching run_production.py in background mode
2. Waiting 10s for system stability
3. Automatically injecting error by modifying agents/opinion_search_agent.py
4. Observing healing agent detection and Llama 3 repair
5. Measuring Mean Time To Recovery (MTTR)

Usage:
    python scripts/chaos_test.py [--quick] [--verbose]
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger


class ChaosTest:
    """
    Chaos testing framework for self-healing system validation
    
    This class orchestrates the complete chaos test workflow:
    - System startup and monitoring
    - Error injection simulation
    - Healing process observation
    - MTTR measurement and reporting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger("chaos_test")
        
        # Test configuration
        self.stabilization_time = self.config.get('stabilization_time', 10)  # seconds
        self.test_duration = self.config.get('test_duration', 120)  # seconds
        self.observation_window = self.config.get('observation_window', 60)  # seconds
        self.verbose = self.config.get('verbose', False)
        self.quick_mode = self.config.get('quick_mode', False)
        
        # File paths
        self.project_root = project_root
        self.production_script = project_root / "scripts" / "run_production.py"
        self.opinion_agent_file = project_root / "agents" / "opinion_search_agent.py"
        self.backup_file = project_root / "agents" / "opinion_search_agent.py.backup"
        self.healing_metrics_file = project_root / "data" / "healing_metrics.json"
        self.production_log_file = project_root / "data" / "production" / "debug_log.txt"
        
        # Test state
        self.production_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.error_injection_time: Optional[float] = None
        self.healing_detected_time: Optional[float] = None
        self.healing_completed_time: Optional[float] = None
        self.test_results: Dict[str, Any] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.system_logs: List[str] = []
        
        self.logger.info("Chaos Test initialized")
    
    async def run_test(self) -> Dict[str, Any]:
        """
        Run the complete chaos test
        
        Returns:
            Test results with MTTR measurements
        """
        try:
            self.logger.info("Starting Chaos Test for Self-Healing System")
            print("=" * 80)
            print("CHAOS TEST: Self-Healing Multi-Agent System")
            print("=" * 80)
            
            # Phase 1: Pre-test validation
            await self._pre_test_validation()
            
            # Phase 2: Start production system
            await self._start_production_system()
            
            # Phase 3: Wait for stabilization
            await self._wait_for_stabilization()
            
            # Phase 4: Inject error
            await self._inject_error()
            
            # Phase 5: Monitor healing process
            await self._monitor_healing_process()
            
            # Phase 6: Analyze results
            results = await self._analyze_results()
            
            # Phase 7: Cleanup
            await self._cleanup()
            
            # Display results
            self._display_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Chaos test failed: {e}")
            await self._cleanup()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def _pre_test_validation(self):
        """Validate test prerequisites"""
        print("\nPhase 1: Pre-test Validation")
        
        # Check required files
        required_files = [
            self.production_script,
            self.opinion_agent_file,
            project_root / "agents" / "healing_agent.py",
            project_root / "core" / "llm_client.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
            print(f"  ✓ Found: {file_path.name}")
        
        # Check Ollama availability
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                llama_available = any('llama' in model['name'].lower() for model in models)
                if llama_available:
                    print("  ✓ Ollama with Llama model available")
                else:
                    print("  ⚠️  Ollama available but no Llama model found")
            else:
                print("  ⚠️  Ollama not responding (will use mock)")
        except Exception:
            print("  ⚠️  Cannot connect to Ollama (will use mock)")
        
        print("✅ Pre-test validation completed\n")
    
    async def _start_production_system(self):
        """Start the production system in background"""
        print("🚀 Phase 2: Starting Production System")
        
        try:
            # Start production system
            self.start_time = time.time()
            
            cmd = [
                sys.executable, 
                str(self.production_script),
                "--background" if self.quick_mode else "--full"
            ]
            
            self.production_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"  ✓ Production system started (PID: {self.production_process.pid})")
            
            # Start monitoring logs
            self._start_log_monitoring()
            
            print("✅ Production system started\n")
            
        except Exception as e:
            raise Exception(f"Failed to start production system: {e}")
    
    def _start_log_monitoring(self):
        """Start monitoring system logs"""
        self.monitoring_active = True
        
        def monitor_logs():
            while self.monitoring_active and self.production_process:
                try:
                    # Check if process is still running
                    if self.production_process.poll() is not None:
                        self.logger.warning("Production process terminated")
                        break
                    
                    # Read production logs if available
                    if self.production_log_file.exists():
                        with open(self.production_log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines and len(self.system_logs) < len(lines):
                                new_lines = lines[len(self.system_logs):]
                                for line in new_lines:
                                    if self.verbose:
                                        print(f"[PROD] {line.strip()}")
                                    self.system_logs.append(line.strip())
                    
                    time.sleep(1)
                    
                except Exception as e:
                    if self.monitoring_active:
                        self.logger.error(f"Log monitoring error: {e}")
            
        self.monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
        self.monitor_thread.start()
    
    async def _wait_for_stabilization(self):
        """Wait for system to stabilize"""
        print(f"⏳ Phase 3: Waiting for Stabilization ({self.stabilization_time}s)")
        
        wait_time = 5 if self.quick_mode else self.stabilization_time
        
        for i in range(wait_time):
            remaining = wait_time - i
            print(f"  {remaining} seconds remaining...", end='\r')
            await asyncio.sleep(1)
        
        print(f"\n  ✓ System stabilization completed")
        print("✅ Stabilization phase completed\n")
    
    async def _inject_error(self):
        """Inject error by modifying CSS selector"""
        print("💥 Phase 4: Error Injection")
        
        try:
            # Create backup
            shutil.copy2(self.opinion_agent_file, self.backup_file)
            print(f"  ✓ Created backup: {self.backup_file.name}")
            
            # Read current file
            with open(self.opinion_agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace the correct selector with wrong one
            original_selector = "article.item-news"
            wrong_selector = "article.wrong-class"
            
            if original_selector in content:
                modified_content = content.replace(original_selector, wrong_selector)
                
                # Write modified content
                with open(self.opinion_agent_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.error_injection_time = time.time()
                injection_delay = self.error_injection_time - self.start_time
                
                print(f"  ✓ Modified CSS selector: '{original_selector}' → '{wrong_selector}'")
                print(f"  ✓ Error injected at T+{injection_delay:.1f}s")
                
            else:
                raise Exception(f"Could not find selector '{original_selector}' in opinion agent")
            
            print("✅ Error injection completed\n")
            
        except Exception as e:
            raise Exception(f"Failed to inject error: {e}")
    
    async def _monitor_healing_process(self):
        """Monitor the healing process"""
        print("🔧 Phase 5: Monitoring Healing Process")
        
        observation_time = 30 if self.quick_mode else self.observation_window
        
        print(f"  Monitoring healing process for {observation_time} seconds...")
        print("  Waiting for Healing Agent to detect error and fix...")
        
        healing_detected = False
        healing_completed = False
        
        start_monitor = time.time()
        
        while time.time() - start_monitor < observation_time:
            # Check logs for healing indicators
            for log_line in self.system_logs[-10:]:  # Check recent logs
                if "healing" in log_line.lower() and not healing_detected:
                    self.healing_detected_time = time.time()
                    healing_detection_delay = self.healing_detected_time - self.error_injection_time
                    print(f"\n  🎯 Healing detected at T+{healing_detection_delay:.1f}s!")
                    healing_detected = True
                
                if "fix applied" in log_line.lower() and healing_detected and not healing_completed:
                    self.healing_completed_time = time.time()
                    healing_completion_delay = self.healing_completed_time - self.error_injection_time
                    print(f"  ✅ Fix applied at T+{healing_completion_delay:.1f}s!")
                    healing_completed = True
                    break
            
            # Check if healing metrics file is updated
            if self.healing_metrics_file.exists():
                try:
                    with open(self.healing_metrics_file, 'r') as f:
                        metrics = json.load(f)
                        if metrics.get('successful_repairs', 0) > 0:
                            self.healing_completed_time = time.time()
                            healing_completion_delay = self.healing_completed_time - self.error_injection_time
                            print(f"\n  📊 Healing metrics updated at T+{healing_completion_delay:.1f}s!")
                            healing_completed = True
                            break
                except:
                    pass
            
            await asyncio.sleep(1)
        
        if not healing_detected:
            print("  ⚠️  Healing process not detected within observation window")
        elif not healing_completed:
            print("  ⚠️  Healing completion not confirmed within observation window")
        
        print("✅ Healing monitoring completed\n")
    
    async def _analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and calculate MTTR"""
        print("📊 Phase 6: Analyzing Results")
        
        # Calculate timing metrics
        end_time = time.time()
        total_test_time = end_time - self.start_time
        
        # Calculate MTTR
        mttr = None
        if self.healing_detected_time and self.healing_completed_time:
            mttr = self.healing_completed_time - self.error_injection_time
        elif self.healing_completed_time:
            mttr = self.healing_completed_time - self.error_injection_time
        
        # Check if original selector was restored
        selector_restored = False
        try:
            with open(self.opinion_agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "article.item-news" in content:
                    selector_restored = True
        except:
            pass
        
        # Load healing metrics if available
        healing_metrics = {}
        if self.healing_metrics_file.exists():
            try:
                with open(self.healing_metrics_file, 'r') as f:
                    healing_metrics = json.load(f)
            except:
                pass
        
        # Prepare results
        results = {
            'test_success': selector_restored and mttr is not None,
            'test_timestamp': datetime.now().isoformat(),
            'timing': {
                'total_test_time_seconds': total_test_time,
                'stabilization_time': self.stabilization_time,
                'error_injection_time': self.error_injection_time - self.start_time if self.error_injection_time else None,
                'healing_detection_time': self.healing_detected_time - self.start_time if self.healing_detected_time else None,
                'healing_completion_time': self.healing_completed_time - self.start_time if self.healing_completed_time else None,
                'mttr_seconds': mttr
            },
            'healing_effectiveness': {
                'selector_restored': selector_restored,
                'healing_detected': self.healing_detected_time is not None,
                'healing_completed': self.healing_completed_time is not None
            },
            'system_metrics': healing_metrics,
            'test_configuration': {
                'stabilization_time': self.stabilization_time,
                'observation_window': self.observation_window,
                'quick_mode': self.quick_mode,
                'verbose': self.verbose
            },
            'logs_sample': self.system_logs[-20:] if self.system_logs else []
        }
        
        self.test_results = results
        print("✅ Results analysis completed\n")
        
        return results
    
    def _display_results(self, results: Dict[str, Any]):
        """Display test results"""
        print("📋 Phase 7: Test Results")
        print("=" * 80)
        
        # Overall result
        success = results.get('test_success', False)
        status = "✅ SUCCESS" if success else "❌ FAILURE"
        print(f"\nOverall Test Status: {status}")
        
        # Timing results
        timing = results.get('timing', {})
        mttr = timing.get('mttr_seconds')
        
        print(f"\n⏱️  Timing Metrics:")
        print(f"  Total Test Time: {timing.get('total_test_time_seconds', 0):.1f}s")
        print(f"  Error Injection: T+{timing.get('error_injection_time', 0):.1f}s")
        print(f"  Healing Detection: T+{timing.get('healing_detection_time', 0):.1f}s" if timing.get('healing_detection_time') else "  Healing Detection: Not detected")
        print(f"  Healing Completion: T+{timing.get('healing_completion_time', 0):.1f}s" if timing.get('healing_completion_time') else "  Healing Completion: Not detected")
        
        if mttr:
            print(f"\n🎯 Mean Time To Recovery (MTTR): {mttr:.2f} seconds")
            if mttr < 60:
                print("  ✅ MTTR meets target (< 60 seconds)")
            else:
                print("  ⚠️  MTTR exceeds target (> 60 seconds)")
        else:
            print("\n⚠️  MTTR could not be calculated")
        
        # Effectiveness results
        effectiveness = results.get('healing_effectiveness', {})
        print(f"\n🔧 Healing Effectiveness:")
        print(f"  Error Detected: {'✅' if effectiveness.get('healing_detected') else '❌'}")
        print(f"  Fix Applied: {'✅' if effectiveness.get('healing_completed') else '❌'}")
        print(f"  Selector Restored: {'✅' if effectiveness.get('selector_restored') else '❌'}")
        
        # System metrics
        system_metrics = results.get('system_metrics', {})
        if system_metrics:
            print(f"\n📊 System Healing Metrics:")
            print(f"  Successful Repairs: {system_metrics.get('successful_repairs', 0)}")
            print(f"  Failed Repairs: {system_metrics.get('failed_repairs', 0)}")
            if system_metrics.get('successful_repairs', 0) > 0:
                success_rate = (system_metrics.get('successful_repairs', 0) / 
                             (system_metrics.get('successful_repairs', 0) + system_metrics.get('failed_repairs', 0))) * 100
                print(f"  Success Rate: {success_rate:.1f}%")
        
        # Research paper conclusion
        print(f"\n📝 Research Paper Conclusion:")
        if success and mttr and mttr < 60:
            print("  🎉 The self-healing system successfully demonstrates:")
            print("     • Automated error detection")
            print("     • LLM-powered diagnosis and repair") 
            print("     • Hot-reload without system downtime")
            print("     • MTTR < 60 seconds target achieved")
        elif success:
            print("  ✅ Self-healing functional, but MTTR needs optimization")
        else:
            print("  ❌ Self-healing system requires further development")
        
        print("\n" + "=" * 80)
    
    async def _cleanup(self):
        """Clean up test environment"""
        print("\n🧹 Phase 8: Cleanup")
        
        try:
            # Stop monitoring
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            
            # Stop production process
            if self.production_process:
                try:
                    self.production_process.terminate()
                    self.production_process.wait(timeout=5)
                    print("  ✓ Production process stopped")
                except:
                    try:
                        self.production_process.kill()
                        self.production_process.wait(timeout=2)
                        print("  ✓ Production process force killed")
                    except:
                        print("  ⚠️  Could not stop production process")
            
            # Restore original file
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, self.opinion_agent_file)
                self.backup_file.unlink()
                print("  ✓ Original opinion agent restored")
            
            # Save test results
            if self.test_results:
                results_file = project_root / "data" / f"chaos_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump(self.test_results, f, indent=2)
                print(f"  ✓ Test results saved: {results_file.name}")
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            print(f"  ⚠️  Cleanup error: {e}")


async def main():
    """Main chaos test entry point"""
    parser = argparse.ArgumentParser(description="Chaos Test for Self-Healing System")
    parser.add_argument("--quick", action="store_true", help="Run quick test (30s instead of 2min)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--stabilization", type=int, default=10, help="Stabilization time in seconds")
    parser.add_argument("--observation", type=int, default=60, help="Observation window in seconds")
    
    args = parser.parse_args()
    
    # Configure test
    config = {
        'stabilization_time': args.stabilization,
        'observation_window': args.observation,
        'quick_mode': args.quick,
        'verbose': args.verbose
    }
    
    # Run chaos test
    chaos_test = ChaosTest(config)
    results = await chaos_test.run_test()
    
    # Return exit code based on success
    exit_code = 0 if results.get('test_success', False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())