#!/usr/bin/env python3
"""
Simple Chaos Test Script for Self-Healing Multi-Agent System

This script demonstrates the paper's results by:
1. Launching run_production.py in background mode
2. Waiting 10s for system stability
3. Automatically injecting error by modifying agents/opinion_search_agent.py
4. Observing healing agent detection and Llama 3 repair
5. Measuring Mean Time To Recovery (MTTR)

Usage:
    python scripts/simple_chaos_test.py [--quick]
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleChaosTest:
    """Simple chaos testing framework for self-healing system validation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # Test configuration
        self.stabilization_time = self.config.get('stabilization_time', 10)
        self.observation_window = self.config.get('observation_window', 60)
        self.quick_mode = self.config.get('quick_mode', False)
        
        # File paths
        self.project_root = project_root
        self.production_script = project_root / "scripts" / "run_production.py"
        self.opinion_agent_file = project_root / "agents" / "opinion_search_agent.py"
        self.backup_file = project_root / "agents" / "opinion_search_agent.py.backup"
        
        # Test state
        self.production_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.error_injection_time: Optional[float] = None
        self.healing_detected_time: Optional[float] = None
        self.healing_completed_time: Optional[float] = None
        self.test_results: Dict[str, Any] = {}
        
        print("Chaos Test initialized")
    
    def _setup_logger(self):
        """Simple logger setup"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger("chaos_test")
    
    async def run_test(self) -> Dict[str, Any]:
        """Run complete chaos test"""
        try:
            print("Starting Chaos Test for Self-Healing System")
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
            print(f"Chaos test failed: {e}")
            await self._cleanup()
            return {
                'success': False,
                'error': str(e),
                'traceback': str(e.__traceback__) if e.__traceback__ else None
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
            if file_path.exists():
                print(f"  OK: {file_path.name}")
            else:
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Check Ollama availability
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                llama_available = any('llama' in model['name'].lower() for model in models)
                if llama_available:
                    print("  OK: Ollama with Llama model available")
                else:
                    print("  WARNING: Ollama available but no Llama model found")
            else:
                print("  WARNING: Ollama not responding (will use mock)")
        except Exception:
            print("  WARNING: Cannot connect to Ollama (will use mock)")
        
        print("Pre-test validation completed\n")
    
    async def _start_production_system(self):
        """Start production system in background"""
        print("Phase 2: Starting Production System")
        
        try:
            # Start production system
            self.start_time = time.time()
            
            cmd = [
                sys.executable, 
                str(self.production_script),
                "--background" if self.quick_mode else "--full",
                "--duration", "120"  # Run for 2 minutes
            ]
            
            self.production_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"  Production system started (PID: {self.production_process.pid})")
            print("Production system started\n")
            
        except Exception as e:
            raise Exception(f"Failed to start production system: {e}")
    
    async def _wait_for_stabilization(self):
        """Wait for system to stabilize"""
        wait_time = 5 if self.quick_mode else self.stabilization_time
        print(f"Phase 3: Waiting for Stabilization ({wait_time}s)")
        
        for i in range(wait_time):
            remaining = wait_time - i
            print(f"  {remaining} seconds remaining...", end='\r')
            await asyncio.sleep(1)
        
        print(f"\n  System stabilization completed")
        print("Stabilization phase completed\n")
    
    async def _inject_error(self):
        """Inject error by modifying CSS selector"""
        print("Phase 4: Error Injection")
        
        try:
            # Create backup
            shutil.copy2(self.opinion_agent_file, self.backup_file)
            print(f"  Created backup: {self.backup_file.name}")
            
            # Read current file
            with open(self.opinion_agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace correct selector with wrong one
            original_selector = "article.item-news"
            wrong_selector = "article.wrong-class"
            
            if original_selector in content:
                modified_content = content.replace(original_selector, wrong_selector)
                
                # Write modified content
                with open(self.opinion_agent_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.error_injection_time = time.time()
                injection_delay = self.error_injection_time - self.start_time
                
                print(f"  Modified CSS selector: '{original_selector}' -> '{wrong_selector}'")
                print(f"  Error injected at T+{injection_delay:.1f}s")
                
            else:
                raise Exception(f"Could not find selector '{original_selector}' in opinion agent")
            
            print("Error injection completed\n")
            
        except Exception as e:
            raise Exception(f"Failed to inject error: {e}")
    
    async def _monitor_healing_process(self):
        """Monitor the healing process"""
        observation_time = 30 if self.quick_mode else self.observation_window
        
        print(f"Phase 5: Monitoring Healing Process")
        print(f"  Monitoring healing process for {observation_time} seconds...")
        print("  Waiting for Healing Agent to detect error and fix...")
        
        healing_detected = False
        healing_completed = False
        
        start_monitor = time.time()
        
        # Simple file monitoring for healing completion
        while time.time() - start_monitor < observation_time:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Check if the error was fixed (selector restored)
            try:
                with open(self.opinion_agent_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "article.item-news" in content:
                    if not healing_completed:
                        self.healing_completed_time = time.time()
                        healing_completion_delay = self.healing_completed_time - self.error_injection_time
                        print(f"\n  FIX APPLIED at T+{healing_completion_delay:.1f}s!")
                        healing_completed = True
                        break
                        
            except Exception:
                pass  # File might be being written
        
        if not healing_detected:
            print("  WARNING: Healing process not detected within observation window")
        elif not healing_completed:
            print("  WARNING: Healing completion not confirmed within observation window")
        
        print("Healing monitoring completed\n")
    
    async def _analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and calculate MTTR"""
        print("Phase 6: Analyzing Results")
        
        # Calculate timing metrics
        end_time = time.time()
        total_test_time = end_time - self.start_time
        
        # Calculate MTTR
        mttr = None
        if self.healing_completed_time:
            mttr = self.healing_completed_time - self.error_injection_time
        
        # Check if original selector was restored
        selector_restored = False
        try:
            with open(self.opinion_agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "article.item-news" in content:
                    selector_restored = True
        except Exception:
            pass
        
        # Prepare results
        results = {
            'test_success': selector_restored and mttr is not None,
            'test_timestamp': datetime.now().isoformat(),
            'timing': {
                'total_test_time_seconds': total_test_time,
                'stabilization_time': self.stabilization_time,
                'error_injection_time': self.error_injection_time - self.start_time if self.error_injection_time else None,
                'healing_completion_time': self.healing_completed_time - self.start_time if self.healing_completed_time else None,
                'mttr_seconds': mttr
            },
            'healing_effectiveness': {
                'selector_restored': selector_restored,
                'error_injected': self.error_injection_time is not None,
                'healing_completed': self.healing_completed_time is not None
            },
            'test_configuration': {
                'stabilization_time': self.stabilization_time,
                'observation_window': self.observation_window,
                'quick_mode': self.quick_mode
            }
        }
        
        self.test_results = results
        print("Results analysis completed\n")
        return results
    
    def _display_results(self, results: Dict[str, Any]):
        """Display test results"""
        print("Phase 7: Test Results")
        print("=" * 80)
        
        # Overall result
        success = results.get('test_success', False)
        status = "SUCCESS" if success else "FAILURE"
        print(f"\nOverall Test Status: {status}")
        
        # Timing results
        timing = results.get('timing', {})
        mttr = timing.get('mttr_seconds')
        
        print(f"\nTiming Metrics:")
        print(f"  Total Test Time: {timing.get('total_test_time_seconds', 0):.1f}s")
        print(f"  Error Injection: T+{timing.get('error_injection_time', 0):.1f}s" if timing.get('error_injection_time') else "  Error Injection: Not detected")
        print(f"  Healing Completion: T+{timing.get('healing_completion_time', 0):.1f}s" if timing.get('healing_completion_time') else "  Healing Completion: Not detected")
        
        if mttr:
            print(f"\nMean Time To Recovery (MTTR): {mttr:.2f} seconds")
            if mttr < 60:
                print("  SUCCESS: MTTR meets target (< 60 seconds)")
            else:
                print("  WARNING: MTTR exceeds target (> 60 seconds)")
        else:
            print("\nWARNING: MTTR could not be calculated")
        
        # Effectiveness results
        effectiveness = results.get('healing_effectiveness', {})
        print(f"\nHealing Effectiveness:")
        print(f"  Error Injected: {'YES' if effectiveness.get('error_injected') else 'NO'}")
        print(f"  Fix Applied: {'YES' if effectiveness.get('healing_completed') else 'NO'}")
        print(f"  Selector Restored: {'YES' if effectiveness.get('selector_restored') else 'NO'}")
        
        # Research paper conclusion
        print(f"\nResearch Paper Conclusion:")
        if success and mttr and mttr < 60:
            print("  SUCCESS! The self-healing system successfully demonstrates:")
            print("    - Automated error detection")
            print("    - LLM-powered diagnosis and repair") 
            print("    - Hot-reload without system downtime")
            print("    - MTTR < 60 seconds target achieved")
        elif success:
            print("  SUCCESS: Self-healing functional, but MTTR needs optimization")
        else:
            print("  FAILURE: Self-healing system requires further development")
        
        print("\n" + "=" * 80)
    
    async def _cleanup(self):
        """Clean up test environment"""
        print("Phase 8: Cleanup")
        
        try:
            # Stop production process
            if self.production_process:
                try:
                    self.production_process.terminate()
                    self.production_process.wait(timeout=5)
                    print("  Production process stopped")
                except subprocess.TimeoutExpired:
                    self.production_process.kill()
                    self.production_process.wait(timeout=2)
                    print("  Production process force killed")
                except Exception as e:
                    print(f"  Could not stop production process: {e}")
            
            # Restore original file
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, self.opinion_agent_file)
                self.backup_file.unlink()
                print("  Original opinion agent restored")
            
            # Save test results
            if self.test_results:
                results_file = project_root / "data" / f"chaos_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump(self.test_results, f, indent=2)
                print(f"  Test results saved: {results_file.name}")
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"  Cleanup error: {e}")


async def main():
    """Main chaos test entry point"""
    parser = argparse.ArgumentParser(description="Simple Chaos Test for Self-Healing System")
    parser.add_argument("--quick", action="store_true", help="Run quick test (30s instead of 2min)")
    
    args = parser.parse_args()
    
    # Configure test
    config = {
        'stabilization_time': 10,
        'observation_window': 60,
        'quick_mode': args.quick
    }
    
    # Run chaos test
    chaos_test = SimpleChaosTest(config)
    results = await chaos_test.run_test()
    
    # Return exit code based on success
    exit_code = 0 if results.get('test_success', False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())