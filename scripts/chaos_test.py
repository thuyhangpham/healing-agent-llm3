#!/usr/bin/env python3
"""
Chaos Test Script for Dynamic Soup Statement Architecture

This script tests the self-healing system by:
1. Injecting a broken selector into data/production/soup_logic.txt
2. Monitoring for the Healing Agent to fix it
3. Measuring Mean Time To Recovery (MTTR)

Usage:
    python scripts/chaos_test.py

Note: Assumes scripts/orchestrator.py is already running in another terminal.
"""

import sys
import time
from pathlib import Path
from datetime import datetime


class ChaosTest:
    """
    Chaos testing for Dynamic Soup Statement architecture.
    
    Tests the healing system by sabotaging soup_logic.txt and measuring
    how long it takes for the Healing Agent to restore it.
    """
    
    def __init__(self):
        # File paths
        self.project_root = Path(__file__).parent.parent
        self.soup_logic_file = self.project_root / "data" / "production" / "soup_logic.txt"
        
        # Test state
        self.original_content: str = ""
        self.broken_content: str = "soup.select('div.chaos-broken-selector-123')"
        self.injection_time: float = 0.0
        self.recovery_time: float = 0.0
        self.test_success: bool = False
        
        print("=" * 80)
        print("CHAOS TEST: Dynamic Soup Statement Architecture")
        print("=" * 80)
        print(f"Target file: {self.soup_logic_file}")
        print(f"Project root: {self.project_root}")
        print("=" * 80)
        print()
    
    def run_test(self) -> dict:
        """
        Run the complete chaos test.
        
        Returns:
            Test results dictionary with MTTR
        """
        try:
            # Phase 1: Pre-check
            print("📋 Phase 1: Pre-check")
            if not self._pre_check():
                return {
                    'success': False,
                    'error': 'Pre-check failed',
                    'mttr_seconds': None
                }
            print("✅ Pre-check passed\n")
            
            # Phase 2: Inject chaos
            print("💥 Phase 2: Injecting Chaos")
            self._inject_chaos()
            print("✅ Chaos injected\n")
            
            # Phase 3: Monitor healing
            print("🔧 Phase 3: Monitoring Healing Process")
            self._monitor_healing()
            print("✅ Monitoring completed\n")
            
            # Phase 4: Calculate results
            print("📊 Phase 4: Calculating Results")
            results = self._calculate_results()
            
            # Phase 5: Cleanup (if needed)
            if not self.test_success:
                print("🧹 Phase 5: Cleanup (Test Failed)")
                self._cleanup()
                print("✅ Cleanup completed\n")
            
            # Display results
            self._display_results(results)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Chaos test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency cleanup
            self._cleanup()
            
            return {
                'success': False,
                'error': str(e),
                'mttr_seconds': None
            }
    
    def _pre_check(self) -> bool:
        """Pre-check: Ensure target file exists and read current content."""
        try:
            # Check if file exists
            if not self.soup_logic_file.exists():
                print(f"❌ Target file does not exist: {self.soup_logic_file}")
                print(f"   Please ensure the file exists before running chaos test.")
                return False
            
            # Read current content
            with open(self.soup_logic_file, 'r', encoding='utf-8') as f:
                self.original_content = f.read().strip()
            
            if not self.original_content:
                print(f"⚠️  Target file is empty. This may cause issues.")
                return False
            
            print(f"✅ Target file exists")
            print(f"   File size: {len(self.original_content)} characters")
            print(f"   Current content preview: {self.original_content[:80]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Pre-check failed: {e}")
            return False
    
    def _inject_chaos(self):
        """Inject broken selector into soup_logic.txt"""
        try:
            # Record injection time
            self.injection_time = time.time()
            
            # Write broken selector
            with open(self.soup_logic_file, 'w', encoding='utf-8') as f:
                f.write(self.broken_content)
            
            print(f"✅ Injected broken selector: {self.broken_content}")
            print(f"   Injection time: {datetime.fromtimestamp(self.injection_time).strftime('%H:%M:%S')}")
            
            # Verify injection
            with open(self.soup_logic_file, 'r', encoding='utf-8') as f:
                current_content = f.read().strip()
            
            if current_content == self.broken_content:
                print(f"   ✅ Verification: Broken selector confirmed in file")
            else:
                raise Exception(f"Injection verification failed. Expected '{self.broken_content}', got '{current_content}'")
            
        except Exception as e:
            raise Exception(f"Failed to inject chaos: {e}")
    
    def _monitor_healing(self):
        """
        Monitor healing process by checking file content every 2 seconds.
        Maximum monitoring time: 60 seconds.
        """
        max_monitoring_time = 60  # seconds
        check_interval = 2  # seconds
        max_checks = max_monitoring_time // check_interval
        
        print(f"   Monitoring for up to {max_monitoring_time} seconds...")
        print(f"   Checking every {check_interval} seconds")
        print(f"   Waiting for Healing Agent to fix the selector...")
        print()
        
        start_monitor = time.time()
        check_count = 0
        
        while check_count < max_checks:
            check_count += 1
            elapsed = time.time() - start_monitor
            
            try:
                # Read current file content
                with open(self.soup_logic_file, 'r', encoding='utf-8') as f:
                    current_content = f.read().strip()
                
                # Check if content has changed (healing occurred)
                if current_content != self.broken_content:
                    self.recovery_time = time.time()
                    self.test_success = True
                    mttr = self.recovery_time - self.injection_time
                    
                    print(f"   ✅ HEALING DETECTED at {check_count * check_interval}s!")
                    print(f"   ✅ File content changed - Healing Agent fixed the selector")
                    print(f"   ✅ Recovery time: {datetime.fromtimestamp(self.recovery_time).strftime('%H:%M:%S')}")
                    print(f"   ✅ MTTR: {mttr:.2f} seconds")
                    
                    # Show what the fixed content looks like
                    print(f"\n   Fixed content preview: {current_content[:80]}...")
                    
                    return
                
                # Still broken - continue monitoring
                print(f"   [{check_count}/{max_checks}] Check at {elapsed:.1f}s: Still broken...", end='\r')
                
            except Exception as e:
                print(f"\n   ⚠️  Error reading file during check {check_count}: {e}")
            
            # Wait before next check
            time.sleep(check_interval)
        
        # Timeout - healing did not occur
        print(f"\n   ❌ TIMEOUT: Healing did not occur within {max_monitoring_time} seconds")
        self.test_success = False
    
    def _calculate_results(self) -> dict:
        """Calculate test results and MTTR"""
        results = {
            'success': self.test_success,
            'test_timestamp': datetime.now().isoformat(),
            'mttr_seconds': None,
            'injection_time': datetime.fromtimestamp(self.injection_time).isoformat() if self.injection_time else None,
            'recovery_time': datetime.fromtimestamp(self.recovery_time).isoformat() if self.recovery_time else None,
        }
        
        if self.test_success and self.recovery_time:
            mttr = self.recovery_time - self.injection_time
            results['mttr_seconds'] = mttr
            results['target_met'] = mttr < 60  # Target: MTTR < 60 seconds
        
        return results
    
    def _cleanup(self):
        """Restore original content if test failed"""
        try:
            if self.original_content:
                with open(self.soup_logic_file, 'w', encoding='utf-8') as f:
                    f.write(self.original_content)
                
                print(f"   ✅ Restored original content to {self.soup_logic_file.name}")
                
                # Verify restoration
                with open(self.soup_logic_file, 'r', encoding='utf-8') as f:
                    restored_content = f.read().strip()
                
                if restored_content == self.original_content:
                    print(f"   ✅ Verification: Original content restored successfully")
                else:
                    print(f"   ⚠️  Warning: Content may not match original exactly")
            else:
                print(f"   ⚠️  No original content to restore")
                
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
            print(f"   ⚠️  MANUAL CLEANUP REQUIRED: Please restore {self.soup_logic_file} manually")
    
    def _display_results(self, results: dict):
        """Display test results"""
        print("\n" + "=" * 80)
        print("CHAOS TEST RESULTS")
        print("=" * 80)
        
        # Overall status
        success = results.get('success', False)
        status = "✅ SUCCESS" if success else "❌ FAILURE"
        print(f"\nTest Status: {status}")
        
        # MTTR
        mttr = results.get('mttr_seconds')
        if mttr is not None:
            print(f"\n⏱️  Mean Time To Recovery (MTTR): {mttr:.2f} seconds")
            
            target_met = results.get('target_met', False)
            if target_met:
                print(f"   ✅ MTTR meets target (< 60 seconds)")
            else:
                print(f"   ⚠️  MTTR exceeds target (>= 60 seconds)")
        else:
            print(f"\n⚠️  MTTR: Could not be calculated (healing did not occur)")
        
        # Timing details
        print(f"\n📅 Timing Details:")
        if results.get('injection_time'):
            print(f"   Error Injection: {results['injection_time']}")
        if results.get('recovery_time'):
            print(f"   Recovery Detected: {results['recovery_time']}")
        
        # Conclusion
        print(f"\n📝 Conclusion:")
        if success and mttr and mttr < 60:
            print("   🎉 Self-healing system successfully demonstrated:")
            print("      • Automated error detection")
            print("      • LLM-powered code repair")
            print("      • Dynamic selector restoration")
            print("      • MTTR < 60 seconds target achieved")
        elif success:
            print("   ✅ Self-healing functional, but MTTR needs optimization")
        else:
            print("   ❌ Self-healing system requires further development")
            print("   ⚠️  Original content has been restored")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point"""
    print("\n🚀 Starting Chaos Test...")
    print("   Note: Ensure scripts/orchestrator.py is running in another terminal\n")
    
    # Run chaos test
    chaos_test = ChaosTest()
    results = chaos_test.run_test()
    
    # Exit with appropriate code
    exit_code = 0 if results.get('success', False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
