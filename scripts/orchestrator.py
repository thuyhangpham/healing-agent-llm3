#!/usr/bin/env python3
"""
Event-Based Orchestrator Runner

This orchestrator uses subprocess-based architecture to ensure fresh code
is loaded on every run, solving the hot-reloading issue.

Architecture:
- Spawns short-lived subprocesses for each agent run
- Monitors for error signals and triggers healing when needed
- No hot-reloading - each process loads fresh code from disk
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


class EventBasedOrchestrator:
    """Event-based orchestrator that manages agent subprocesses"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.error_signal_file = self.project_root / "data" / "production" / "error_signal.json"
        self.opinion_agent_script = self.project_root / "agents" / "opinion_search_agent.py"
        self.healing_agent_script = self.project_root / "agents" / "healing_agent.py"
        
        # Ensure error signal directory exists
        self.error_signal_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("EVENT-BASED ORCHESTRATOR INITIALIZED")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Error signal file: {self.error_signal_file}")
        print(f"Opinion agent: {self.opinion_agent_script}")
        print(f"Healing agent: {self.healing_agent_script}")
        print("="*80)
        print()
    
    def run_forever(self):
        """Main loop - runs forever, spawning subprocesses"""
        cycle_count = 0
        
        print("🚀 Starting event-based orchestrator loop...")
        print("   Press Ctrl+C to stop gracefully")
        print()
        
        try:
            while True:
                cycle_count += 1
                print(f"\n{'='*80}")
                print(f"CYCLE {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}\n")
                
                # Step A: Run opinion search agent as subprocess
                print("📊 Step A: Running Opinion Search Agent...")
                print(f"   Command: python {self.opinion_agent_script}")
                
                agent_exit_code = 0
                agent_crashed = False
                
                try:
                    result = subprocess.run(
                        [sys.executable, str(self.opinion_agent_script)],
                        cwd=str(self.project_root),
                        timeout=300,  # 5 minute timeout
                        capture_output=False,  # Show output in real-time
                        text=True
                    )
                    
                    agent_exit_code = result.returncode
                    
                    if agent_exit_code == 0:
                        print(f"\n✅ Opinion Search Agent finished successfully (exit code: 0)")
                    else:
                        print(f"\n❌ Opinion Search Agent crashed with exit code: {agent_exit_code}")
                        agent_crashed = True
                    
                except subprocess.TimeoutExpired:
                    print(f"\n⚠️  Opinion Search Agent timed out after 5 minutes")
                    agent_exit_code = -1
                    agent_crashed = True
                except Exception as e:
                    print(f"\n❌ Error running Opinion Search Agent: {e}")
                    agent_exit_code = -1
                    agent_crashed = True
                
                # Step B: Check for error signal OR crash
                print(f"\n🔍 Step B: Checking for error signal or crash...")
                error_detected = self.error_signal_file.exists()
                
                if error_detected:
                    try:
                        # Try to read error signal to verify it's valid
                        with open(self.error_signal_file, 'r', encoding='utf-8') as f:
                            error_data = json.load(f)
                        print(f"   ✅ Error signal file found and valid")
                        print(f"   Error type: {error_data.get('error_type', 'unknown')}")
                        print(f"   Error message: {error_data.get('error_message', 'unknown')[:100]}...")
                    except Exception as e:
                        print(f"   ⚠️  Error signal file exists but couldn't read: {e}")
                        error_detected = True  # Still treat as error
                elif agent_crashed:
                    # Agent crashed but didn't write error_signal.json (import error, etc.)
                    print(f"   ⚠️  Agent crashed (exit code: {agent_exit_code}) but no error signal file found")
                    print(f"   This likely indicates an import error or early crash before error handling")
                    # Create a basic error signal for healing agent
                    try:
                        error_data = {
                            'timestamp': datetime.now().isoformat(),
                            'error_type': 'ProcessCrash',
                            'error_message': f'Agent process crashed with exit code {agent_exit_code}',
                            'agent_name': 'opinion_search_agent',
                            'function_name': 'main',
                            'file_path': str(self.opinion_agent_script),
                            'severity': 'critical',
                            'exit_code': agent_exit_code
                        }
                        with open(self.error_signal_file, 'w', encoding='utf-8') as f:
                            json.dump(error_data, f, indent=2, ensure_ascii=False)
                        print(f"   📝 Created error signal file for healing agent")
                        error_detected = True
                    except Exception as e:
                        print(f"   ❌ Failed to create error signal file: {e}")
                else:
                    print(f"   ✅ No error signal file found and agent exited successfully")
                
                # Step C: Handle error or continue
                if error_detected or agent_crashed:
                    print(f"\n🚨 Step C: Error Detected! Triggering Healing Agent...")
                    print(f"   Command: python {self.healing_agent_script}")
                    
                    try:
                        healing_result = subprocess.run(
                            [sys.executable, str(self.healing_agent_script)],
                            cwd=str(self.project_root),
                            timeout=180,  # 3 minute timeout for healing
                            capture_output=False,  # Show output in real-time
                            text=True
                        )
                        
                        healing_exit_code = healing_result.returncode
                        print(f"\n✅ Healing Agent finished with exit code: {healing_exit_code}")
                        
                        # Check if error signal was removed (healing successful)
                        if not self.error_signal_file.exists():
                            print(f"   ✅ Error signal file removed - healing successful!")
                        else:
                            print(f"   ⚠️  Error signal file still exists - healing may have failed")
                            
                    except subprocess.TimeoutExpired:
                        print(f"\n⚠️  Healing Agent timed out after 3 minutes")
                    except Exception as e:
                        print(f"\n❌ Error running Healing Agent: {e}")
                else:
                    print(f"\n✅ Step C: Cycle completed successfully.")
                    print(f"   No errors detected. Sleeping for 10 seconds...")
                
                # Sleep before next cycle
                print(f"\n⏳ Sleeping for 10 seconds before next cycle...")
                time.sleep(10)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("🛑 Orchestrator stopped by user (Ctrl+C)")
            print(f"   Total cycles completed: {cycle_count}")
            print(f"{'='*80}")
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ Fatal error in orchestrator: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    orchestrator = EventBasedOrchestrator()
    orchestrator.run_forever()


if __name__ == "__main__":
    main()

