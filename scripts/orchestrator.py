#!/usr/bin/env python3
"""
Event-Based Orchestrator Runner with Finite State Machine (FSM)

This orchestrator uses an async FSM pattern to coordinate the Worker Agent
(OpinionSearchAgent) and Doctor Agent (HealingAgent).

Architecture:
- Finite State Machine (FSM) pattern for state management
- Worker Agent: Scrapes data using dynamic selectors, creates error_signal.json on failure
- Healing Agent: Monitors error file, uses LLM to fix selector, updates soup_logic.txt
- Orchestrator: Manages state transitions between IDLE, SCRAPING, ERROR_DETECTED, HEALING, ROLLBACK
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to sys.path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.metrics_monitor import MetricsMonitor
from utils.constants import *
from agents.opinion_search_agent import AutonomousOpinionSearchAgent
from agents.healing_agent import HealingAgent
from utils.config import load_config


class EventBasedOrchestrator:
    """
    Event-based orchestrator that manages agents using FSM pattern.
    
    FSM States:
    - IDLE: Waiting state, transitions to SCRAPING
    - SCRAPING: Running worker agent, transitions based on result
    - ERROR_DETECTED: Error found, transitions to HEALING
    - HEALING: Running healing agent, transitions based on result
    - ROLLBACK: Rolling back failed healing, transitions to IDLE
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.error_signal_file = self.project_root / "data" / "production" / "error_signal.json"
        
        # Ensure error signal directory exists
        self.error_signal_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics monitor
        self.monitor = MetricsMonitor()
        
        # FSM State
        self.state = STATE_IDLE
        self.retries = 0
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize agents (will be initialized in async context)
        self.worker = None  # AutonomousOpinionSearchAgent
        self.doctor = None  # HealingAgent
        
        print("="*80)
        print("EVENT-BASED ORCHESTRATOR INITIALIZED (FSM Pattern)")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Error signal file: {self.error_signal_file}")
        print(f"Initial state: {self.state}")
        print("="*80)
        print()
    
    def _load_config(self) -> dict:
        """Load agent configuration from config file"""
        try:
            config_file = self.project_root / "config" / "agents.yaml"
            if config_file.exists():
                full_config = load_config(str(config_file))
                return full_config
            else:
                print(f"⚠️  Config file not found: {config_file}, using defaults")
                return {}
        except Exception as e:
            print(f"⚠️  Could not load config: {e}, using defaults")
            return {}
    
    async def initialize(self):
        """Initialize agents asynchronously"""
        try:
            print("🔧 Initializing agents...")
            
            # Initialize Worker Agent (OpinionSearchAgent)
            worker_config = self.config.get('agents', {}).get('opinion_search_agent', {})
            self.worker = AutonomousOpinionSearchAgent(worker_config)
            await self.worker.initialize()
            print("   ✅ Worker Agent (OpinionSearchAgent) initialized")
            
            # Initialize Doctor Agent (HealingAgent)
            doctor_config = self.config.get('agents', {}).get('healing_agent', {})
            self.doctor = HealingAgent("orchestrator_healing", doctor_config)
            await self.doctor.initialize()
            print("   ✅ Doctor Agent (HealingAgent) initialized")
            
            print("✅ All agents initialized successfully\n")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run(self):
        """
        Main FSM loop - runs forever, controlled by state transitions.
        """
        # Safety check: ensure agents are initialized
        if self.worker is None or self.doctor is None:
            print("❌ ERROR: Agents not initialized!")
            print("   Worker initialized:", self.worker is not None)
            print("   Doctor initialized:", self.doctor is not None)
            print("   Please call initialize() before run()")
            raise RuntimeError("Agents must be initialized before running FSM loop")
        
        cycle_count = 0
        
        print("🚀 Starting FSM-based orchestrator loop...")
        print("   Press Ctrl+C to stop gracefully")
        print()
        
        try:
            while True:
                cycle_count += 1
                
                print(f"\n{'='*80}")
                print(f"CYCLE {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"[FSM] Current State: {self.state}")
                print(f"{'='*80}\n")
                
                # FSM State Machine
                if self.state == STATE_IDLE:
                    await self._handle_idle_state()
                
                elif self.state == STATE_SCRAPING:
                    await self._handle_scraping_state()
                
                elif self.state == STATE_ERROR_DETECTED:
                    await self._handle_error_detected_state()
                
                elif self.state == STATE_HEALING:
                    await self._handle_healing_state()
                
                elif self.state == STATE_ROLLBACK:
                    await self._handle_rollback_state()
                
                else:
                    print(f"⚠️  Unknown state: {self.state}, resetting to IDLE")
                    self.state = STATE_IDLE
                
                # Prevent CPU spiking
                await asyncio.sleep(LOOP_SLEEP)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("🛑 Orchestrator stopped by user (Ctrl+C)")
            print(f"   Total cycles completed: {cycle_count}")
            print(f"   Final state: {self.state}")
            print(f"{'='*80}")
            
            # Cleanup
            await self.shutdown()
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ Fatal error in orchestrator: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup
            await self.shutdown()
            sys.exit(1)
    
    async def _handle_idle_state(self):
        """Handle IDLE state: Wait, then transition to SCRAPING"""
        print(f"[FSM] ⏸️  {STATE_IDLE} - Waiting...")
        await asyncio.sleep(IDLE_SLEEP)
        print(f"[FSM]   → Transitioning to {STATE_SCRAPING}")
        self.state = STATE_SCRAPING
    
    async def _handle_scraping_state(self):
        """Handle SCRAPING state: Run worker agent"""
        print(f"[FSM] 🕵️  {STATE_SCRAPING} - Running Worker Agent...")
        
        # Safety check: ensure worker is initialized
        if self.worker is None:
            print(f"[FSM]   ❌ Worker Agent not initialized!")
            print(f"[FSM]   → Transitioning to {STATE_IDLE}")
            self.state = STATE_IDLE
            return
        
        try:
            # Call worker.search()
            result = await self.worker.search()
            
            # Check if search was successful
            if result and result.get('status') in ['success', 'limit_reached', 'no_keywords']:
                print(f"[FSM]   ✅ Success - Logging cycle")
                
                # Log successful cycle
                articles_count = result.get('new_articles_count', 0)
                self.monitor.log_cycle('success', articles_count)
                
                print(f"[FSM]   → Transitioning to {STATE_IDLE}")
                self.state = STATE_IDLE
                self.retries = 0  # Reset retries on success
                
            else:
                # Search returned error status
                print(f"[FSM]   ⚠️  Search returned status: {result.get('status', 'unknown')}")
                raise Exception(f"Search failed with status: {result.get('status', 'unknown')}")
                
        except Exception as e:
            print(f"[FSM]   ❌ Exception during scraping: {e}")
            
            # Check if error_signal.json exists
            if self.error_signal_file.exists():
                print(f"[FSM]   ✅ Error Signal Found")
                print(f"[FSM]   → Transitioning to {STATE_ERROR_DETECTED}")
                self.state = STATE_ERROR_DETECTED
            else:
                print(f"[FSM]   ⚠️  Network/Unknown Error (no error signal)")
                print(f"[FSM]   → Transitioning to {STATE_IDLE}")
                self.state = STATE_IDLE
    
    async def _handle_error_detected_state(self):
        """Handle ERROR_DETECTED state: Activate healing protocol"""
        print(f"[FSM] 🚨 {STATE_ERROR_DETECTED} - Activating Healing Protocol")
        print(f"[FSM]   → Transitioning to {STATE_HEALING}")
        self.state = STATE_HEALING
    
    async def _handle_healing_state(self):
        """Handle HEALING state: Run healing agent"""
        print(f"[FSM] 🔧 {STATE_HEALING} - Running Doctor Agent...")
        
        # Safety check: ensure doctor is initialized
        if self.doctor is None:
            print(f"[FSM]   ❌ Doctor Agent not initialized!")
            print(f"[FSM]   → Transitioning to {STATE_ROLLBACK}")
            self.state = STATE_ROLLBACK
            return
        
        # Safety check: ensure error signal file exists
        if not self.error_signal_file.exists():
            print(f"[FSM]   ⚠️  Error signal file not found, skipping healing")
            print(f"[FSM]   → Transitioning to {STATE_IDLE}")
            self.state = STATE_IDLE
            return
        
        try:
            # Call doctor.process_error_signal()
            success = await self.doctor.process_error_signal(self.error_signal_file)
            
            if success:
                print(f"[FSM]   ✅ Fix Applied")
                
                # Log successful healing
                # Note: Duration is tracked inside process_error_signal
                
                print(f"[FSM]   → Transitioning to {STATE_SCRAPING} (Immediate retry to verify fix)")
                self.state = STATE_SCRAPING
                self.retries = 0  # Reset retries on successful healing
                
            else:
                print(f"[FSM]   ❌ Healing Failed")
                print(f"[FSM]   → Transitioning to {STATE_ROLLBACK}")
                self.state = STATE_ROLLBACK
                
        except Exception as e:
            print(f"[FSM]   ❌ Exception during healing: {e}")
            print(f"[FSM]   → Transitioning to {STATE_ROLLBACK}")
            self.state = STATE_ROLLBACK
    
    async def _handle_rollback_state(self):
        """Handle ROLLBACK state: Rollback configuration"""
        print(f"[FSM] ⏪ {STATE_ROLLBACK} - Rolling back configuration...")
        
        # Optional: Call rollback logic if needed
        # The healing agent should handle rollback internally, but we track the state
        
        print(f"[FSM]   → Transitioning to {STATE_IDLE} (Cool down)")
        self.state = STATE_IDLE
    
    async def shutdown(self):
        """Shutdown orchestrator and agents"""
        print("\n🛑 Shutting down orchestrator...")
        
        try:
            if self.worker:
                await self.worker.shutdown()
                print("   ✅ Worker Agent shut down")
        except Exception as e:
            print(f"   ⚠️  Error shutting down worker: {e}")
        
        try:
            if self.doctor:
                await self.doctor.shutdown()
                print("   ✅ Doctor Agent shut down")
        except Exception as e:
            print(f"   ⚠️  Error shutting down doctor: {e}")
        
        print("✅ Orchestrator shutdown complete")
    
    def run_forever(self):
        """
        Legacy method name - calls async run() for backward compatibility.
        """
        asyncio.run(self._run_with_init())
    
    async def _run_with_init(self):
        """Initialize agents and run the FSM loop"""
        # Initialize agents
        initialized = await self.initialize()
        if not initialized:
            print("❌ Failed to initialize agents, exiting")
            sys.exit(1)
        
        # Run FSM loop
        await self.run()


async def main():
    """Main entry point"""
    orchestrator = EventBasedOrchestrator()
    await orchestrator._run_with_init()


if __name__ == "__main__":
    asyncio.run(main())
