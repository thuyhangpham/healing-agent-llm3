#!/usr/bin/env python3
"""
Fixed version of run_system.py with all syntax errors resolved
"""

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.healing_agent import HealingAgent
from agents.law_search_agent import AutonomousLawSearchAgent
from agents.opinion_search_agent import AutonomousOpinionSearchAgent
from core.chaos_engineering import ChaosEngine
from utils.logger import get_logger
from utils.config import load_config


class ExperimentRunner:
    """Orchestrates controlled chaos experiments and evidence generation"""
    
    def __init__(self):
        self.logger = get_logger("experiment_runner")
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.data_dir = project_root / "data"
        self.experiments_dir = project_root / "data" / "experiments"
        
        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.healing_agent = None
        self.law_agent = None
        self.opinion_agent = None
        self.chaos_engine = None
        self.current_experiment = {}
        self.experiment_results = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_file = self.config_dir / "agents.yaml"
        if config_file.exists():
            return load_config(str(config_file))
        return {}
    
    async def initialize_agents(self):
        """Initialize all agents for experiments"""
        try:
            # Initialize healing agent
            healing_config = self.config.get('healing_agent', {})
            self.healing_agent = HealingAgent("experiment_healing", healing_config)
            await self.healing_agent.initialize()
            self.logger.info("Healing agent initialized")
            
            # Initialize law search agent
            law_config = self.config.get('law_search_agent', {})
            self.law_agent = AutonomousLawSearchAgent(law_config)
            self.logger.info("Law search agent initialized")
            
            # Initialize opinion search agent
            opinion_config = self.config.get('opinion_search_agent', {})
            self.opinion_agent = AutonomousOpinionSearchAgent(opinion_config)
            self.logger.info("Opinion search agent initialized")
            
            # Initialize chaos engine
            chaos_config = self.config.get('chaos_engineering', {})
            self.chaos_engine = ChaosEngine(chaos_config)
            self.logger.info("Chaos engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def run_experiment(self, experiment_type: str, target_agent: str, 
                         experiment_id: str = None, start_time: str = None) -> Dict[str, Any]:
        """Run a single chaos experiment"""
        # Generate ID and start time if not provided
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}"
        if start_time is None:
            start_time = datetime.now().isoformat()
        
        self.current_experiment = {
            'id': experiment_id,
            'type': experiment_type,
            'target_agent': target_agent,
            'start_time': start_time,
            'status': 'running'
        }
        
        self.logger.info(f"Starting experiment {experiment_id}: {experiment_type} on {target_agent}")
        
        try:
            # Select target agent
            if target_agent == 'healing_agent':
                agent = self.healing_agent
            elif target_agent == 'law_search_agent':
                agent = self.law_agent
            elif target_agent == 'opinion_search_agent':
                agent = self.opinion_agent
            else:
                raise ValueError(f"Unknown target agent: {target_agent}")
            
            # Apply chaos based on experiment type
            if experiment_type == 'css_selector_failure':
                chaos_result = await self.chaos_engine.inject_css_selector_failure(agent)
            elif experiment_type == 'network_timeout':
                chaos_result = await self.chaos_engine.inject_network_timeout(agent)
            elif experiment_type == 'encoding_error':
                chaos_result = await self.chaos_engine.inject_encoding_error(agent)
            elif experiment_type == 'javascript_heavy':
                chaos_result = await self.chaos_engine.inject_javascript_heavy_page(agent)
            elif experiment_type == 'rate_limiting':
                chaos_result = await self.chaos_engine.inject_rate_limiting(agent)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
            
            # Monitor healing response
            healing_start = time.time()
            healing_result = None
            
            if hasattr(agent, 'handle_error') and chaos_result.get('error_injected'):
                healing_result = await agent.handle_error(chaos_result['error_context'])
            
            healing_time = time.time() - healing_start
            
            # Complete experiment record
            end_time = datetime.now().isoformat()
            self.current_experiment.update({
                'end_time': end_time,
                'status': 'completed',
                'chaos_result': chaos_result,
                'healing_result': healing_result,
                'healing_time': healing_time
            })
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'chaos_injection': chaos_result,
                'healing_response': healing_result,
                'healing_time': healing_time
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.current_experiment.update({
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            
            return {
                'success': False,
                'experiment_id': experiment_id,
                'error': str(e)
            }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.experiment_results:
            return {}
        
        # Calculate healing agent performance
        healing_experiments = [exp for exp in self.experiment_results if exp.get('target_agent') == 'healing_agent']
        
        if healing_experiments:
            successful_healing = sum(1 for exp in healing_experiments if exp.get('result', {}).get('success', False))
            total_healing = len(healing_experiments)
            healing_times = [exp.get('result', {}).get('healing_time', 0) for exp in healing_experiments if exp.get('result', {}).get('success', False)]
            avg_healing_time = sum(healing_times) / len(healing_times) if healing_times else 0
            
            healing_status = {
                'total_experiments': total_healing,
                'successful_experiments': successful_healing,
                'success_rate': (successful_healing / total_healing * 100) if total_healing > 0 else 0,
                'average_mttr': avg_healing_time
            }
        else:
            healing_status = {
                'total_experiments': 0,
                'successful_experiments': 0,
                'success_rate': 0,
                'average_mttr': 0
            }
        
        # Get chaos engine statistics
        chaos_stats = await self.chaos_engine.get_statistics() if self.chaos_engine else {}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiment_results),
            'healing_agent_performance': healing_status,
            'chaos_engine_statistics': chaos_stats,
            'system_performance': {
                'total_experiments': len(self.experiment_results) + 1,
                'success_rate': healing_status.get('success_rate', 0),
                'average_mttr': healing_status.get('average_mttr', 0),
                'target_achievement': {
                    'mttr_target_met': healing_status.get('average_mttr', 0) < 60,
                    'success_rate_target_met': healing_status.get('success_rate', 0) > 80
                }
            }
        }
    
    async def save_report(self, report_data: Dict[str, Any]) -> str:
        """Save performance report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.experiments_dir / f"performance_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {report_file}")
        return str(report_file)


async def main():
    """Main demo generator function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo Generator for Self-Healing System")
    parser.add_argument("--experiment", type=str, help="Run specific experiment (css_selector_failure, network_timeout, encoding_error, javascript_heavy, rate_limiting)")
    parser.add_argument("--target", type=str, help="Target agent for experiment")
    parser.add_argument("--demo", action="store_true", help="Run complete demo sequence")
    
    args = parser.parse_args()
    
    # Create demo generator
    demo = ExperimentRunner()
    
    try:
        # Initialize agents
        await demo.initialize_agents()
        
        if args.demo:
            # Run complete demo sequence
            experiments = [
                ('css_selector_failure', 'healing_agent'),
                ('network_timeout', 'law_search_agent'),
                ('encoding_error', 'opinion_search_agent'),
                ('javascript_heavy', 'healing_agent'),
                ('rate_limiting', 'law_search_agent')
            ]
            
            # Run experiments
            for i, (exp_type, target_agent) in enumerate(experiments, 1):
                # Generate experiment ID and start time before running
                experiment_id = f"exp_{int(time.time())}"
                start_time = datetime.now().isoformat()
                
                print(f"\n{'='*60}")
                print(f"EXPERIMENT {i}/{len(experiments)}")
                print(f"EXPERIMENT: {exp_type.upper()}")
                print(f"ID: {experiment_id}")
                print(f"Target: {target_agent}")
                print(f"Start Time: {start_time}")
                print(f"{'='*60}")
                
                result = await demo.run_experiment(exp_type, target_agent, experiment_id, start_time)
                demo.experiment_results.append(demo.current_experiment)
                
                # Small delay between experiments
                await asyncio.sleep(2)
                
                print(f"Success: {result.get('success', False)}")
                
                if result.get('success'):
                    print(f"Healing completed successfully!")
                else:
                    print(f"Healing failed")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                
                print(f"{'='*60}")
            
            # Generate final report
            print(f"\n{'='*80}")
            print("GENERATING PERFORMANCE REPORT")
            print(f"{'='*80}")
            
            performance = await demo.generate_performance_report()
            report_file = await demo.save_report(performance)
            
            print(f"Performance report saved to: {report_file}")
            
            # Display summary
            healing_effectiveness = performance.get('healing_agent_performance', {})
            print(f"AVG HEALING TIME: {healing_effectiveness.get('average_mttr', 0):.1f}s")
            
            system_perf = performance.get('system_performance', {})
            print(f"SUCCESS RATE: {system_perf.get('success_rate', 0):.1f}%")
            
            targets = system_perf.get('target_achievement', {})
            print(f"MTTR TARGET MET: {'✅ YES' if targets.get('mttr_target_met', False) else '❌ NO'}")
            print(f"SUCCESS RATE TARGET MET: {'✅ YES' if targets.get('success_rate_target_met', False) else '❌ NO'}")
            
            print(f"\n{'='*80}")
            print("DEMO COMPLETION CHECKLIST")
            print(f"{'='*80}")
            print(f"Check: data/experiments/")
            print(f"Check: data/experiments/final_output.json")
            print(f"Check: data/experiments/performance_report_*.json")
            print(f"{'='*80}")
            
            return 0
        elif args.experiment and args.target:
            # Run single experiment
            result = await demo.run_experiment(args.experiment, args.target)
            demo.experiment_results.append(demo.current_experiment)
            
            print(f"Experiment completed: {result.get('success', False)}")
            return 0
        else:
            print("Please specify either --demo with experiment types, or --experiment and --target")
            return 1
            
    except Exception as e:
        print(f"\nDemo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))