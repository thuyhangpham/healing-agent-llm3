"""
Healing Metrics System

This module provides comprehensive metrics collection and analysis for the
self-healing system, tracking MTTR, success rates, and providing
data for research and continuous improvement.

Features:
- MTTR (Mean Time To Repair) calculation and tracking
- Success rate analysis and trend detection
- Research data export capabilities (CSV/JSON)
- Chaos engineering testing framework
- Performance benchmarking tools
- Real-time metrics dashboard data
"""

import asyncio
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

from utils.logger import get_logger


class MetricType(Enum):
    """Types of metrics collected"""
    HEALING_EVENT = "healing_event"
    ERROR_DETECTED = "error_detected"
    FIX_GENERATED = "fix_generated"
    FIX_APPLIED = "fix_applied"
    FIX_VALIDATED = "fix_validated"
    ROLLBACK_PERFORMED = "rollback_performed"
    MTTR_CALCULATION = "mttr_calculation"
    SUCCESS_RATE = "success_rate"
    CHAOS_TEST = "chaos_test"


@dataclass
class HealingEvent:
    """Individual healing event record"""
    event_id: str
    timestamp: datetime
    error_id: str
    error_type: str
    agent_name: str
    category: str
    severity: str
    success: bool
    time_to_repair: float
    fix_applied: bool
    validation_passed: bool
    rollback_performed: bool
    backup_created: bool
    llm_response_time: float
    patch_validation_time: float
    hot_reload_time: float
    error_context: Dict[str, Any]
    fix_description: str


@dataclass
class MTTRCalculation:
    """MTTR calculation result"""
    timestamp: datetime
    period_hours: int
    mttr_seconds: float
    total_events: int
    successful_events: int
    failed_events: int
    median_ttr: float
    p95_ttr: float
    p99_ttr: float
    min_ttr: float
    max_ttr: float


@dataclass
class SuccessRateCalculation:
    """Success rate calculation result"""
    timestamp: datetime
    period_hours: int
    success_rate: float
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    rolled_back_attempts: int
    by_category: Dict[str, float]
    by_agent: Dict[str, float]
    by_severity: Dict[str, float]


@dataclass
class ChaosTestResult:
    """Chaos engineering test result"""
    test_id: str
    timestamp: datetime
    test_type: str
    target_agent: str
    error_injected: str
    healing_triggered: bool
    healing_successful: bool
    time_to_heal: float
    system_impact: Dict[str, Any]
    test_passed: bool
    notes: str


class HealingMetrics:
    """
    Healing metrics collection and analysis system
    
    This class provides comprehensive metrics tracking for the self-healing
    system, including MTTR calculations, success rates, and research data
    export capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize healing metrics"""
        self.config = config or {}
        self.logger = get_logger("healing_metrics")
        
        # Data storage
        self.healing_events: List[HealingEvent] = []
        self.mttr_history: List[MTTRCalculation] = []
        self.success_rate_history: List[SuccessRateCalculation] = []
        self.chaos_test_results: List[ChaosTestResult] = []
        
        # Real-time metrics cache
        self.recent_events: deque = deque(maxlen=1000)
        self.category_stats: defaultdict = defaultdict(lambda: defaultdict(int))
        self.agent_stats: defaultdict = defaultdict(lambda: defaultdict(int))
        self.severity_stats: defaultdict = defaultdict(lambda: defaultdict(int))
        
        # Configuration
        self.metrics_dir = Path(self.config.get('metrics_dir', 'data/metrics'))
        self.export_dir = Path(self.config.get('export_dir', 'data/exports'))
        self.mttr_periods = self.config.get('mttr_periods', [1, 6, 24, 168])  # hours
        self.success_rate_periods = self.config.get('success_rate_periods', [1, 6, 24, 168])
        self.max_events_memory = self.config.get('max_events_memory', 10000)
        self.auto_export_interval = self.config.get('auto_export_interval', 3600)  # seconds
        
        # Ensure directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance targets
        self.mttr_target = self.config.get('mttr_target_seconds', 60)
        self.success_rate_target = self.config.get('success_rate_target', 0.8)
        
        self.logger.info("HealingMetrics initialized")
    
    async def initialize(self) -> bool:
        """Initialize healing metrics"""
        try:
            # Load existing data
            await self._load_historical_data()
            
            # Start background tasks
            asyncio.create_task(self._metrics_calculation_loop())
            asyncio.create_task(self._auto_export_loop())
            
            self.logger.info("HealingMetrics initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HealingMetrics: {e}")
            return False
    
    async def record_healing_event(
        self,
        error_id: str,
        success: bool,
        time_to_repair: float,
        error_type: str,
        agent_name: str,
        **kwargs
    ):
        """Record a healing event"""
        try:
            event_id = f"heal_{int(time.time())}_{hash(error_id) % 10000:04d}"
            
            event = HealingEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                error_id=error_id,
                error_type=error_type,
                agent_name=agent_name,
                category=kwargs.get('category', 'unknown'),
                severity=kwargs.get('severity', 'medium'),
                success=success,
                time_to_repair=time_to_repair,
                fix_applied=kwargs.get('fix_applied', False),
                validation_passed=kwargs.get('validation_passed', False),
                rollback_performed=kwargs.get('rollback_performed', False),
                backup_created=kwargs.get('backup_created', False),
                llm_response_time=kwargs.get('llm_response_time', 0.0),
                patch_validation_time=kwargs.get('patch_validation_time', 0.0),
                hot_reload_time=kwargs.get('hot_reload_time', 0.0),
                error_context=kwargs.get('error_context', {}),
                fix_description=kwargs.get('fix_description', '')
            )
            
            # Store event
            self.healing_events.append(event)
            self.recent_events.append(event)
            
            # Update statistics
            self._update_statistics(event)
            
            # Limit memory usage
            if len(self.healing_events) > self.max_events_memory:
                self.healing_events = self.healing_events[-self.max_events_memory:]
            
            self.logger.debug(f"Recorded healing event {event_id}: success={success}, ttr={time_to_repair:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to record healing event: {e}")
    
    async def record_chaos_test(
        self,
        test_type: str,
        target_agent: str,
        error_injected: str,
        healing_triggered: bool,
        healing_successful: bool,
        time_to_heal: float,
        system_impact: Dict[str, Any],
        test_passed: bool,
        notes: str = ""
    ):
        """Record a chaos engineering test result"""
        try:
            test_id = f"chaos_{int(time.time())}_{hash(target_agent) % 10000:04d}"
            
            test_result = ChaosTestResult(
                test_id=test_id,
                timestamp=datetime.now(),
                test_type=test_type,
                target_agent=target_agent,
                error_injected=error_injected,
                healing_triggered=healing_triggered,
                healing_successful=healing_successful,
                time_to_heal=time_to_heal,
                system_impact=system_impact,
                test_passed=test_passed,
                notes=notes
            )
            
            self.chaos_test_results.append(test_result)
            
            self.logger.info(f"Recorded chaos test {test_id}: {test_type} on {target_agent}")
            
        except Exception as e:
            self.logger.error(f"Failed to record chaos test: {e}")
    
    def _update_statistics(self, event: HealingEvent):
        """Update real-time statistics"""
        # Category statistics
        self.category_stats[event.category]['total'] += 1
        if event.success:
            self.category_stats[event.category]['successful'] += 1
        else:
            self.category_stats[event.category]['failed'] += 1
        
        # Agent statistics
        self.agent_stats[event.agent_name]['total'] += 1
        if event.success:
            self.agent_stats[event.agent_name]['successful'] += 1
        else:
            self.agent_stats[event.agent_name]['failed'] += 1
        
        # Severity statistics
        self.severity_stats[event.severity]['total'] += 1
        if event.success:
            self.severity_stats[event.severity]['successful'] += 1
        else:
            self.severity_stats[event.severity]['failed'] += 1
    
    async def calculate_mttr(self, period_hours: int = 24) -> MTTRCalculation:
        """Calculate MTTR for a given period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            
            # Filter events in period
            period_events = [
                event for event in self.healing_events
                if event.timestamp >= cutoff_time and event.success
            ]
            
            if not period_events:
                return MTTRCalculation(
                    timestamp=datetime.now(),
                    period_hours=period_hours,
                    mttr_seconds=0.0,
                    total_events=0,
                    successful_events=0,
                    failed_events=0,
                    median_ttr=0.0,
                    p95_ttr=0.0,
                    p99_ttr=0.0,
                    min_ttr=0.0,
                    max_ttr=0.0
                )
            
            # Calculate metrics
            ttr_values = [event.time_to_repair for event in period_events]
            
            mttr = statistics.mean(ttr_values)
            median_ttr = statistics.median(ttr_values)
            p95_ttr = self._percentile(ttr_values, 95)
            p99_ttr = self._percentile(ttr_values, 99)
            min_ttr = min(ttr_values)
            max_ttr = max(ttr_values)
            
            # Count failed events in period
            failed_events = len([
                event for event in self.healing_events
                if event.timestamp >= cutoff_time and not event.success
            ])
            
            calculation = MTTRCalculation(
                timestamp=datetime.now(),
                period_hours=period_hours,
                mttr_seconds=mttr,
                total_events=len(period_events) + failed_events,
                successful_events=len(period_events),
                failed_events=failed_events,
                median_ttr=median_ttr,
                p95_ttr=p95_ttr,
                p99_ttr=p99_ttr,
                min_ttr=min_ttr,
                max_ttr=max_ttr
            )
            
            self.mttr_history.append(calculation)
            
            # Keep only recent calculations
            if len(self.mttr_history) > 1000:
                self.mttr_history = self.mttr_history[-1000:]
            
            return calculation
            
        except Exception as e:
            self.logger.error(f"Failed to calculate MTTR: {e}")
            return None
    
    async def calculate_success_rate(self, period_hours: int = 24) -> SuccessRateCalculation:
        """Calculate success rate for a given period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            
            # Filter events in period
            period_events = [
                event for event in self.healing_events
                if event.timestamp >= cutoff_time
            ]
            
            if not period_events:
                return SuccessRateCalculation(
                    timestamp=datetime.now(),
                    period_hours=period_hours,
                    success_rate=0.0,
                    total_attempts=0,
                    successful_attempts=0,
                    failed_attempts=0,
                    rolled_back_attempts=0,
                    by_category={},
                    by_agent={},
                    by_severity={}
                )
            
            # Calculate overall success rate
            successful = sum(1 for event in period_events if event.success)
            failed = sum(1 for event in period_events if not event.success)
            rolled_back = sum(1 for event in period_events if event.rollback_performed)
            
            success_rate = successful / len(period_events)
            
            # Calculate by category
            by_category = {}
            for category in set(event.category for event in period_events):
                category_events = [e for e in period_events if e.category == category]
                category_success = sum(1 for e in category_events if e.success)
                by_category[category] = category_success / len(category_events)
            
            # Calculate by agent
            by_agent = {}
            for agent in set(event.agent_name for event in period_events):
                agent_events = [e for e in period_events if e.agent_name == agent]
                agent_success = sum(1 for e in agent_events if e.success)
                by_agent[agent] = agent_success / len(agent_events)
            
            # Calculate by severity
            by_severity = {}
            for severity in set(event.severity for event in period_events):
                severity_events = [e for e in period_events if e.severity == severity]
                severity_success = sum(1 for e in severity_events if e.success)
                by_severity[severity] = severity_success / len(severity_events)
            
            calculation = SuccessRateCalculation(
                timestamp=datetime.now(),
                period_hours=period_hours,
                success_rate=success_rate,
                total_attempts=len(period_events),
                successful_attempts=successful,
                failed_attempts=failed,
                rolled_back_attempts=rolled_back,
                by_category=by_category,
                by_agent=by_agent,
                by_severity=by_severity
            )
            
            self.success_rate_history.append(calculation)
            
            # Keep only recent calculations
            if len(self.success_rate_history) > 1000:
                self.success_rate_history = self.success_rate_history[-1000:]
            
            return calculation
            
        except Exception as e:
            self.logger.error(f"Failed to calculate success rate: {e}")
            return None
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard"""
        try:
            # Recent events (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_events = [
                event for event in self.healing_events
                if event.timestamp >= one_hour_ago
            ]
            
            # Calculate recent metrics
            recent_successful = sum(1 for event in recent_events if event.success)
            recent_total = len(recent_events)
            recent_success_rate = recent_successful / recent_total if recent_total > 0 else 0
            
            recent_ttr = [event.time_to_repair for event in recent_events if event.success]
            recent_mttr = statistics.mean(recent_ttr) if recent_ttr else 0
            
            # Get latest calculations
            latest_mttr = self.mttr_history[-1] if self.mttr_history else None
            latest_success_rate = self.success_rate_history[-1] if self.success_rate_history else None
            
            return {
                'timestamp': datetime.now().isoformat(),
                'recent_events': {
                    'total': recent_total,
                    'successful': recent_successful,
                    'success_rate': recent_success_rate,
                    'mttr': recent_mttr
                },
                'latest_mttr': asdict(latest_mttr) if latest_mttr else None,
                'latest_success_rate': asdict(latest_success_rate) if latest_success_rate else None,
                'targets': {
                    'mttr_target': self.mttr_target,
                    'success_rate_target': self.success_rate_target
                },
                'category_stats': dict(self.category_stats),
                'agent_stats': dict(self.agent_stats),
                'severity_stats': dict(self.severity_stats),
                'total_events': len(self.healing_events),
                'chaos_tests': len(self.chaos_test_results)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time metrics: {e}")
            return {}
    
    async def export_research_data(self, format: str = "json", period_days: int = 30) -> str:
        """Export data for research analysis"""
        try:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            # Filter data for period
            period_events = [
                event for event in self.healing_events
                if event.timestamp >= cutoff_date
            ]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                return await self._export_json(period_events, timestamp)
            elif format.lower() == "csv":
                return await self._export_csv(period_events, timestamp)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export research data: {e}")
            return ""
    
    async def _export_json(self, events: List[HealingEvent], timestamp: str) -> str:
        """Export data in JSON format"""
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'period_days': len(events),
                    'total_events': len(events),
                    'mttr_target': self.mttr_target,
                    'success_rate_target': self.success_rate_target
                },
                'events': [asdict(event) for event in events],
                'mttr_history': [asdict(calc) for calc in self.mttr_history[-100:]],
                'success_rate_history': [asdict(calc) for calc in self.success_rate_history[-100:]],
                'chaos_tests': [asdict(test) for test in self.chaos_test_results[-100:]]
            }
            
            filename = f"healing_metrics_{timestamp}.json"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported JSON data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return ""
    
    async def _export_csv(self, events: List[HealingEvent], timestamp: str) -> str:
        """Export data in CSV format"""
        try:
            filename = f"healing_events_{timestamp}.csv"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'event_id', 'timestamp', 'error_id', 'error_type', 'agent_name',
                    'category', 'severity', 'success', 'time_to_repair', 'fix_applied',
                    'validation_passed', 'rollback_performed', 'backup_created',
                    'llm_response_time', 'patch_validation_time', 'hot_reload_time',
                    'fix_description'
                ]
                writer.writerow(header)
                
                # Write events
                for event in events:
                    row = [
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.error_id,
                        event.error_type,
                        event.agent_name,
                        event.category,
                        event.severity,
                        event.success,
                        event.time_to_repair,
                        event.fix_applied,
                        event.validation_passed,
                        event.rollback_performed,
                        event.backup_created,
                        event.llm_response_time,
                        event.patch_validation_time,
                        event.hot_reload_time,
                        event.fix_description
                    ]
                    writer.writerow(row)
            
            self.logger.info(f"Exported CSV data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return ""
    
    async def run_chaos_test(self, test_config: Dict[str, Any]) -> ChaosTestResult:
        """Run a chaos engineering test"""
        try:
            test_type = test_config.get('type', 'error_injection')
            target_agent = test_config.get('target_agent', 'unknown')
            error_type = test_config.get('error_type', 'runtime_error')
            
            self.logger.info(f"Running chaos test: {test_type} on {target_agent}")
            
            start_time = time.time()
            
            # Record system state before test
            pre_test_state = await self._capture_system_state()
            
            # Inject error (this would integrate with the actual system)
            healing_triggered = True  # This would be determined by the system
            healing_successful = True  # This would be determined by the system
            
            # Wait for healing or timeout
            await asyncio.sleep(test_config.get('timeout', 60))
            
            time_to_heal = time.time() - start_time
            
            # Record system state after test
            post_test_state = await self._capture_system_state()
            
            # Calculate system impact
            system_impact = {
                'pre_test_state': pre_test_state,
                'post_test_state': post_test_state,
                'recovery_time': time_to_heal,
                'service_disruption': time_to_heal > self.mttr_target
            }
            
            # Determine if test passed
            test_passed = (
                healing_triggered and
                healing_successful and
                time_to_heal <= self.mttr_target * 2  # Allow some tolerance
            )
            
            # Record the test result
            await self.record_chaos_test(
                test_type=test_type,
                target_agent=target_agent,
                error_injected=error_type,
                healing_triggered=healing_triggered,
                healing_successful=healing_successful,
                time_to_heal=time_to_heal,
                system_impact=system_impact,
                test_passed=test_passed,
                notes=test_config.get('notes', '')
            )
            
            return self.chaos_test_results[-1]
            
        except Exception as e:
            self.logger.error(f"Chaos test failed: {e}")
            return None
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for chaos testing"""
        # This would integrate with the actual system monitoring
        return {
            'timestamp': datetime.now().isoformat(),
            'active_agents': [],  # Would be populated from actual system
            'error_rate': 0.0,
            'response_time': 0.0,
            'system_health': 'healthy'
        }
    
    async def _metrics_calculation_loop(self):
        """Background loop for calculating metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Calculate every 5 minutes
                
                # Calculate MTTR for different periods
                for period in self.mttr_periods:
                    await self.calculate_mttr(period)
                
                # Calculate success rates for different periods
                for period in self.success_rate_periods:
                    await self.calculate_success_rate(period)
                
            except Exception as e:
                self.logger.error(f"Metrics calculation loop error: {e}")
    
    async def _auto_export_loop(self):
        """Background loop for automatic data export"""
        while True:
            try:
                await asyncio.sleep(self.auto_export_interval)
                
                # Auto-export data
                await self.export_research_data("json", 7)  # Last 7 days
                await self.export_research_data("csv", 7)
                
            except Exception as e:
                self.logger.error(f"Auto export loop error: {e}")
    
    async def _load_historical_data(self):
        """Load historical data from disk"""
        try:
            # Load healing events
            events_file = self.metrics_dir / "healing_events.json"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                    for event_data in events_data:
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        self.healing_events.append(HealingEvent(**event_data))
            
            # Load MTTR history
            mttr_file = self.metrics_dir / "mttr_history.json"
            if mttr_file.exists():
                with open(mttr_file, 'r') as f:
                    mttr_data = json.load(f)
                    for calc_data in mttr_data:
                        calc_data['timestamp'] = datetime.fromisoformat(calc_data['timestamp'])
                        self.mttr_history.append(MTTRCalculation(**calc_data))
            
            # Load success rate history
            success_rate_file = self.metrics_dir / "success_rate_history.json"
            if success_rate_file.exists():
                with open(success_rate_file, 'r') as f:
                    sr_data = json.load(f)
                    for calc_data in sr_data:
                        calc_data['timestamp'] = datetime.fromisoformat(calc_data['timestamp'])
                        self.success_rate_history.append(SuccessRateCalculation(**calc_data))
            
            # Load chaos test results
            chaos_file = self.metrics_dir / "chaos_tests.json"
            if chaos_file.exists():
                with open(chaos_file, 'r') as f:
                    chaos_data = json.load(f)
                    for test_data in chaos_data:
                        test_data['timestamp'] = datetime.fromisoformat(test_data['timestamp'])
                        self.chaos_test_results.append(ChaosTestResult(**test_data))
            
            self.logger.info(f"Loaded {len(self.healing_events)} historical events")
            
        except Exception as e:
            self.logger.warning(f"Failed to load historical data: {e}")
    
    async def _save_data(self):
        """Save current data to disk"""
        try:
            # Save healing events
            events_data = [asdict(event) for event in self.healing_events]
            with open(self.metrics_dir / "healing_events.json", 'w') as f:
                json.dump(events_data, f, indent=2, default=str)
            
            # Save MTTR history
            mttr_data = [asdict(calc) for calc in self.mttr_history]
            with open(self.metrics_dir / "mttr_history.json", 'w') as f:
                json.dump(mttr_data, f, indent=2, default=str)
            
            # Save success rate history
            sr_data = [asdict(calc) for calc in self.success_rate_history]
            with open(self.metrics_dir / "success_rate_history.json", 'w') as f:
                json.dump(sr_data, f, indent=2, default=str)
            
            # Save chaos test results
            chaos_data = [asdict(test) for test in self.chaos_test_results]
            with open(self.metrics_dir / "chaos_tests.json", 'w') as f:
                json.dump(chaos_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
    
    async def get_performance_report(self, period_days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            period_events = [
                event for event in self.healing_events
                if event.timestamp >= cutoff_date
            ]
            
            if not period_events:
                return {"error": "No data available for period"}
            
            # Calculate metrics
            total_events = len(period_events)
            successful_events = sum(1 for e in period_events if e.success)
            success_rate = successful_events / total_events
            
            ttr_values = [e.time_to_repair for e in period_events if e.success]
            mttr = statistics.mean(ttr_values) if ttr_values else 0
            
            # Category breakdown
            category_breakdown = {}
            for category in set(e.category for e in period_events):
                cat_events = [e for e in period_events if e.category == category]
                cat_success = sum(1 for e in cat_events if e.success)
                category_breakdown[category] = {
                    'total': len(cat_events),
                    'successful': cat_success,
                    'success_rate': cat_success / len(cat_events)
                }
            
            # Performance vs targets
            mttr_target_met = mttr <= self.mttr_target
            success_rate_target_met = success_rate >= self.success_rate_target
            
            return {
                'period_days': period_days,
                'total_events': total_events,
                'successful_events': successful_events,
                'success_rate': success_rate,
                'mttr': mttr,
                'targets': {
                    'mttr_target': self.mttr_target,
                    'success_rate_target': self.success_rate_target,
                    'mttr_target_met': mttr_target_met,
                    'success_rate_target_met': success_rate_target_met
                },
                'category_breakdown': category_breakdown,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown healing metrics"""
        try:
            # Save data before shutdown
            await self._save_data()
            
            self.logger.info("HealingMetrics shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")