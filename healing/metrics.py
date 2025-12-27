"""
Healing Metrics

Metrics collection for self-healing operations including
MTTR tracking, success rates, and empirical data.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter

from utils.logger import get_logger


class MetricType(Enum):
    """Types of metrics collected"""
    HEALING_OPERATION = "healing_operation"
    ERROR_PATTERN = "error_pattern"
    PERFORMANCE = "performance"
    SYSTEM_HEALTH = "system_health"
    RESEARCH_DATA = "research_data"


@dataclass
class HealingEvent:
    """Record of a healing event"""
    event_id: str
    timestamp: datetime
    error_id: str
    error_type: str
    agent_name: str
    success: bool
    time_to_repair: float
    fix_applied: bool
    backup_created: bool
    validation_passed: bool
    rollback_performed: bool
    error_category: str
    severity: int
    confidence: float
    mttr_target: float
    success_rate_target: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for the healing system"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    llm_response_time: float
    patch_application_time: float
    validation_time: float
    total_operations: int
    concurrent_operations: int


@dataclass
class SystemHealthMetrics:
    """System health indicators"""
    uptime: float
    error_rate: float
    healing_success_rate: float
    average_mttr: float
    active_errors: int
    queued_healing_operations: int
    system_load: float
    available_resources: float


class HealingMetrics:
    """
    Comprehensive metrics collection for self-healing operations
    
    This class tracks MTTR, success rates, performance metrics, and provides
    research data export capabilities for empirical validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize healing metrics"""
        self.config = config or {}
        self.logger = get_logger("healing_metrics")
        
        # Storage
        self.healing_events: List[HealingEvent] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.system_health_metrics: List[SystemHealthMetrics] = []
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.daily_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.metrics_file = Path(self.config.get('metrics_file', 'data/metrics/healing_metrics.json'))
        self.research_data_dir = Path(self.config.get('research_data_dir', 'data/research'))
        self.max_events = self.config.get('max_events', 10000)
        self.export_interval = self.config.get('export_interval', 3600)  # 1 hour
        self.retention_days = self.config.get('retention_days', 30)
        
        # Targets
        self.mttr_target = self.config.get('mttr_target', 60.0)  # seconds
        self.success_rate_target = self.config.get('success_rate_target', 0.8)
        
        # Ensure directories exist
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.research_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Background tasks
        self._export_task = None
        self._cleanup_task = None
        
        self.logger.info("HealingMetrics initialized")
    
    async def initialize(self) -> bool:
        """Initialize metrics system"""
        try:
            # Load existing metrics
            await self._load_metrics()
            
            # Start background tasks
            self._export_task = asyncio.create_task(self._export_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
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
        error_category: str = "unknown",
        severity: int = 2,
        confidence: float = 0.0,
        fix_applied: bool = False,
        backup_created: bool = False,
        validation_passed: bool = False,
        rollback_performed: bool = False
    ):
        """Record a healing event"""
        try:
            event = HealingEvent(
                event_id=f"event_{int(time.time())}_{hash(error_id) % 10000:04d}",
                timestamp=datetime.now(),
                error_id=error_id,
                error_type=error_type,
                agent_name=agent_name,
                success=success,
                time_to_repair=time_to_repair,
                fix_applied=fix_applied,
                backup_created=backup_created,
                validation_passed=validation_passed,
                rollback_performed=rollback_performed,
                error_category=error_category,
                severity=severity,
                confidence=confidence,
                mttr_target=self.mttr_target,
                success_rate_target=self.success_rate_target
            )
            
            self.healing_events.append(event)
            
            # Update error patterns
            pattern_key = f"{error_category}:{agent_name}"
            self.error_patterns[pattern_key] += 1
            
            # Update daily stats
            date_key = event.timestamp.strftime("%Y-%m-%d")
            if date_key not in self.daily_stats:
                self.daily_stats[date_key] = {
                    'total_events': 0,
                    'successful_events': 0,
                    'failed_events': 0,
                    'total_mttr': 0.0,
                    'min_mttr': float('inf'),
                    'max_mttr': 0.0,
                    'error_categories': defaultdict(int),
                    'agent_performance': defaultdict(lambda: {'total': 0, 'success': 0})
                }
            
            daily = self.daily_stats[date_key]
            daily['total_events'] += 1
            
            if success:
                daily['successful_events'] += 1
                daily['total_mttr'] += time_to_repair
                daily['min_mttr'] = min(daily['min_mttr'], time_to_repair)
                daily['max_mttr'] = max(daily['max_mttr'], time_to_repair)
            else:
                daily['failed_events'] += 1
            
            daily['error_categories'][error_category] += 1
            daily['agent_performance'][agent_name]['total'] += 1
            if success:
                daily['agent_performance'][agent_name]['success'] += 1
            
            # Limit events in memory
            if len(self.healing_events) > self.max_events:
                self.healing_events = self.healing_events[-self.max_events:]
            
            self.logger.debug(f"Recorded healing event: {error_id} - {'SUCCESS' if success else 'FAILED'}")
            
        except Exception as e:
            self.logger.error(f"Failed to record healing event: {e}")
    
    async def record_performance_metrics(self, metrics: Dict[str, float]):
        """Record performance metrics"""
        try:
            perf_metrics = PerformanceMetrics(
                cpu_usage=metrics.get('cpu_usage', 0.0),
                memory_usage=metrics.get('memory_usage', 0.0),
                disk_usage=metrics.get('disk_usage', 0.0),
                network_latency=metrics.get('network_latency', 0.0),
                llm_response_time=metrics.get('llm_response_time', 0.0),
                patch_application_time=metrics.get('patch_application_time', 0.0),
                validation_time=metrics.get('validation_time', 0.0),
                total_operations=metrics.get('total_operations', 0),
                concurrent_operations=metrics.get('concurrent_operations', 0)
            )
            
            self.performance_metrics.append(perf_metrics)
            
            # Keep only last 1000 performance records
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]
                
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")
    
    async def record_system_health(self, health_data: Dict[str, float]):
        """Record system health metrics"""
        try:
            health_metrics = SystemHealthMetrics(
                uptime=health_data.get('uptime', 0.0),
                error_rate=health_data.get('error_rate', 0.0),
                healing_success_rate=health_data.get('healing_success_rate', 0.0),
                average_mttr=health_data.get('average_mttr', 0.0),
                active_errors=health_data.get('active_errors', 0),
                queued_healing_operations=health_data.get('queued_healing_operations', 0),
                system_load=health_data.get('system_load', 0.0),
                available_resources=health_data.get('available_resources', 100.0)
            )
            
            self.system_health_metrics.append(health_metrics)
            
            # Keep only last 1000 health records
            if len(self.system_health_metrics) > 1000:
                self.system_health_metrics = self.system_health_metrics[-1000:]
                
        except Exception as e:
            self.logger.error(f"Failed to record system health: {e}")
    
    def get_success_rate(self, time_window: Optional[int] = None) -> float:
        """
        Calculate success rate for healing operations
        
        Args:
            time_window: Time window in hours (None for all time)
            
        Returns:
            Success rate as percentage (0-100)
        """
        try:
            if not self.healing_events:
                return 0.0
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - timedelta(hours=time_window)
                events = [e for e in self.healing_events if e.timestamp > cutoff_time]
            else:
                events = self.healing_events
            
            if not events:
                return 0.0
            
            successful = sum(1 for e in events if e.success)
            return (successful / len(events)) * 100
            
        except Exception as e:
            self.logger.error(f"Failed to calculate success rate: {e}")
            return 0.0
    
    def get_average_mttr(self, time_window: Optional[int] = None) -> float:
        """
        Calculate Mean Time To Repair (MTTR)
        
        Args:
            time_window: Time window in hours (None for all time)
            
        Returns:
            Average MTTR in seconds
        """
        try:
            if not self.healing_events:
                return 0.0
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - timedelta(hours=time_window)
                events = [e for e in self.healing_events if e.success and e.timestamp > cutoff_time]
            else:
                events = [e for e in self.healing_events if e.success]
            
            if not events:
                return 0.0
            
            total_mttr = sum(e.time_to_repair for e in events)
            return total_mttr / len(events)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate MTTR: {e}")
            return 0.0
    
    def get_error_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of error patterns"""
        return dict(self.error_patterns)
    
    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics by agent"""
        agent_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'avg_mttr': 0.0})
        
        for event in self.healing_events:
            agent = event.agent_name
            agent_stats[agent]['total'] += 1
            if event.success:
                agent_stats[agent]['success'] += 1
        
        # Calculate success rates and average MTTR
        for agent, stats in agent_stats.items():
            if stats['total'] > 0:
                stats['success_rate'] = (stats['success'] / stats['total']) * 100
                
                # Calculate average MTTR for successful operations
                successful_events = [e for e in self.healing_events 
                                  if e.agent_name == agent and e.success]
                if successful_events:
                    stats['avg_mttr'] = sum(e.time_to_repair for e in successful_events) / len(successful_events)
            else:
                stats['success_rate'] = 0.0
        
        return dict(agent_stats)
    
    def get_daily_statistics(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Get daily statistics for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        daily_stats = {}
        
        for date_key, stats in self.daily_stats.items():
            date_obj = datetime.strptime(date_key, "%Y-%m-%d")
            if date_obj >= cutoff_date:
                # Calculate derived metrics
                total = stats['total_events']
                if total > 0:
                    stats['success_rate'] = (stats['successful_events'] / total) * 100
                    if stats['successful_events'] > 0:
                        stats['avg_mttr'] = stats['total_mttr'] / stats['successful_events']
                    else:
                        stats['avg_mttr'] = 0.0
                else:
                    stats['success_rate'] = 0.0
                    stats['avg_mttr'] = 0.0
                
                # Convert defaultdicts to regular dicts
                stats['error_categories'] = dict(stats['error_categories'])
                stats['agent_performance'] = dict(stats['agent_performance'])
                
                daily_stats[date_key] = stats
        
        return daily_stats
    
    def get_research_data(self) -> Dict[str, Any]:
        """Get comprehensive research data for empirical analysis"""
        try:
            # Basic statistics
            total_events = len(self.healing_events)
            successful_events = sum(1 for e in self.healing_events if e.success)
            failed_events = total_events - successful_events
            
            # Time-based analysis
            last_24h = datetime.now() - timedelta(hours=24)
            recent_events = [e for e in self.healing_events if e.timestamp > last_24h]
            recent_success = sum(1 for e in recent_events if e.success)
            
            # MTTR analysis
            successful_mttrs = [e.time_to_repair for e in self.healing_events if e.success]
            mttr_stats = {}
            if successful_mttrs:
                mttr_stats = {
                    'mean': sum(successful_mttrs) / len(successful_mttrs),
                    'median': sorted(successful_mttrs)[len(successful_mttrs) // 2],
                    'min': min(successful_mttrs),
                    'max': max(successful_mttrs),
                    'std_dev': self._calculate_std_dev(successful_mttrs)
                }
            
            # Error category analysis
            category_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            for event in self.healing_events:
                category_stats[event.error_category]['total'] += 1
                if event.success:
                    category_stats[event.error_category]['success'] += 1
            
            # Calculate success rates by category
            for category, stats in category_stats.items():
                if stats['total'] > 0:
                    stats['success_rate'] = (stats['success'] / stats['total']) * 100
            
            # Performance metrics
            performance_summary = {}
            if self.performance_metrics:
                cpu_usage = [m.cpu_usage for m in self.performance_metrics]
                memory_usage = [m.memory_usage for m in self.performance_metrics]
                
                performance_summary = {
                    'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                    'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                    'max_cpu_usage': max(cpu_usage),
                    'max_memory_usage': max(memory_usage)
                }
            
            return {
                'summary': {
                    'total_events': total_events,
                    'successful_events': successful_events,
                    'failed_events': failed_events,
                    'overall_success_rate': (successful_events / total_events * 100) if total_events > 0 else 0,
                    'recent_24h_events': len(recent_events),
                    'recent_24h_success_rate': (recent_success / len(recent_events) * 100) if recent_events else 0
                },
                'mttr_analysis': mttr_stats,
                'error_categories': dict(category_stats),
                'agent_performance': self.get_agent_performance(),
                'error_patterns': dict(self.error_patterns),
                'performance_metrics': performance_summary,
                'targets': {
                    'mttr_target': self.mttr_target,
                    'success_rate_target': self.success_rate_target * 100,
                    'mttr_achieved': mttr_stats.get('mean', 0) <= self.mttr_target,
                    'success_rate_achieved': (successful_events / total_events * 100 if total_events > 0 else 0) >= (self.success_rate_target * 100)
                },
                'data_collection_period': {
                    'start_date': min(e.timestamp for e in self.healing_events).isoformat() if self.healing_events else None,
                    'end_date': max(e.timestamp for e in self.healing_events).isoformat() if self.healing_events else None,
                    'total_days': (max(e.timestamp for e in self.healing_events) - min(e.timestamp for e in self.healing_events)).days if len(self.healing_events) > 1 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate research data: {e}")
            return {}
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    async def export_research_data(self, output_dir: Optional[str] = None) -> bool:
        """Export research data for empirical analysis"""
        try:
            export_dir = Path(output_dir) if output_dir else self.research_data_dir
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export comprehensive research data
            research_data = self.get_research_data()
            research_file = export_dir / f"healing_research_data_{timestamp}.json"
            with open(research_file, 'w') as f:
                json.dump(research_data, f, indent=2, default=str)
            
            # Export detailed events CSV
            events_file = export_dir / f"healing_events_{timestamp}.csv"
            with open(events_file, 'w') as f:
                if self.healing_events:
                    # Write header
                    f.write("event_id,timestamp,error_id,error_type,agent_name,success,time_to_repair,")
                    f.write("fix_applied,backup_created,validation_passed,rollback_performed,")
                    f.write("error_category,severity,confidence\n")
                    
                    # Write events
                    for event in self.healing_events:
                        f.write(f"{event.event_id},{event.timestamp.isoformat()},{event.error_id},")
                        f.write(f"{event.error_type},{event.agent_name},{event.success},{event.time_to_repair},")
                        f.write(f"{event.fix_applied},{event.backup_created},{event.validation_passed},")
                        f.write(f"{event.rollback_performed},{event.error_category},{event.severity},{event.confidence}\n")
            
            # Export daily statistics
            daily_stats = self.get_daily_statistics(30)  # Last 30 days
            daily_file = export_dir / f"daily_statistics_{timestamp}.json"
            with open(daily_file, 'w') as f:
                json.dump(daily_stats, f, indent=2, default=str)
            
            # Export performance metrics
            if self.performance_metrics:
                perf_file = export_dir / f"performance_metrics_{timestamp}.csv"
                with open(perf_file, 'w') as f:
                    f.write("timestamp,cpu_usage,memory_usage,disk_usage,network_latency,")
                    f.write("llm_response_time,patch_application_time,validation_time,")
                    f.write("total_operations,concurrent_operations\n")
                    
                    for metric in self.performance_metrics:
                        f.write(f"{datetime.now().isoformat()},{metric.cpu_usage},{metric.memory_usage},")
                        f.write(f"{metric.disk_usage},{metric.network_latency},{metric.llm_response_time},")
                        f.write(f"{metric.patch_application_time},{metric.validation_time},")
                        f.write(f"{metric.total_operations},{metric.concurrent_operations}\n")
            
            self.logger.info(f"Research data exported to {export_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export research data: {e}")
            return False
    
    async def _load_metrics(self):
        """Load existing metrics from disk"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Load healing events
                if 'healing_events' in data:
                    for event_data in data['healing_events']:
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        self.healing_events.append(HealingEvent(**event_data))
                
                # Load error patterns
                if 'error_patterns' in data:
                    self.error_patterns = defaultdict(int, data['error_patterns'])
                
                # Load daily stats
                if 'daily_stats' in data:
                    for date_key, stats in data['daily_stats'].items():
                        # Convert defaultdicts back
                        if 'error_categories' in stats:
                            stats['error_categories'] = defaultdict(int, stats['error_categories'])
                        if 'agent_performance' in stats:
                            stats['agent_performance'] = defaultdict(
                                lambda: {'total': 0, 'success': 0}, 
                                {k: defaultdict(lambda: {'total': 0, 'success': 0}, v) 
                                 for k, v in stats['agent_performance'].items()}
                            )
                        self.daily_stats[date_key] = stats
                
                self.logger.info(f"Loaded {len(self.healing_events)} healing events from disk")
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing metrics: {e}")
    
    async def _save_metrics(self):
        """Save metrics to disk"""
        try:
            data = {
                'healing_events': [asdict(event) for event in self.healing_events],
                'error_patterns': dict(self.error_patterns),
                'daily_stats': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert daily stats to serializable format
            for date_key, stats in self.daily_stats.items():
                data['daily_stats'][date_key] = {
                    'total_events': stats['total_events'],
                    'successful_events': stats['successful_events'],
                    'failed_events': stats['failed_events'],
                    'total_mttr': stats['total_mttr'],
                    'min_mttr': stats['min_mttr'],
                    'max_mttr': stats['max_mttr'],
                    'error_categories': dict(stats['error_categories']),
                    'agent_performance': {
                        agent: dict(perf) for agent, perf in stats['agent_performance'].items()
                    }
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    async def _export_loop(self):
        """Background loop for periodic data export"""
        while True:
            try:
                await asyncio.sleep(self.export_interval)
                await self.export_research_data()
                await self._save_metrics()
            except Exception as e:
                self.logger.error(f"Export loop error: {e}")
    
    async def _cleanup_loop(self):
        """Background loop for data cleanup"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean old healing events
            original_count = len(self.healing_events)
            self.healing_events = [e for e in self.healing_events if e.timestamp > cutoff_date]
            removed_events = original_count - len(self.healing_events)
            
            # Clean old daily stats
            dates_to_remove = []
            for date_key in self.daily_stats:
                date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                if date_obj < cutoff_date:
                    dates_to_remove.append(date_key)
            
            for date_key in dates_to_remove:
                del self.daily_stats[date_key]
            
            # Clean old performance metrics
            if self.performance_metrics:
                # Keep only recent performance metrics (last 1000)
                self.performance_metrics = self.performance_metrics[-1000:]
            
            # Clean old system health metrics
            if self.system_health_metrics:
                self.system_health_metrics = self.system_health_metrics[-1000:]
            
            if removed_events > 0:
                self.logger.info(f"Cleaned up {removed_events} old healing events")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        try:
            return {
                'healing_performance': {
                    'success_rate': self.get_success_rate(),
                    'success_rate_24h': self.get_success_rate(24),
                    'average_mttr': self.get_average_mttr(),
                    'average_mttr_24h': self.get_average_mttr(24),
                    'total_events': len(self.healing_events),
                    'successful_events': sum(1 for e in self.healing_events if e.success),
                    'failed_events': sum(1 for e in self.healing_events if not e.success)
                },
                'target_achievement': {
                    'mttr_target_met': self.get_average_mttr() <= self.mttr_target,
                    'success_rate_target_met': self.get_success_rate() >= (self.success_rate_target * 100),
                    'mttr_target': self.mttr_target,
                    'success_rate_target': self.success_rate_target * 100
                },
                'error_analysis': {
                    'error_patterns': dict(self.error_patterns),
                    'agent_performance': self.get_agent_performance(),
                    'daily_statistics': self.get_daily_statistics(7)
                },
                'system_performance': {
                    'performance_metrics_count': len(self.performance_metrics),
                    'health_metrics_count': len(self.system_health_metrics),
                    'data_retention_days': self.retention_days
                },
                'research_data': self.get_research_data()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown metrics system"""
        try:
            self.logger.info("Shutting down HealingMetrics...")
            
            # Cancel background tasks
            if self._export_task:
                self._export_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Save final metrics
            await self._save_metrics()
            await self.export_research_data()
            
            self.logger.info("HealingMetrics shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")