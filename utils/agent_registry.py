"""
Agent Registry and Discovery System

Centralized system for agent registration, discovery,
health monitoring, and configuration management.
"""

import asyncio
import yaml
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import os

from agents.base_agent import BaseAgent, AgentStatus
from utils.logger import StructuredLogger
from utils.config import settings
from utils.global_state import get_global_state_manager, StateChangeType


class AgentRegistrationStatus(Enum):
    """Agent registration status."""
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"
    SUSPENDED = "suspended"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    agent_type: str
    capabilities: Set[str]
    status: AgentRegistrationStatus
    registered_at: datetime
    last_heartbeat: Optional[datetime] = None
    config: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    health_status: str = "unknown"
    performance_metrics: Optional[Dict[str, Any]] = None


class AgentRegistry:
    """
    Centralized agent registry for registration and discovery.
    
    Features:
    - Agent registration and deregistration
    - Capability-based discovery
    - Health monitoring and heartbeat tracking
    - Configuration management
    - Event notifications for lifecycle changes
    - Persistent storage of agent information
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = StructuredLogger("agent_registry", settings.log_level)
        
        # Registry storage
        self._agents: Dict[str, AgentInfo] = {}
        self._agent_instances: Dict[str, BaseAgent] = {}  # agent_id -> agent instance
        self._capabilities_index: Dict[str, Set[str]] = {}  # capability -> agent names
        self._lock = threading.RLock()
        
        # Event system
        self._event_listeners: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.config_file = config_file or "config/agents.yaml"
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # Health monitoring
        self.heartbeat_timeout = 300  # 5 minutes
        self.health_check_interval = 60  # 1 minute
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Load initial configuration
        self._load_agent_configs()
        
        self.logger.info("Agent registry initialized")
    
    def _load_agent_configs(self):
        """Load agent configurations from YAML file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    self.agent_configs = config_data.get('agents', {})
                    self.logger.info(f"Loaded {len(self.agent_configs)} agent configurations")
            else:
                self.logger.warning(f"Agent config file not found: {self.config_file}")
                self.agent_configs = {}
        except Exception as e:
            self.logger.error(f"Failed to load agent configs: {e}")
            self.agent_configs = {}
    
    def register_agent(self, agent: BaseAgent, capabilities: List[str] = None, 
                   config_overrides: Dict[str, Any] = None) -> bool:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent instance to register
            capabilities: List of agent capabilities
            config_overrides: Configuration overrides for this agent
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            with self._lock:
                if agent.name in self._agents:
                    self.logger.warning(f"Agent {agent.name} already registered")
                    return False
                
                # Get agent configuration
                agent_config = self.agent_configs.get(agent.name, {})
                if config_overrides:
                    agent_config.update(config_overrides)
                
                # Create agent info
                agent_info = AgentInfo(
                    name=agent.name,
                    agent_type=agent_config.get('type', 'unknown'),
                    capabilities=set(capabilities or []),
                    status=AgentRegistrationStatus.REGISTERED,
                    registered_at=datetime.utcnow(),
                    config=agent_config,
                    metadata={
                        'version': agent_config.get('version', '1.0.0'),
                        'description': agent_config.get('description', ''),
                        'author': agent_config.get('author', 'unknown')
                    }
                )
                
                # Store agent info and instance
                self._agents[agent.name] = agent_info
                self._agent_instances[agent.name] = agent
                
                # Update capabilities index
                for capability in agent_info.capabilities:
                    if capability not in self._capabilities_index:
                        self._capabilities_index[capability] = set()
                    self._capabilities_index[capability].add(agent.name)
                
                # Update global state
                state_manager = get_global_state_manager()
                state_manager.set_state(f"agent.{agent.name}", {
                    'status': 'registered',
                    'capabilities': list(agent_info.capabilities),
                    'registered_at': agent_info.registered_at.isoformat(),
                    'config': agent_config
                }, agent_name="registry")
                
                # Emit registration event
                self._emit_agent_event('registered', agent_info)
                
                self.logger.info(f"Agent {agent.name} registered with capabilities: {agent_info.capabilities}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.name}: {e}")
            return False
    
    def unregister_agent(self, agent_name: str, reason: str = "manual") -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_name: Name of agent to unregister
            reason: Reason for unregistration
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            with self._lock:
                if agent_name not in self._agents:
                    self.logger.warning(f"Agent {agent_name} not registered")
                    return False
                
                agent_info = self._agents[agent_name]
                
                # Remove from capabilities index
                for capability in agent_info.capabilities:
                    if capability in self._capabilities_index:
                        self._capabilities_index[capability].discard(agent_name)
                        if not self._capabilities_index[capability]:
                            del self._capabilities_index[capability]
                
                # Update agent status
                agent_info.status = AgentRegistrationStatus.UNREGISTERED
                
                # Update global state
                state_manager = get_global_state_manager()
                state_manager.set_state(f"agent.{agent_name}", {
                    'status': 'unregistered',
                    'unregistered_at': datetime.utcnow().isoformat(),
                    'reason': reason
                }, agent_name="registry")
                
                # Remove from registry
                del self._agents[agent_name]
                if agent_name in self._agent_instances:
                    del self._agent_instances[agent_name]
                
                # Emit unregistration event
                self._emit_agent_event('unregistered', agent_info)
                
                self.logger.info(f"Agent {agent_name} unregistered: {reason}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_name}: {e}")
            return False
    
    def discover_agents_by_capability(self, capability: str, 
                                   status_filter: Optional[List[AgentRegistrationStatus]] = None) -> List[str]:
        """
        Discover agents that have a specific capability.
        
        Args:
            capability: The capability to search for
            status_filter: Optional filter for agent status
            
        Returns:
            List of agent names with the capability
        """
        with self._lock:
            agent_names = self._capabilities_index.get(capability, set()).copy()
            
            if status_filter:
                filtered_agents = []
                for agent_name in agent_names:
                    if agent_name in self._agents:
                        agent_info = self._agents[agent_name]
                        if agent_info.status in status_filter:
                            filtered_agents.append(agent_name)
                return filtered_agents
            
            return list(agent_names)
    
    def discover_agents_by_capabilities(self, capabilities: List[str], 
                                    require_all: bool = False,
                                    status_filter: Optional[List[AgentRegistrationStatus]] = None) -> List[str]:
        """
        Discover agents with multiple capabilities.
        
        Args:
            capabilities: List of required capabilities
            require_all: If True, agent must have ALL capabilities
                         If False, agent must have AT LEAST ONE capability
            status_filter: Optional filter for agent status
            
        Returns:
            List of agent names matching the criteria
        """
        if require_all:
            # Find agents that have ALL required capabilities
            matching_agents = []
            for agent_name, agent_info in self._agents.items():
                if status_filter and agent_info.status not in status_filter:
                    continue
                
                if all(cap in agent_info.capabilities for cap in capabilities):
                    matching_agents.append(agent_name)
            return matching_agents
        else:
            # Find agents that have AT LEAST ONE required capability
            matching_agents = set()
            for capability in capabilities:
                agents_with_cap = self.discover_agents_by_capability(capability, status_filter)
                matching_agents.update(agents_with_cap)
            return list(matching_agents)
    
    def get_agent_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get information about a registered agent."""
        with self._lock:
            return self._agents.get(agent_name)
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent instance by its configured ID.
        
        This method looks up agents by their configured ID from agents.yaml
        and returns the actual agent instance for direct interaction.
        
        Args:
            agent_id: The configured ID of the agent (e.g., "orchestrator", "healing_agent")
            
        Returns:
            The agent instance if found and registered, None otherwise
        """
        with self._lock:
            # First try direct lookup by agent name
            if agent_id in self._agent_instances:
                return self._agent_instances[agent_id]
            
            # If not found, try to match by configured ID from agent configs
            for agent_name, agent_info in self._agents.items():
                config = self.agent_configs.get(agent_name, {})
                if config.get('id') == agent_id:
                    return self._agent_instances.get(agent_name)
            
            # Agent not found
            self.logger.warning(f"Agent with ID '{agent_id}' not found in registry")
            return None
    
    def get_all_agents(self, status_filter: Optional[List[AgentRegistrationStatus]] = None) -> Dict[str, AgentInfo]:
        """Get all registered agents, optionally filtered by status."""
        with self._lock:
            if status_filter:
                return {
                    name: info for name, info in self._agents.items()
                    if info.status in status_filter
                }
            return self._agents.copy()
    
    def get_agent_capabilities(self, agent_name: str) -> Set[str]:
        """Get capabilities of a specific agent."""
        agent_info = self.get_agent_info(agent_name)
        return agent_info.capabilities if agent_info else set()
    
    def update_agent_heartbeat(self, agent_name: str) -> bool:
        """Update the heartbeat timestamp for an agent."""
        try:
            with self._lock:
                if agent_name not in self._agents:
                    self.logger.warning(f"Heartbeat from unregistered agent: {agent_name}")
                    return False
                
                agent_info = self._agents[agent_name]
                agent_info.last_heartbeat = datetime.utcnow()
                
                # Update global state
                state_manager = get_global_state_manager()
                state_manager.set_state(f"agent.{agent_name}.last_heartbeat", 
                                    agent_info.last_heartbeat.isoformat(), 
                                    agent_name="registry")
                
                # Emit heartbeat event
                self._emit_agent_event('heartbeat', agent_info)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {agent_name}: {e}")
            return False
    
    def update_agent_health(self, agent_name: str, health_status: str, 
                          metrics: Dict[str, Any] = None) -> bool:
        """Update the health status of an agent."""
        try:
            with self._lock:
                if agent_name not in self._agents:
                    self.logger.warning(f"Health update for unregistered agent: {agent_name}")
                    return False
                
                agent_info = self._agents[agent_name]
                agent_info.health_status = health_status
                agent_info.performance_metrics = metrics or {}
                
                # Update global state
                state_manager = get_global_state_manager()
                state_manager.set_state(f"agent.{agent_name}.health", {
                    'status': health_status,
                    'metrics': metrics,
                    'updated_at': datetime.utcnow().isoformat()
                }, agent_name="registry")
                
                # Emit health update event
                self._emit_agent_event('health_update', agent_info)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update health for {agent_name}: {e}")
            return False
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the configuration for a specific agent."""
        return self.agent_configs.get(agent_name, {})
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in configuration."""
        config = self.get_agent_config(agent_name)
        return config.get('enabled', True)
    
    def get_capabilities_summary(self) -> Dict[str, int]:
        """Get a summary of all capabilities and agent counts."""
        with self._lock:
            summary = {}
            for capability, agents in self._capabilities_index.items():
                summary[capability] = len(agents)
            return summary
    
    def add_event_listener(self, event_type: str, listener: Callable[[AgentInfo], None]):
        """Add an event listener for agent lifecycle events."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(listener)
        self.logger.debug(f"Added agent event listener for: {event_type}")
    
    def remove_event_listener(self, event_type: str, listener: Callable[[AgentInfo], None]):
        """Remove an event listener."""
        if event_type in self._event_listeners:
            try:
                self._event_listeners[event_type].remove(listener)
                self.logger.debug(f"Removed agent event listener for: {event_type}")
            except ValueError:
                pass  # Listener not found
    
    def _emit_agent_event(self, event_type: str, agent_info: AgentInfo):
        """Emit agent lifecycle events to listeners."""
        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    listener(agent_info)
                except Exception as e:
                    self.logger.error(f"Agent event listener error: {e}")
    
    async def start_health_monitoring(self):
        """Start background health monitoring for all agents."""
        if self._monitoring_task and not self._monitoring_task.done():
            self.logger.warning("Health monitoring already running")
            return
        
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("Started agent health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("Stopped agent health monitoring")
    
    async def _health_monitoring_loop(self):
        """Background loop for health monitoring."""
        while True:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _check_agent_health(self):
        """Check health of all registered agents."""
        current_time = datetime.utcnow()
        
        with self._lock:
            for agent_name, agent_info in list(self._agents.items()):
                # Skip unregistered agents
                if agent_info.status != AgentRegistrationStatus.REGISTERED:
                    continue
                
                # Check heartbeat timeout
                if (agent_info.last_heartbeat and 
                    current_time - agent_info.last_heartbeat > timedelta(seconds=self.heartbeat_timeout)):
                    
                    self.logger.warning(f"Agent {agent_name} heartbeat timeout")
                    agent_info.health_status = "timeout"
                    
                    # Update global state
                    state_manager = get_global_state_manager()
                    state_manager.set_state(f"agent.{agent_name}.health", {
                        'status': 'timeout',
                        'last_heartbeat': agent_info.last_heartbeat.isoformat(),
                        'checked_at': current_time.isoformat()
                    }, agent_name="registry")
                    
                    # Emit health event
                    self._emit_agent_event('health_issue', agent_info)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            stats = {
                'total_agents': len(self._agents),
                'registered_agents': len([info for info in self._agents.values() 
                                      if info.status == AgentRegistrationStatus.REGISTERED]),
                'unregistered_agents': len([info for info in self._agents.values() 
                                        if info.status == AgentRegistrationStatus.UNREGISTERED]),
                'suspended_agents': len([info for info in self._agents.values() 
                                        if info.status == AgentRegistrationStatus.SUSPENDED]),
                'error_agents': len([info for info in self._agents.values() 
                                     if info.status == AgentRegistrationStatus.ERROR]),
                'capabilities_count': len(self._capabilities_index),
                'agent_instances_count': len(self._agent_instances),
                'health_monitoring_active': self._monitoring_task is not None and not self._monitoring_task.done(),
                'config_file': self.config_file,
                'loaded_configs': len(self.agent_configs)
            }
            
            # Add capability breakdown
            stats['capabilities_breakdown'] = self.get_capabilities_summary()
            
            return stats
    
    def export_registry(self, file_path: str) -> bool:
        """Export registry information to a file."""
        try:
            with self._lock:
                export_data = {
                    'exported_at': datetime.utcnow().isoformat(),
                    'registry_stats': self.get_registry_stats(),
                    'agents': {
                        name: asdict(info) for name, info in self._agents.items()
                    }
                }
            
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Registry exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    async def cleanup_stale_agents(self, max_age_hours: int = 24) -> int:
        """Clean up agents that haven't sent heartbeat in specified hours."""
        cleaned_count = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._lock:
            stale_agents = []
            for agent_name, agent_info in list(self._agents.items()):
                if (agent_info.last_heartbeat and 
                    agent_info.last_heartbeat < cutoff_time):
                    stale_agents.append(agent_name)
            
            for agent_name in stale_agents:
                self.unregister_agent(agent_name, f"stale (no heartbeat for {max_age_hours}h)")
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} stale agents")
        
        return cleaned_count
    
    def __str__(self) -> str:
        """String representation of agent registry."""
        return f"AgentRegistry(agents={len(self._agents)}, capabilities={len(self._capabilities_index)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AgentRegistry(agents={len(self._agents)}, "
                f"registered={len([info for info in self._agents.values() if info.status == AgentRegistrationStatus.REGISTERED])}, "
                f"capabilities={len(self._capabilities_index)})")


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get the singleton agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(agent: BaseAgent, capabilities: List[str] = None) -> bool:
    """Convenience function to register an agent."""
    return get_agent_registry().register_agent(agent, capabilities)


def unregister_agent(agent_name: str, reason: str = "manual") -> bool:
    """Convenience function to unregister an agent."""
    return get_agent_registry().unregister_agent(agent_name, reason)


def discover_agents_by_capability(capability: str) -> List[str]:
    """Convenience function to discover agents by capability."""
    return get_agent_registry().discover_agents_by_capability(capability)


def discover_agents_by_capabilities(capabilities: List[str], require_all: bool = False) -> List[str]:
    """Convenience function to discover agents by multiple capabilities."""
    return get_agent_registry().discover_agents_by_capabilities(capabilities, require_all)