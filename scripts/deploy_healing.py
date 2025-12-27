#!/usr/bin/env python3
"""
Deploy Healing System Script

This script deploys the self-healing system, including:
- Environment validation
- Dependency installation
- Configuration setup
- Service initialization
- Health checks
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logger import get_logger
from utils.config import load_config


class HealingSystemDeployer:
    """Deployer for the self-healing system"""
    
    def __init__(self):
        self.logger = get_logger("healing_deployer")
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.data_dir = project_root / "data"
        
    async def deploy(self) -> bool:
        """Deploy the complete healing system"""
        try:
            self.logger.info("Starting healing system deployment...")
            
            # Step 1: Validate environment
            if not await self._validate_environment():
                return False
            
            # Step 2: Check dependencies
            if not await self._check_dependencies():
                return False
            
            # Step 3: Setup directories
            if not await self._setup_directories():
                return False
            
            # Step 4: Validate configuration
            if not await self._validate_configuration():
                return False
            
            # Step 5: Initialize services
            if not await self._initialize_services():
                return False
            
            # Step 6: Run health checks
            if not await self._run_health_checks():
                return False
            
            self.logger.info("✅ Healing system deployment completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            return False
    
    async def _validate_environment(self) -> bool:
        """Validate deployment environment"""
        self.logger.info("Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8+ required")
            return False
        
        # Check required directories
        required_dirs = ["agents", "core", "config", "utils"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.logger.error(f"Required directory missing: {dir_name}")
                return False
        
        # Check required files
        required_files = [
            "agents/healing_agent.py",
            "core/llm_client.py",
            "core/error_detector.py",
            "core/code_patcher.py",
            "core/healing_metrics.py",
            "config/healing.yaml"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.logger.error(f"Required file missing: {file_path}")
                return False
        
        self.logger.info("✅ Environment validation passed")
        return True
    
    async def _check_dependencies(self) -> bool:
        """Check and install dependencies"""
        self.logger.info("Checking dependencies...")
        
        required_packages = [
            "aiohttp",
            "beautifulsoup4",
            "pyyaml",
            "pytest",
            "pytest-asyncio"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.warning(f"Missing packages: {missing_packages}")
            
            # Try to install missing packages
            try:
                import subprocess
                for package in missing_packages:
                    self.logger.info(f"Installing {package}...")
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        self.logger.error(f"Failed to install {package}: {result.stderr}")
                        return False
                    else:
                        self.logger.info(f"✅ Installed {package}")
            except Exception as e:
                self.logger.error(f"Failed to install dependencies: {e}")
                return False
        
        self.logger.info("✅ Dependencies check passed")
        return True
    
    async def _setup_directories(self) -> bool:
        """Setup required directories"""
        self.logger.info("Setting up directories...")
        
        required_dirs = [
            "data/backups",
            "data/temp",
            "data/metrics",
            "data/exports",
            "data/logs",
            "data/state"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {dir_path}: {e}")
                return False
        
        self.logger.info("✅ Directory setup completed")
        return True
    
    async def _validate_configuration(self) -> bool:
        """Validate configuration files"""
        self.logger.info("Validating configuration...")
        
        try:
            # Load healing configuration
            healing_config_path = self.config_dir / "healing.yaml"
            if not healing_config_path.exists():
                self.logger.error("Healing configuration file not found")
                return False
            
            healing_config = load_config(str(healing_config_path))
            
            # Validate required configuration sections
            required_sections = ["healing_agent", "llm", "error_detection", "code_patching", "metrics"]
            for section in required_sections:
                if section not in healing_config:
                    self.logger.error(f"Missing configuration section: {section}")
                    return False
            
            # Validate LLM configuration
            llm_config = healing_config.get("llm", {})
            if not llm_config.get("base_url"):
                self.logger.error("LLM base_url not configured")
                return False
            
            # Test Ollama connection
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{llm_config['base_url']}/api/tags", timeout=10) as response:
                        if response.status != 200:
                            self.logger.warning(f"Ollama server not accessible: {response.status}")
            except Exception as e:
                self.logger.warning(f"Could not connect to Ollama: {e}")
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _initialize_services(self) -> bool:
        """Initialize healing system services"""
        self.logger.info("Initializing services...")
        
        try:
            # Import and initialize healing agent
            from agents.healing_agent import HealingAgent
            
            healing_config = load_config(str(self.config_dir / "healing.yaml"))
            healing_agent = HealingAgent("healing_agent", healing_config.get("healing_agent", {}))
            
            # Initialize healing agent
            if not await healing_agent.initialize():
                self.logger.error("Failed to initialize healing agent")
                return False
            
            # Test healing agent status
            status = await healing_agent.get_status()
            self.logger.info(f"Healing agent status: {status.get('status', 'unknown')}")
            
            self.logger.info("✅ Service initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            return False
    
    async def _run_health_checks(self) -> bool:
        """Run system health checks"""
        self.logger.info("Running health checks...")
        
        health_checks = [
            ("File System", self._check_file_system),
            ("Configuration", self._check_config_health),
            ("Dependencies", self._check_dependency_health),
            ("Services", self._check_service_health)
        ]
        
        all_passed = True
        
        for check_name, check_func in health_checks:
            try:
                self.logger.info(f"Running {check_name} health check...")
                if await check_func():
                    self.logger.info(f"✅ {check_name} health check passed")
                else:
                    self.logger.error(f"❌ {check_name} health check failed")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {check_name} health check error: {e}")
                all_passed = False
        
        return all_passed
    
    async def _check_file_system(self) -> bool:
        """Check file system health"""
        # Check if we can write to data directories
        test_file = self.data_dir / "health_check.tmp"
        try:
            test_file.write_text("health check")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    async def _check_config_health(self) -> bool:
        """Check configuration health"""
        try:
            healing_config = load_config(str(self.config_dir / "healing.yaml"))
            agents_config = load_config(str(self.config_dir / "agents.yaml"))
            
            # Check if healing agent is configured
            if "healing_agent" not in agents_config.get("agents", {}):
                return False
            
            return True
        except Exception:
            return False
    
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
            # Try to import healing components
            from agents.healing_agent import HealingAgent
            from core.llm_client import LLMClient
            from core.error_detector import ErrorDetector
            from core.code_patcher import CodePatcher
            from core.healing_metrics import HealingMetrics
            return True
        except ImportError:
            return False
    
    def print_deployment_summary(self):
        """Print deployment summary"""
        print("\n" + "="*60)
        print("🚀 HEALING SYSTEM DEPLOYMENT SUMMARY")
        print("="*60)
        print(f"📁 Project Root: {self.project_root}")
        print(f"⚙️  Config Dir: {self.config_dir}")
        print(f"📊 Data Dir: {self.data_dir}")
        print("\n📋 DEPLOYED COMPONENTS:")
        print("  ✅ Healing Agent Core")
        print("  ✅ LLM Client (Ollama)")
        print("  ✅ Error Detection System")
        print("  ✅ Code Patching & Hot-Reload")
        print("  ✅ Healing Metrics System")
        print("  ✅ Configuration Files")
        print("  ✅ Orchestrator Integration")
        print("  ✅ Comprehensive Tests")
        print("  ✅ Deployment Scripts")
        print("\n🎯 PERFORMANCE TARGETS:")
        print("  📈 MTTR Target: < 60 seconds")
        print("  🎯 Success Rate Target: > 80%")
        print("  ⏰ Availability Target: > 95%")
        print("\n📊 MONITORING ENDPOINTS:")
        print("  🔍 Healing Metrics: data/metrics/")
        print("  📈 Export Data: data/exports/")
        print("  💾 Backups: data/backups/")
        print("  📝 Logs: data/logs/")
        print("\n🚀 NEXT STEPS:")
        print("  1. Start Ollama service: ollama serve")
        print("  2. Run healing system: python scripts/run_healing.py")
        print("  3. Monitor status: python scripts/healing_status.py")
        print("  4. Run tests: python -m pytest tests/test_healing_system.py")
        print("="*60)


async def main():
    """Main deployment function"""
    deployer = HealingSystemDeployer()
    
    print("Starting Healing System Deployment...")
    print(f"Project directory: {deployer.project_root}")
    
    success = await deployer.deploy()
    
    if success:
        deployer.print_deployment_summary()
        print("\n✅ Deployment completed successfully!")
        return 0
    else:
        print("\n❌ Deployment failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))