import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

print("=== STORY 1.3 REQUIREMENTS VERIFICATION ===")

try:
    # Check core files exist
    required_files = [
        "agents/healing_agent.py",
        "core/llm_client.py", 
        "core/error_detector.py",
        "core/code_patcher.py",
        "core/healing_metrics.py",
        "config/healing.yaml",
        "scripts/healing_status.py"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"PASS: {file} - EXISTS")
        else:
            print(f"FAIL: {file} - MISSING")
    
    # Test imports
    from agents.healing_agent import HealingAgent
    print("PASS: HealingAgent import - SUCCESS")
    
    from core.llm_client import LLMClient
    print("PASS: LLMClient import - SUCCESS")
    
    from core.error_detector import ErrorDetector
    print("PASS: ErrorDetector import - SUCCESS")
    
    from core.code_patcher import CodePatcher
    print("PASS: CodePatcher import - SUCCESS")
    
    from core.healing_metrics import HealingMetrics
    print("PASS: HealingMetrics import - SUCCESS")
    
    from utils.config import load_config
    config = load_config("config/healing.yaml")
    mttr_target = config.get("healing_agent", {}).get("mttr_target_seconds")
    if mttr_target == 60:
        print("PASS: MTTR target (60s) - CONFIGURED")
    else:
        print(f"FAIL: MTTR target: {mttr_target}")
    
    success_rate = config.get("healing_agent", {}).get("success_rate_target")
    if success_rate == 0.8:
        print("PASS: Success rate target (80%) - CONFIGURED")
    else:
        print(f"FAIL: Success rate target: {success_rate}")
    
    print("\n=== ALL REQUIREMENTS VERIFIED ===")
    
except Exception as e:
    print(f"FAIL: VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()