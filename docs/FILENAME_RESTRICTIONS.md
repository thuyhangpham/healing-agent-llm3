# Filename Restrictions for ETL Sentiment Project

This document lists filenames that **CANNOT** be used in this project due to conflicts with Python reserved keywords, standard library modules, or existing project modules.

## 🚫 Python Reserved Keywords (Cannot Use)

These Python keywords cannot be used as module filenames:

- `if.py`, `else.py`, `elif.py`
- `for.py`, `while.py`
- `def.py`, `class.py`
- `import.py`, `from.py`
- `try.py`, `except.py`, `finally.py`
- `raise.py`, `assert.py`
- `return.py`, `yield.py`
- `break.py`, `continue.py`
- `pass.py`, `del.py`
- `and.py`, `or.py`, `not.py`
- `in.py`, `is.py`, `as.py`
- `with.py`, `async.py`, `await.py`
- `lambda.py`, `global.py`, `nonlocal.py`
- `True.py`, `False.py`, `None.py`

## 🚫 Standard Library Module Names (Cannot Use)

These standard library modules are imported in the project and would cause import conflicts:

- `asyncio.py` - Used for async operations
- `sys.py` - System-specific parameters
- `os.py` - Operating system interface
- `time.py` - Time-related functions
- `json.py` - JSON encoder/decoder
- `pathlib.py` - Object-oriented filesystem paths
- `datetime.py` - Date and time handling
- `subprocess.py` - Subprocess management
- `threading.py` - Thread-based parallelism
- `csv.py` - CSV file handling
- `re.py` - Regular expressions
- `hashlib.py` - Secure hashes and message digests
- `shutil.py` - High-level file operations
- `traceback.py` - Print or retrieve stack traces
- `argparse.py` - Command-line argument parsing
- `collections.py` - Specialized container datatypes
- `typing.py` - Support for type hints
- `dataclasses.py` - Data classes
- `enum.py` - Support for enumerations
- `abc.py` - Abstract base classes
- `logging.py` - Logging facility
- `yaml.py` - YAML parser (if used)

## 🚫 Existing Project Module Names (Cannot Use)

These modules already exist in the project:

### In `agents/` directory:
- `base_agent.py`
- `healing_agent.py`
- `law_search_agent.py`
- `opinion_search_agent.py`
- `orchestrator.py` ⚠️ **CONFLICT**: Also exists in `scripts/`
- `pdf_analysis_agent.py`
- `sentiment_agent.py`
- `sentiment_analysis_agent.py`

### In `scripts/` directory:
- `orchestrator.py` ⚠️ **CONFLICT**: Also exists in `agents/`
- `metrics_monitor.py`
- `evaluate_nlp_performance.py`
- `chaos_test.py`
- `run_production.py`
- `run_system.py`
- `status.py`
- `workflow_demo.py`
- And others...

### In `utils/` directory:
- `agent_registry.py`
- `communication.py`
- `config.py`
- `constants.py`
- `file_utils.py`
- `global_state.py`
- `logger.py`
- `workflow_engine.py`

### In `core/` directory:
- `chaos_engineering.py`
- `code_patcher.py`
- `error_detector.py`
- `healing_metrics.py`
- `llm_client.py`
- `research_exporter.py`

### In `healing/` directory:
- `code_patcher.py` ⚠️ **DUPLICATE**: Also in `core/`
- `error_handler.py`
- `llm_client.py` ⚠️ **DUPLICATE**: Also in `core/`
- `metrics.py`
- `validator.py`

## ⚠️ Special Cases

### Conflicting Names Between Directories:
- `orchestrator.py` exists in both `agents/` and `scripts/` - Use fully qualified imports
- `llm_client.py` exists in both `core/` and `healing/` - Use fully qualified imports
- `code_patcher.py` exists in both `core/` and `healing/` - Use fully qualified imports

## ✅ Safe Naming Conventions

### Recommended Patterns:
- Use descriptive names: `opinion_search_agent.py` ✅
- Use snake_case: `metrics_monitor.py` ✅
- Prefix with purpose: `test_*.py`, `run_*.py` ✅
- Use descriptive suffixes: `*_agent.py`, `*_monitor.py` ✅

### Avoid:
- Single letter names: `a.py`, `x.py` ❌
- Generic names: `utils.py`, `helpers.py` (if conflicts exist) ❌
- Names starting with numbers: `1_agent.py` ❌ (technically allowed but not recommended)
- Names with hyphens: `my-agent.py` ❌ (Python doesn't support)
- Names with spaces: `my agent.py` ❌

## 🔍 How to Check for Conflicts

Before creating a new file, check:

1. **Is it a Python keyword?**
   ```python
   import keyword
   print(keyword.kwlist)  # List of all Python keywords
   ```

2. **Does it conflict with standard library?**
   ```python
   import sys
   print(sys.modules.keys())  # Check if module already loaded
   ```

3. **Does it exist in project?**
   ```bash
   find . -name "your_filename.py" -type f
   ```

4. **Will imports conflict?**
   - Check if `from your_module import *` would shadow existing names
   - Use fully qualified imports if conflicts exist

## 📝 Best Practices

1. **Use descriptive, unique names**: `sentiment_analysis_agent.py` is better than `agent.py`
2. **Follow project conventions**: Use snake_case, descriptive prefixes
3. **Check before creating**: Search the codebase for existing files
4. **Use namespaces**: Organize files in appropriate directories (`agents/`, `scripts/`, `utils/`)
5. **Document conflicts**: If you must use a conflicting name, document why and how to import it

