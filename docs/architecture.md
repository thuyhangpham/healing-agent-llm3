# Technical Architecture: Self-Healing Multi-Agent Framework

## Executive Summary

This document outlines the technical architecture for a simplified self-healing multi-agent framework focused on core innovation: automated detection and repair of web scraping failures using local LLM integration. The architecture prioritizes minimal infrastructure complexity while achieving MTTR < 60 seconds and >80% success rate targets.

## System Overview

### Core Design Principles
- **Simplicity First**: Minimal external dependencies and straightforward implementation
- **Self-Healing Focus**: Primary emphasis on automated error detection and repair
- **Local Processing**: All AI operations performed locally using Ollama
- **File-Based Operations**: Simple file storage and logging for research validation
- **Hot-Reload Capability**: Runtime code modification without system shutdown

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Healing Framework                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agent     │  │   Agent     │  │   Agent     │         │
│  │  Framework  │  │ Orchestrator│  │  Healing    │         │
│  │             │  │             │  │   Agent     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │               │                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Worker     │  │  Worker     │  │  Worker     │         │
│  │  Agents     │  │  Agents     │  │  Agents     │         │
│  │ (Data Coll.)│  │ (Sentiment) │  │ (PDF Proc.) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────┐
                    │  Local LLM       │
                    │  (Ollama/Llama3) │
                    └─────────────────┘
```

## Core Components

### 1. Agent Framework Layer

#### Base Agent Class
```python
class BaseAgent:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = self._setup_logging()
        self.error_handler = ErrorHandler()
    
    def execute(self, task: dict) -> dict:
        try:
            return self._process_task(task)
        except Exception as e:
            error_context = self.error_handler.capture_error(e, self.name)
            self._report_error(error_context)
            raise
    
    def _process_task(self, task: dict) -> dict:
        raise NotImplementedError
```

#### Key Features
- **Unified Error Handling**: All agents use consistent error capture and reporting
- **Structured Logging**: JSON-formatted logs for research analysis
- **Configuration Management**: YAML-based configuration files
- **Health Monitoring**: Simple status reporting via CLI commands

### 2. Worker Agents

#### Data Collection Agents
```python
class AutonomousLawSearchAgent(BaseAgent):
    def _process_task(self, task: dict) -> dict:
        # Government portal scraping with error capture
        sources = self.config.get('law_sources', [])
        results = []
        
        for source in sources:
            try:
                data = self._scrape_source(source)
                results.append(data)
            except Exception as e:
                # Capture HTML snapshot for healing
                error_context = {
                    'error': str(e),
                    'source': source,
                    'html_snapshot': self._capture_html(),
                    'traceback': traceback.format_exc()
                }
                self._report_error(error_context)
        
        return {'results': results, 'status': 'completed'}

class AutonomousOpinionSearchAgent(BaseAgent):
    # Similar implementation for newspaper scraping
```

#### Error Capture Mechanism
- **HTML Snapshots**: Capture current page structure when scraping fails
- **Traceback Logging**: Full error context for LLM analysis
- **Source Metadata**: URL, selectors, timestamps for debugging
- **Retry Logic**: Configurable retry attempts with exponential backoff

### 3. Self-Healing Agent (Core Innovation)

#### Architecture
```python
class HealingAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__('HealingAgent', config)
        self.llm_client = OllamaClient(model='llama3')
        self.code_patcher = CodePatcher()
        self.validator = CodeValidator()
        self.metrics = HealingMetrics()
    
    def _process_task(self, task: dict) -> dict:
        error_context = task['error_context']
        start_time = time.time()
        
        try:
            # Step 1: Analyze error and generate fix
            fix_code = self._generate_fix(error_context)
            
            # Step 2: Validate generated code
            if not self.validator.validate_syntax(fix_code):
                raise ValueError("Generated code has syntax errors")
            
            # Step 3: Apply fix with backup
            self._apply_fix(error_context, fix_code)
            
            # Step 4: Hot-reload affected module
            self._reload_module(error_context['module_name'])
            
            # Step 5: Verify fix effectiveness
            if self._verify_fix(error_context):
                mttr = time.time() - start_time
                self.metrics.record_success(mttr)
                return {'status': 'healed', 'mttr': mttr}
            else:
                self.metrics.record_failure()
                return {'status': 'failed', 'reason': 'verification_failed'}
                
        except Exception as e:
            mttr = time.time() - start_time
            self.metrics.record_failure()
            self.logger.error(f"Healing failed: {e}")
            return {'status': 'failed', 'reason': str(e), 'mttr': mttr}
```

#### LLM Integration for Code Generation
```python
def _generate_fix(self, error_context: dict) -> str:
    prompt = self._build_healing_prompt(error_context)
    
    response = self.llm_client.generate(
        prompt=prompt,
        max_tokens=2000,
        temperature=0.1  # Low temperature for consistent code
    )
    
    # Extract code from LLM response
    fix_code = self._extract_code_from_response(response)
    return fix_code

def _build_healing_prompt(self, error_context: dict) -> str:
    return f"""
You are an expert Python developer specializing in web scraping automation.
Analyze this error and generate a fix for the scraping code.

ERROR DETAILS:
- Error Type: {error_context['error']}
- Traceback: {error_context['traceback']}
- Failed File: {error_context['file_path']}
- HTML Snapshot: {error_context['html_snapshot'][:1000]}...

CURRENT CODE:
{error_context['current_code']}

REQUIREMENTS:
1. Fix the specific error (CSS selector, parsing logic, etc.)
2. Maintain existing functionality
3. Add error handling for similar issues
4. Return only the fixed Python code
5. Use robust selectors that are less likely to break

Generate the complete fixed function or class:
"""
```

#### Code Validation and Hot-Reload
```python
class CodeValidator:
    def validate_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def validate_logic(self, code: str, test_cases: list) -> bool:
        # Basic logic validation with test cases
        try:
            exec(code, globals())
            # Run test cases
            return all(test_case() for test_case in test_cases)
        except Exception:
            return False

class CodePatcher:
    def apply_fix(self, file_path: str, fix_code: str) -> bool:
        try:
            # Create backup
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            
            # Apply fix
            with open(file_path, 'w') as f:
                f.write(fix_code)
            
            return True
        except Exception:
            # Restore from backup if available
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
            return False
```

### 4. Local LLM Integration

#### Ollama Client
```python
class OllamaClient:
    def __init__(self, model: str = 'llama3', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        payload = {
            'model': self.model,
            'prompt': prompt,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
```

#### Model Configuration
- **Model**: Llama 3 (8B recommended for balance of performance and resource usage)
- **Temperature**: 0.1 for consistent code generation
- **Max Tokens**: 2000 for complete function/class fixes
- **Timeout**: 30 seconds to meet MTTR requirements

### 5. Metrics and Research Validation

#### Healing Metrics Collection
```python
class HealingMetrics:
    def __init__(self, metrics_file: str = 'data/healing_metrics.json'):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
    
    def record_success(self, mttr: float):
        self.metrics['successful_repairs'] += 1
        self.metrics['total_mttr'] += mttr
        self.metrics['repair_times'].append(mttr)
        self._save_metrics()
    
    def record_failure(self):
        self.metrics['failed_repairs'] += 1
        self._save_metrics()
    
    def get_success_rate(self) -> float:
        total = self.metrics['successful_repairs'] + self.metrics['failed_repairs']
        return (self.metrics['successful_repairs'] / total * 100) if total > 0 else 0
    
    def get_average_mttr(self) -> float:
        return (self.metrics['total_mttr'] / self.metrics['successful_repairs'] 
                if self.metrics['successful_repairs'] > 0 else 0)
```

#### Research Data Export
```python
def export_research_data(self, output_dir: str = 'research_data'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Export metrics
    with open(f"{output_dir}/healing_metrics.json", 'w') as f:
        json.dump(self.metrics, f, indent=2)
    
    # Export detailed logs
    with open(f"{output_dir}/healing_logs.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'agent', 'error_type', 'mttr', 'success'])
        for log in self.metrics['detailed_logs']:
            writer.writerow([
                log['timestamp'], log['agent'], log['error_type'],
                log['mttr'], log['success']
            ])
```

## Data Flow Architecture

### Normal Operation Flow
```
1. Orchestrator schedules data collection tasks
2. Worker agents execute scraping with error capture
3. Results stored in JSON files with metadata
4. Status logged to structured log files
5. CLI commands provide system status
```

### Self-Healing Flow
```
1. Worker agent encounters error and captures context
2. Error event queued for Healing Agent
3. Healing Agent analyzes error with LLM
4. Code fix generated and validated
5. Backup created and fix applied
6. Module hot-reloaded without shutdown
7. Fix verification and metrics recorded
8. Process resumes with healed code
```

## File Structure

```
etl-sentiment/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── healing_agent.py
│   ├── law_search_agent.py
│   ├── opinion_search_agent.py
│   └── pdf_analysis_agent.py
├── core/
│   ├── __init__.py
│   ├── error_handler.py
│   ├── code_patcher.py
│   ├── llm_client.py
│   ├── metrics.py
│   └── validator.py
├── config/
│   ├── agents.yaml
│   ├── healing.yaml
│   └── sources.yaml
├── data/
│   ├── collected_data/
│   ├── healing_metrics.json
│   └── logs/
├── scripts/
│   ├── run_system.py
│   ├── status.py
│   ├── export_data.py
│   └── test_healing.py
├── tests/
│   ├── test_healing_agent.py
│   ├── test_code_patcher.py
│   └── test_integration.py
├── requirements.txt
├── README.md
└── main.py
```

## Performance Optimization

### MTTR Optimization Strategies
1. **Parallel Processing**: Error analysis and code generation in parallel
2. **Caching**: Cache common error patterns and fixes
3. **Pre-emptive Analysis**: Monitor for gradual selector degradation
4. **Fast Validation**: Quick syntax checks before comprehensive testing

### Success Rate Optimization
1. **Multiple Fix Attempts**: Generate 2-3 fix variations and test all
2. **Robust Selectors**: Prioritize CSS selectors over XPath
3. **Error Pattern Learning**: Store successful fixes for similar errors
4. **Fallback Mechanisms**: Alternative scraping methods when primary fails

## Deployment Architecture

### Local Deployment Requirements
- **Python 3.8+**: Core runtime environment
- **Ollama**: Local LLM server with Llama 3 model
- **4GB RAM Minimum**: For LLM operations
- **GPU Optional**: Recommended for faster LLM inference
- **10GB Storage**: For data, logs, and backups

### Configuration Management
```yaml
# config/healing.yaml
healing:
  max_attempts: 3
  timeout_seconds: 30
  backup_enabled: true
  validation_enabled: true
  
llm:
  model: "llama3"
  temperature: 0.1
  max_tokens: 2000
  base_url: "http://localhost:11434"

metrics:
  export_interval: 3600  # seconds
  retention_days: 30
```

## Security Considerations

### Code Safety
1. **Sandbox Validation**: Test generated code in isolated environment
2. **Backup System**: Automatic rollback capability for failed fixes
3. **Access Control**: Limit file system access for healing operations
4. **Audit Logging**: Complete audit trail of all healing activities

### Data Privacy
1. **Local Processing**: All data processed locally, no external API calls
2. **Configurable Retention**: User-controlled data retention policies
3. **Secure Storage**: Encrypted storage for sensitive collected data

## Testing Strategy

### Unit Testing
- Individual agent functionality
- Code generation and validation
- Error handling and recovery
- Metrics collection accuracy

### Integration Testing
- End-to-end healing workflow
- Multi-agent coordination
- LLM integration reliability
- Hot-reload functionality

### Validation Testing
- MTTR measurement accuracy
- Success rate calculation
- System stability under load
- Research data integrity

## Monitoring and Observability

### CLI Status Commands
```bash
# System status
python scripts/status.py --system

# Agent health
python scripts/status.py --agents

# Healing metrics
python scripts/status.py --healing

# Recent errors
python scripts/status.py --errors
```

### Log Structure
```json
{
  "timestamp": "2024-12-14T10:30:00Z",
  "agent": "AutonomousLawSearchAgent",
  "level": "ERROR",
  "event": "scraping_failure",
  "context": {
    "source": "https://example.gov/laws",
    "error": "CSS selector not found",
    "html_snapshot": "...",
    "traceback": "..."
  }
}
```

## Success Metrics and KPIs

### Primary KPIs
- **MTTR**: Target < 60 seconds (measured from error detection to successful repair)
- **Success Rate**: Target > 80% (successful repairs / total repair attempts)
- **System Uptime**: Target > 95% during continuous operation
- **Data Collection Rate**: Maintain > 90% of expected data volume

### Research Metrics
- **Error Pattern Distribution**: Categorize and analyze failure types
- **Fix Durability**: Measure how long fixes remain effective
- **LLM Performance**: Track code generation quality and speed
- **Resource Utilization**: Monitor CPU, memory, and storage usage

## Future Extensibility

### Potential Enhancements
1. **Multi-Model Support**: Integration with other local LLMs
2. **Distributed Healing**: Multiple healing agents for large-scale deployments
3. **Advanced Analytics**: Machine learning for error prediction
4. **Web Interface**: Optional simple web dashboard for monitoring

### Integration Points
1. **External Data Sources**: API integration for structured data sources
2. **Cloud Deployment**: Containerization for cloud deployment options
3. **Database Integration**: PostgreSQL/MongoDB for large-scale data storage
4. **Alerting Systems**: Integration with monitoring and alerting platforms

This architecture provides a solid foundation for implementing the core self-healing innovation while maintaining simplicity and focus on the research objectives. The modular design allows for incremental development and testing while ensuring the system can achieve the specified MTTR and success rate targets.