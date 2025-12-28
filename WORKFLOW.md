# ETL Sentiment System - Complete Workflow Documentation

## System Overview

The ETL Sentiment system is a multi-agent architecture designed for:
- **Legal Document Collection** from VBPL (Vietnamese legal portal)
- **Public Opinion Mining** from VnExpress (Vietnamese news portal)
- **PDF Document Processing** and analysis
- **Sentiment Analysis** of collected content
- **Self-Healing** for automated error detection and recovery

```

## Production Workflow

### 1. System Initialization
```bash
# Start the complete ETL system
python scripts/run_production.py --duration 60 --agents law_search_agent,opinion_search_agent,sentiment_analysis_agent,healing_agent
```

### 2. Agent Initialization Sequence
1. **Agent Registry** loads all agent configurations from `config/agents.yaml`
2. **Individual Agents** initialize in parallel:
   - Law Search Agent → VBPL web scraping
   - Opinion Search Agent → VnExpress web scraping  
   - PDF Analysis Agent → Document processing
   - Sentiment Analysis Agent → Text sentiment analysis
   - Healing Agent → Error monitoring and recovery

### 3. Production Data Collection Loop
Each agent runs continuously for the specified duration:
- **Law Search**: Crawls VBPL for legal documents
- **Opinion Search**: Mines VnExpress for public opinions
- **PDF Analysis**: Processes uploaded PDF documents
- **Sentiment Analysis**: Analyzes collected text content
- **Healing Agent**: Monitors system health and handles errors


### Agent Configuration (`config/agents.yaml`)
```yaml
agents:
  law_search_agent:
    id: "law_search_agent"
    enabled: true
    type: "law_search"
    sources:
      - name: "vbpl_khcn"
        url: "https://vbpl.vn/"
        search_queries:
          - "Luật Công nghệ cao"
          - "Luật Khoa học và công nghệ"
          - "Luật An ninh mạng"
          - "Luật Giao dịch điện tử"
          - "Sở hữu trí tuệ"
          - "Bảo vệ dữ liệu cá nhân"
          - "Chuyển đổi số"
          - "Trí tuệ nhân tạo"
          - "An ninh mạng"
    
  opinion_search_agent:
    id: "opinion_search_agent"
    enabled: true
    type: "opinion_search"
    sources:
      - name: "vnexpress_sohoa"
        url: "https://vnexpress.net/so-hoa"
        search_topics:
          - "công nghệ việt nam"
          - "chuyển đổi số"
          - "an ninh mạng"
          - "công nghệ"
          - "số hóa"
          - "bán dẫn"
          - "thông minh"
    
  sentiment_analysis_agent:
    id: "sentiment_analysis_agent"
    enabled: true
    type: "sentiment_analysis"
    model: "llama3:latest"
    confidence_threshold: 0.7
    sentiment_labels: ["positive", "negative", "neutral"]
    
  healing_agent:
    id: "healing_agent"
    enabled: true
    type: "healing"
    auto_heal_enabled: true
    max_healing_attempts: 3
    mttr_target_seconds: 60
    success_rate_target: 0.8
```

## Data Storage Structure

### Directory Layout
```
data/
├── experiments/           # Chaos testing results
├── production/           # Production run results
│   ├── production_results_*.json
│   └── debug_log.txt
├── logs/               # System logs
├── metrics/             # Performance metrics
├── pdfs/               # PDF processing
│   ├── raw/           # Original PDFs
│   └── processed/     # Analyzed PDFs
└── snapshots/           # System state snapshots
```

###  Production Commands

### Basic Production Run
```bash
# Run all agents for 1 hour
python scripts/run_production.py --duration 60 --agents law_search_agent,opinion_search_agent,sentiment_analysis_agent,healing_agent

# Run specific agents for 30 minutes
python scripts/run_production.py --duration 30 --agents law_search_agent,opinion_search_agent

# Run with custom configuration
python scripts/run_production.py --duration 120 --agents law_search_agent --vbpl-url "https://vbpl.vn/"
```

### Development & Testing Commands
```bash
# Test web scraping functionality
python scripts/test_web_scraping.py

# Test individual agent
python scripts/run_production.py --duration 5 --agents law_search_agent

# Monitor system status
python scripts/status.py

# Export data for analysis
python scripts/export_data.py
```

### Common Issues & Solutions

#### 1. Network Connectivity
**Problem**: Cannot connect to VBPL/VnExpress
**Solution**: 
```bash
# Test connectivity
ping vbpl.vn
curl -I https://vnexpress.net/

# Check proxy settings
python scripts/run_production.py --duration 1 --agents law_search_agent --debug
```

#### 2. Parsing Errors
**Problem**: HTML structure changes breaking scrapers
**Solution**: 
- Update CSS selectors in `config/agents.yaml`
- Enable debug logging to identify parsing issues
- Use fallback parsing methods

#### 3. Performance Issues
**Problem**: Slow processing or high memory usage
**Solution**: 
- Reduce batch sizes in agent configurations
- Increase timeout values
- Monitor system resources

#### 4. Data Quality Issues
**Problem**: Incomplete or incorrect data extraction
**Solution**: 
- Validate extracted data quality
- Implement data cleaning pipelines
- Add manual review processes

## 📚 Development Guidelines

### Code Structure
```
agents/
├── law_search_agent.py      # VBPL web scraping
├── opinion_search_agent.py    # VnExpress mining
├── pdf_analysis_agent.py      # Document processing
├── sentiment_analysis_agent.py # Text analysis
├── healing_agent.py         # Error recovery
└── base_agent.py           # Common agent base

scripts/
├── run_production.py       # Main production runner
├── test_web_scraping.py   # Testing utilities
├── status.py              # Monitoring dashboard
└── export_data.py          # Data export utilities

core/
├── llm_client.py          # LLM integration
├── error_detector.py       # Error analysis
├── code_patcher.py         # Code modification
└── healing_metrics.py       # Performance tracking

utils/
├── agent_registry.py       # Agent management
├── global_state.py        # State management
├── logger.py              # Logging system
└── config.py              # Configuration management
```

### Best Practices
1. **Configuration Management**: Use YAML files for all agent settings
2. **Error Handling**: Implement try-catch blocks with proper logging
3. **Resource Management**: Use context managers for file operations
4. **Async/Await**: Properly handle asynchronous operations
5. **Testing**: Write unit tests for all agent functionality
6. **Documentation**: Maintain comprehensive documentation for all components

## Production Deployment

### Environment Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp config/agents.yaml.example config/agents.yaml

# 3. Set up directories
mkdir -p data/{production,logs,metrics,pdfs/{raw,processed}}

# 4. Start production system
python scripts/run_production.py --duration 480 --agents all
```

### Monitoring & Maintenance
```bash
# Daily health check
python scripts/status.py --daily

# Weekly performance report
python scripts/export_data.py --period week

# Monthly system cleanup
python scripts/run_production.py --duration 60 --agents healing_agent --cleanup
```

## System Evolution

### Version History
- **v1.0**: Basic agent framework
- **v2.0**: Added self-healing capabilities
- **v3.0**: Enhanced error detection and LLM integration
- **v4.0**: Production-ready multi-agent system with real web scraping

### Future Roadmap
- **Enhanced LLM Integration**: GPT-4 and Claude API support
- **Distributed Processing**: Multi-node data processing
- **Real-time Dashboard**: Web-based monitoring interface
- **Advanced Analytics**: Machine learning for sentiment prediction
- **API Integration**: RESTful APIs for external data access

---

*Last Updated: 2025-12-15*
*System Version: v4.0*
*Status: Production Ready*