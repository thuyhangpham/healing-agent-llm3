PROJECT MASTER PLAN: SELF-HEALING LEGAL SENTIMENT MONITORING SYSTEM

Technical Direction & Implementation DocumentProject: Enhancing Reliability in Legal Sentiment Monitoring: A Self-Healing Multi-Agent FrameworkVersion: 1.0 - Draft for Implementation

1. PROJECT OVERVIEW

1.1. Context & Problem Statement

Current Policy Sentiment Analysis systems typically operate on a "run once" (batch processing) mechanism and are highly unstable when deployed in real-time scenarios.
- Core Problem: Data sources (online newspapers, government portals) frequently change their interface structures (HTML/CSS Drift).
- Consequences: Traditional Scrapers/Crawlers crash, causing data interruptions and requiring significant manual maintenance costs.
- Research Gap: Lack of empirical studies on the robustness and self-healing capabilities of AI systems for sentiment monitoring in volatile data environments.
1.2. Objectives
Build a Self-Adaptive Multi-Agent System capable of:
- Continuous Monitoring: Collect data 24/7 from multiple sources.
- Self-Healing: Automatically detect web structure errors, use LLMs to rewrite data collection code, and automatically redeploy (Hot-reload) without human intervention.
- Empirical Validation: Achieve an MTTR (Mean Time To Repair) of under 60 seconds and a self-repair success rate > 80%.

2. SYSTEM ARCHITECTURE
The system is built on the LangGraph framework, extended with a Self-Healing Layer.
2.1. Core Components
- Orchestrator (Manager Agent):
    + Coordinates the main workflow. 
    + Manages the Global State.
- Adaptive Worker Agents (Execution Agents):These Agents perform specific tasks but are wrapped in a try...except error-catching mechanism.
    + AutonomousLawSearchAgent: Searches for new legal texts from government portals.
    + AutonomousOpinionSearchAgent: Collects articles from online newspapers (Most prone to errors - Main Focus).       
    + AutonomousPdfAnalysisAgent: Extracts content from legal texts (PDF/Doc).
    + SentimentAnalysisAgent: Analyzes sentiment using LLMs.
- The Healer (Healing Agent - Virtual Doctor):
    + Role: Virtual Python/DevOps Expert.
    + Task: Receive error info -> Analyze -> Call LLM to fix code -> Test -> Hot-patch.
    + Tool: Patcher Module.
- Infrastructure:
    + Local LLM: Ollama (Llama 3) running locally (Zero-cost, High privacy).
    + Code Patcher: System file processing module (AST, Importlib).

2.2. Logical Flow

Code graph TD
    Start([Start]) --> Manager[Manager Agent]
    Manager -->|Dispatch Task| Worker[OpinionSearchAgent]
    
    subgraph "Execution Layer"
        Worker -->|Run Tool| Tool{Tool Execution}
        Tool -->|Success| Result[Update State]
        Tool -->|Exception (e.g. NoSuchElement)| Error[Catch Error]
    end
    
    Error -->|Route to Healer| Healer[Healing Agent]
    
    subgraph "Self-Healing Layer"
        Healer -->|1. Read Broken Code| Reader[Code Reader]
        Healer -->|2. Send Context (Code + Traceback)| LLM[Ollama (Llama 3)]
        LLM -->|3. Generate Fix| Patcher[Patcher Module]
        Patcher -->|4. Syntax Check & Sandbox| Test{Valid?}
        Test -- No --> LLM
        Test -- Yes --> HotReload[Hot Reload Module]
    end
    
    HotReload -->|Retry Task| Worker
    Result --> Manager
    Manager --> End([End Workflow])


3. SELF-HEALING MECHANISM (THE CORE INNOVATION)
This is the "heart" of the project.
3.1. Recovery Workflow
- Detection: Worker Agent catches an Exception (e.g., NoSuchElementException when a newspaper changes CSS classes).
- Contextualization: Worker packages: Traceback + Path to failed file + HTML Snapshot (if available) into the State.
- Diagnosis & Repair:
    + HealingAgent reads the current code file.
    + Sends Prompt to Llama 3: "This code failed with error X at line Y. Rewrite the logic (e.g., update the new selector) to make it run. Return only Python code."
- Verification:Patcher uses ast.parse() to check Python syntax.(Advanced) Test run the new code in an isolated environment.
- Hot-Patching:Create a backup of the old file (.bak).
Overwrite the .py file with the new code.
Use importlib.reload() to reload the module into memory without shutting down the main program.


4. EXPERIMENTAL DESIGN & EVALUATION
To write a scientific paper (Gap Research), Chaos Engineering (Intentional Fault Injection) must be performed.
4.1. Environment 
- Local PC: GPU supporting Ollama.
- Target: Simulated 3 online newspaper pages or real (can ask me to have the link of real news).
4.2. Test Scenarios 
- The Layout Shift:Action: Manually edit the tool file, changing the correct selector (.title) to a wrong one (.wrong-title).
- Expectation: System auto-detects -> Calls AI to fix back to .title (or equivalent selector) -> Continues running.
- The Logic Bug:
Action: Add code causing a logic error (e.g., division by zero or wrong date format).
Expectation: AI detects and rewrites logic for safe handling (try-catch or data validation).
4.3. MetricsMTTR (Mean Time To Repair): Average time for the system to fix itself.
Success Rate: Ratio of successful fixes / Total errors.Downtime: Service interruption time.

5. TECH STACK
Category Technology Notes 
- Framewor: kLangGraph / LangChain: Workflow and State Management.
- AI Model: Ollama (Llama 3): Runs locally, high privacy, zero cost.
- Scraping: Selenium / BeautifulSoup4: Web data collection.
- Core Ops: Python, Importlib: Code processing and hot-reload.
- Data: Pandas, JSON: Data storage and processing.