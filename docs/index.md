# Documentation Index

## Root Documents

### [Architecture](./architecture.md)

Technical architecture document outlining the self-healing multi-agent framework design. Includes system overview, core components (Agent Framework, Worker Agents, Self-Healing Agent), local LLM integration, metrics collection, data flow architecture, file structure, performance optimization strategies, deployment architecture, security considerations, testing strategy, monitoring, and success metrics.

### [Product Requirements Document](./prd.md)

Comprehensive PRD for the self-healing legal sentiment monitoring system. Defines goals, functional and non-functional requirements, user interface design goals, technical assumptions, epic breakdown, and detailed story requirements. Includes acceptance criteria for all three epics: Core Multi-Agent System, Self-Healing Innovation, and Research Validation.

## Ideas

Documents within the `ideas/` directory:

### [Multi-Agent Sentiment Idea](./ideas/multi-agent-sentiment-idea.md)

Project master plan and technical direction document for the self-healing legal sentiment monitoring system. Provides project overview, system architecture using LangGraph framework, detailed self-healing mechanism workflow, experimental design for chaos engineering validation, and technology stack specifications.

## Stories

Documents within the `stories/` directory:

### [Story 1.1: Project Setup](./stories/1.1.project-setup.md)

Story for establishing the development environment and project infrastructure. Includes acceptance criteria for Python project structure, LangGraph/LangChain dependencies, Selenium/BeautifulSoup4 setup, Ollama with Llama 3 configuration, development tools (pytest, black, flake8), Git repository initialization, logging configuration, and environment configuration system. Contains detailed dev notes on project structure, dependencies, and testing requirements.

### [Story 1.2: Core Agent Framework and Orchestrator](./stories/1.2.core-agent-framework.md)

Story for implementing the core agent orchestration framework using LangGraph. Defines acceptance criteria for LangGraph workflow engine, Base Agent class with common functionality, Orchestrator (Manager Agent) implementation, global state management, agent registration and discovery, inter-agent communication patterns, and end-to-end workflow demonstration. Includes implementation notes and testing requirements.

### [Story 1.3: Self-Healing Agent](./stories/1.3.self-healing-agent.md)

Comprehensive story for implementing the self-healing agent system. Includes goals, background context, functional and non-functional requirements, user interface design goals, technical assumptions, and detailed task breakdown. Covers healing agent core implementation, LLM integration, error detection, hot-reload capabilities, metrics collection, configuration, testing, and documentation. All tasks marked as completed with comprehensive file list and implementation notes.

