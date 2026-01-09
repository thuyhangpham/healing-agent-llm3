"""
Constants for ETL Sentiment Analysis System

This module contains all system-wide constants including FSM states,
configuration values, and thresholds.
"""

# ============================================================================
# FSM States
# ============================================================================
STATE_IDLE = "IDLE"
STATE_SCRAPING = "SCRAPING"
STATE_ERROR_DETECTED = "ERROR_DETECTED"
STATE_HEALING = "HEALING"
STATE_ROLLBACK = "ROLLBACK"

# ============================================================================
# Configuration Constants
# ============================================================================
MAX_RETRIES = 3
HTML_MIN_LENGTH = 2000

# ============================================================================
# Agent Timeouts (seconds)
# ============================================================================
AGENT_TIMEOUT = 300  # 5 minutes for opinion search agent
HEALING_TIMEOUT = 180  # 3 minutes for healing agent

# ============================================================================
# Sleep Intervals (seconds)
# ============================================================================
IDLE_SLEEP = 5  # Sleep in IDLE state
LOOP_SLEEP = 1  # Sleep at end of FSM loop to prevent CPU spiking
CYCLE_SLEEP = 10  # Sleep between cycles (legacy, may be removed)

