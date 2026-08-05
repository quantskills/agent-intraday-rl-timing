---
name: agent-intraday-rl-timing
description: "Research-only intraday reinforcement-learning environment and walk-forward evaluation workflow. Use for offline minute-bar experiments and baseline comparisons; never for live order placement."
license: GPL-3.0-only
category: research-agent
metadata:
  repository: agent-intraday-rl-timing
  project_type: agent
  runtime_entrypoint: AGENTS.md
---

# Intraday RL Timing

This agent's detailed operating instructions live in [AGENTS.md](AGENTS.md). Load that
file after this metadata when the agent is invoked. It defines the data boundary,
walk-forward workflow, output contract, and research-only safety limits.

The agent does not connect to brokers or place live orders. Keep in-sample and
out-of-sample results clearly labeled, cite data dates, and disclose missing inputs.
