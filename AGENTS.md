---
name: agent-intraday-rl-timing
description: "A research-only reinforcement-learning lab for intraday timing on minute bars. Builds a Gymnasium-style trading environment from Pandadata minute data, ships baseline policies (TWAP, intraday momentum, intraday reversal), and a leakage-aware walk-forward train/eval harness for offline/online RL (DQN-style). Use when a research agent needs to prototype and benchmark intraday-timing policies against baselines with realistic fees/slippage — never for live order placement."
quantSkills:
  organization: QuantSkills
  organization_url: https://github.com/quantskills
  repository: agent-intraday-rl-timing
  repository_url: https://github.com/quantskills/agent-intraday-rl-timing
  project_type: agent
  collection: intraday-research
  license: GPL-3.0-only
  category: research-agent
  tags: [reinforcement-learning, intraday, minute-bars, gymnasium, walk-forward, research-only]
  platforms: [claude-code, codex, openclaw]
  language: zh-en
  status: draft
  validation_level: listed
  maintainer_type: community
  requires: []
  summary_zh: 纯研究的日内强化学习实验台：分钟数据建 Gym 环境 + 基线策略(TWAP/动量/反转) + 防泄漏 walk-forward 训练评估，绝不实盘下单。
  summary_en: "Research-only RL lab for intraday timing on minute bars: Gym env, baseline policies, leakage-aware walk-forward train/eval. No live orders."
---

# Intraday RL Timing (Research Lab)

Use this agent to **prototype and benchmark** intraday-timing policies on minute bars. It is
the community's first RL and intraday-research asset.

> ⚠️ **Research only.** This agent builds environments, trains policies, and evaluates them
> on historical minute data. It does **not** connect to any broker and **does not place
> orders**, consistent with the QUANTSKILLS organization rule against automated order
> placement. Outputs are research artifacts, not investment advice.

## Operating Boundary

- Read public or user-provided minute data only (`panda_data.get_*_min`, or local exports).
- Never connect to a trading account; never place, route, or simulate live orders to a broker.
- Produce reviewable artifacts (configs, metrics, plots, a report) and cite the data window used.
- All evaluation is strict walk-forward; in-sample metrics must be labeled as such.

## Components

| Component | What it is |
| --- | --- |
| `IntradayEnv` | Gymnasium-style env: state = intraday features + position + time-to-close; action = {flat, long, short} or target weight; reward = next-bar PnL net of fees/slippage |
| Baseline policies | `twap`, `intraday_momentum`, `intraday_reversal` — the bar every RL policy must beat |
| Trainer | thin wrapper around `stable-baselines3` (DQN/PPO); falls back to an inspectable 81-state tabular Q policy fitted on train-only return/VWAP/time/position bins |
| Evaluator | walk-forward OOS rollout: Sharpe, hit-rate, turnover, vs-baseline lift |

## Core Workflow

1. **Ingest**: load minute bars for a symbol/variety and a date range; engineer intraday
   features (returns, VWAP distance, volume profile, minutes-to-close).
2. **Split**: chronological train / validation / test by date — never shuffle bars.
3. **Baselines**: run TWAP + momentum + reversal to establish the benchmark.
4. **Train**: fit the RL policy on the train window only; select on validation.
5. **Evaluate**: walk-forward OOS rollout with fees/slippage; compare to baselines.
6. **Report**: env config, reward spec, train/val/test windows, metrics table, caveats.

## Output Contract

Produce:

- `episodes/` — per-window OOS rollouts (positions, PnL)
- `rl_report.md` — env/reward spec, walk-forward layout, metrics vs baselines, overfitting caveats
- a list of human review items (reward shaping risks, data quality, look-ahead checks)

## Data Sources

- `panda_data.get_stock_min`, `panda_data.get_future_min`, `panda_data.get_index_min`（分钟频率参数使用 `frequency="1m"`，不是 `period`）
- or user-provided minute exports

## Limitations & Risk Boundary

- **RL on financial minute data overfits easily**: tiny effective sample, non-stationarity,
  and reward hacking make impressive in-sample curves meaningless. Always read OOS vs baseline.
- The env's fill model (fees/slippage) is a stub; real intraday impact and queue dynamics are
  not modeled. Capacity is unbounded by assumption.
- Minute data has gaps, auctions, and limit halts; the env must handle non-tradable bars.
- **No live trading.** This agent is a research lab and does not constitute investment advice.
- Community Project; validate outputs against the cited data and local review requirements.

## ✅ Quality Bar

Before delivering artifacts (degrade & disclose rather than pass silently):

- **Traceable**: every key figure maps to a specific Pandadata interface + data date; missing data goes to `degraded[]` — never fabricate or pass approximations off as real values.
- **Transparent degradation**: when any source is empty/limited, the report states it and lowers confidence.
- **Consistent conventions**: units, frequency, and benchmark conventions are stated explicitly.
- **Research only**: artifacts are for research/education, not investment advice, with no return promises.
- All evaluation strictly walk-forward; in-sample metrics labeled as such; no live orders, ever.

## References

- `references/env-and-reward.md` — env spec, reward/fill model, walk-forward and anti-overfit rules.
- `references/source_boundary.md` — allowed data sources.
