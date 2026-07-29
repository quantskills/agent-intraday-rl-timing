# 🤖 Intraday RL Timing (Research Lab)

[简体中文](README.md) | **English**

> A Gym env on minute bars + baseline policies + leakage-aware walk-forward train/eval —
> the community's first RL and intraday-research asset.

## 📖 What This Is

The community has minute-data interfaces (`get_stock_min` / `get_future_min` /
`get_index_min`) but no intraday or RL capability. This agent fills that gap as a
**research-only lab**:

- `IntradayEnv` — Gymnasium-style env (state = intraday features + position + time-to-close;
  action = flat/long/short; reward = next-bar PnL net of fees/slippage).
- **Baselines** — TWAP / intraday momentum / intraday reversal, the bar every RL policy must beat.
- **Train/eval** — a thin `stable-baselines3` (DQN/PPO) wrapper with a tabular Q-learning
  fallback; strictly chronological walk-forward.

> ⚠️ **Research only.** No broker connection, **no live orders**, consistent with the
> QUANTSKILLS no-automated-order rule. Outputs are research artifacts, not investment advice.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python scripts/train_eval.py --algo tabular   # toy minute demo (structure only)
python scripts/test_intraday.py
```

## ⚠️ Disclaimer

This repository provides an intraday RL research environment and code skeleton only. It does
not connect to any trading account, places no orders, verifies no performance claims, and
does not constitute investment advice. RL on minute data overfits easily — always read OOS
vs baselines. Community Project — not officially validated or endorsed until maintainer review.

## 📜 License

GPL-3.0. See [LICENSE](LICENSE).

## 🐼 PandaAI / QUANTSKILLS Community

<div align="center">
  <img src="https://raw.githubusercontent.com/quantskills/.github/main/profile/assets/pandaai-community-qr.jpg" alt="PandaAI community QR code" width="220">
  <br>
  <sub>Scan the QR code to join the PandaAI community for QUANTSKILLS skills, agent workflows, and quantitative research practice.</sub>
</div>
