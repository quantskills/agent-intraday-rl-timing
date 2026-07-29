"""Baseline intraday policies — the bar every RL policy must beat.

Skeleton for agent-intraday-rl-timing. Research only, no live orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def twap_policy(obs: np.ndarray) -> int:
    """Time-weighted: always hold a flat target (placeholder = stay flat / accumulate evenly).

    For a pure timing benchmark, TWAP holds a constant target exposure regardless of state.
    """
    return 1  # constant long target; replace with schedule-aware accumulation if needed


def intraday_momentum_policy(obs: np.ndarray, ret_idx: int = 0) -> int:
    """Go with the last bar's sign: positive return -> long, negative -> short."""
    r = obs[ret_idx]
    return 1 if r > 0 else (2 if r < 0 else 0)


def intraday_reversal_policy(obs: np.ndarray, ret_idx: int = 0) -> int:
    """Fade the last bar: positive return -> short, negative -> long."""
    r = obs[ret_idx]
    return 2 if r > 0 else (1 if r < 0 else 0)


BASELINES = {
    "twap": twap_policy,
    "intraday_momentum": intraday_momentum_policy,
    "intraday_reversal": intraday_reversal_policy,
}


def run_policy(env, policy) -> pd.DataFrame:
    """Roll a stateless policy(obs)->action through an env; return per-step PnL frame."""
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    rows = []
    done = False
    while not done:
        action = policy(obs)
        step = env.step(action)
        if len(step) == 5:
            obs, reward, done, _, info = step
        else:
            obs, reward, done, info = step
        rows.append({"reward": reward, "position": info.get("position", 0)})
    df = pd.DataFrame(rows)
    df["nav"] = (1 + df["reward"]).cumprod()
    return df


def sharpe(rewards: pd.Series, periods_per_year: int = 252 * 240) -> float:
    r = rewards.dropna()
    return float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() else float("nan")
