"""Gymnasium-style intraday trading environment (research only).

Skeleton for agent-intraday-rl-timing. State = intraday features + position +
time-to-close; action = discrete {flat, long, short}; reward = next-bar PnL net of
fees/slippage. No broker connection, no live orders.

Falls back to a minimal env interface if gymnasium is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _BASE = object
    _HAS_GYM = False


@dataclass
class EnvConfig:
    fee_bp: float = 1.0           # per-trade fee, bp of notional
    slippage_bp: float = 1.0      # per-trade slippage, bp
    max_position: int = 1         # {-1, 0, +1}
    feature_cols: tuple[str, ...] = ("ret_1", "vwap_dist", "vol_z", "mins_to_close")


# --------------------------------------------------------------------------- #
# Feature engineering (leak-free: only past/current bar info)
# --------------------------------------------------------------------------- #
def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: date, time, open, high, low, close, volume (one trading session per group)."""
    df = bars.copy()
    df["ret_1"] = df.groupby("date")["close"].pct_change().fillna(0.0)
    vwap = (df["close"] * df["volume"]).groupby(df["date"]).cumsum() / \
           df["volume"].groupby(df["date"]).cumsum().replace(0, np.nan)
    df["vwap_dist"] = (df["close"] / vwap - 1).fillna(0.0)
    vol = df.groupby("date")["volume"]
    df["vol_z"] = ((df["volume"] - vol.transform("mean")) / (vol.transform("std") + 1e-9)).fillna(0.0)
    n = df.groupby("date")["close"].transform("count")
    df["mins_to_close"] = (n - df.groupby("date").cumcount()) / n
    return df


class IntradayEnv(_BASE):
    """Single-asset intraday timing env. Reward is next-bar PnL net of trading cost."""

    def __init__(self, bars: pd.DataFrame, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.df = engineer_features(bars).reset_index(drop=True)
        self.n = len(self.df)
        self._i = 0
        self.position = 0
        if _HAS_GYM:
            self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.cfg.feature_cols) + 1,), dtype=np.float32)

    # --- core API -------------------------------------------------------- #
    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self._i]
        feats = [float(row[c]) for c in self.cfg.feature_cols]
        return np.array(feats + [float(self.position)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._i = 0
        self.position = 0
        obs = self._obs()
        return (obs, {}) if _HAS_GYM else obs

    def step(self, action: int):
        target = {0: 0, 1: 1, 2: -1}[int(action)]
        target = int(np.clip(target, -self.cfg.max_position, self.cfg.max_position))
        trade = abs(target - self.position)
        cost = trade * (self.cfg.fee_bp + self.cfg.slippage_bp) / 1e4

        # reward = position carried into next bar * next-bar return - trade cost
        nxt = self._i + 1
        nxt_ret = float(self.df.iloc[nxt]["ret_1"]) if nxt < self.n else 0.0
        reward = target * nxt_ret - cost

        self.position = target
        self._i += 1
        done = self._i >= self.n - 1
        obs = self._obs() if not done else np.zeros_like(self._obs())
        if _HAS_GYM:
            return obs, reward, done, False, {"position": self.position}
        return obs, reward, done, {"position": self.position}
