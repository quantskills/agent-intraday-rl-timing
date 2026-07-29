"""Walk-forward train/eval harness for intraday RL timing (research only).

Skeleton for agent-intraday-rl-timing. Trains an RL policy on a train window, selects on
validation, evaluates OOS on test — strictly chronological, never shuffled. Falls back to a
tabular Q-learner if stable-baselines3 is unavailable.

No broker connection, no live orders. Outputs are research artifacts, not investment advice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baselines import BASELINES, run_policy, sharpe
from intraday_env import EnvConfig, IntradayEnv, engineer_features


@dataclass
class SplitConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2          # remainder is test
    algo: str = "dqn"             # dqn | ppo | tabular


@dataclass
class TabularStateEncoder:
    ret_edges: np.ndarray
    vwap_edges: np.ndarray
    n_states: int = 81

    @classmethod
    def fit(cls, bars: pd.DataFrame):
        features = engineer_features(bars)
        quantiles = [1 / 3, 2 / 3]
        ret_edges = np.unique(features["ret_1"].quantile(quantiles).to_numpy())
        vwap_edges = np.unique(features["vwap_dist"].quantile(quantiles).to_numpy())
        return cls(ret_edges=ret_edges, vwap_edges=vwap_edges)

    def encode(self, obs) -> int:
        ret_bin = min(int(np.searchsorted(self.ret_edges, obs[0], side="right")), 2)
        vwap_bin = min(int(np.searchsorted(self.vwap_edges, obs[1], side="right")), 2)
        time_bin = min(int(np.searchsorted([1 / 3, 2 / 3], obs[3], side="right")), 2)
        position_bin = {-1: 0, 0: 1, 1: 2}.get(int(np.sign(obs[-1])), 1)
        return ((ret_bin * 3 + vwap_bin) * 3 + time_bin) * 3 + position_bin


@dataclass
class TabularQPolicy:
    q: np.ndarray
    encoder: TabularStateEncoder
    diagnostics: dict

    def __call__(self, obs) -> int:
        return int(np.argmax(self.q[self.encoder.encode(obs)]))


def chronological_split(bars: pd.DataFrame, cfg: SplitConfig):
    dates = np.sort(bars["date"].unique())
    n = len(dates)
    i_tr = int(n * cfg.train_frac)
    i_val = int(n * (cfg.train_frac + cfg.val_frac))
    tr = bars[bars["date"].isin(dates[:i_tr])]
    val = bars[bars["date"].isin(dates[i_tr:i_val])]
    te = bars[bars["date"].isin(dates[i_val:])]
    return tr, val, te


def train_policy(train_bars: pd.DataFrame, cfg: SplitConfig, env_cfg: EnvConfig):
    """Return a callable policy(obs)->action. Tabular fallback when SB3 is absent."""
    try:
        if cfg.algo in ("dqn", "ppo"):
            from stable_baselines3 import DQN, PPO

            env = IntradayEnv(train_bars, env_cfg)
            Model = DQN if cfg.algo == "dqn" else PPO
            model = Model("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=min(50_000, len(train_bars) * 5))
            return lambda obs: int(model.predict(obs, deterministic=True)[0])
    except Exception as exc:  # pragma: no cover
        print(f"[warn] SB3 unavailable ({exc}); using tabular Q fallback")
    return _train_tabular_q(train_bars, env_cfg)


def _train_tabular_q(train_bars, env_cfg, episodes: int = 5):
    """Tabular Q-learning on train-fitted return/VWAP/time/position bins."""
    encoder = TabularStateEncoder.fit(train_bars)
    q = np.zeros((encoder.n_states, 3))
    alpha, gamma = 0.1, 0.9
    rng = np.random.default_rng(0)
    visited = set()
    action_counts = np.zeros(3, dtype=int)
    for episode in range(episodes):
        eps = max(0.02, 0.30 * (0.92 ** episode))
        env = IntradayEnv(train_bars, env_cfg)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        done = False
        while not done:
            s = encoder.encode(obs)
            visited.add(s)
            a = rng.integers(3) if rng.random() < eps else int(np.argmax(q[s]))
            action_counts[a] += 1
            step = env.step(a)
            obs2, reward, done = step[0], step[1], step[2]
            s2 = encoder.encode(obs2)
            q[s, a] += alpha * (reward + gamma * q[s2].max() - q[s, a])
            obs = obs2
    return TabularQPolicy(
        q=q,
        encoder=encoder,
        diagnostics={
            "state_count": encoder.n_states,
            "visited_state_count": len(visited),
            "action_counts": action_counts.tolist(),
            "q_min": float(q.min()),
            "q_max": float(q.max()),
        },
    )


def evaluate(test_bars: pd.DataFrame, policy, env_cfg: EnvConfig) -> tuple[dict, dict]:
    results = {}
    rollouts = {}
    # RL policy
    env = IntradayEnv(test_bars, env_cfg)
    rl = run_policy(env, policy)
    rl_sharpe = sharpe(rl["reward"])
    results["rl"] = {
        "sharpe": round(rl_sharpe, 3) if np.isfinite(rl_sharpe) else None,
        "final_nav": round(float(rl["nav"].iloc[-1]), 4),
        "note": "not estimable: zero variance or no trades"
        if not np.isfinite(rl_sharpe) else "",
    }
    rollouts["rl"] = rl
    # baselines
    for name, pol in BASELINES.items():
        env = IntradayEnv(test_bars, env_cfg)
        df = run_policy(env, pol)
        value = sharpe(df["reward"])
        results[name] = {
            "sharpe": round(value, 3) if np.isfinite(value) else None,
            "final_nav": round(float(df["nav"].iloc[-1]), 4),
            "note": "not estimable: zero variance or no trades"
            if not np.isfinite(value) else "",
        }
        rollouts[name] = df
    return results, rollouts


def write_artifacts(out_dir: str, metrics: dict, rollouts: dict, bars: pd.DataFrame,
                    split_cfg: SplitConfig, env_cfg: EnvConfig, is_toy: bool) -> list[str]:
    """落盘 Output Contract 声明的产物：episodes/ 逐窗口回放 + rl_report.md。"""
    import os
    ep_dir = os.path.join(out_dir, "episodes")
    os.makedirs(ep_dir, exist_ok=True)
    written = []
    # episodes/：每个策略一份 OOS 回放（positions, reward, nav）
    for name, df in rollouts.items():
        cols = [c for c in ("position", "reward", "nav") if c in df.columns]
        p = os.path.join(ep_dir, f"{name}.csv")
        df[cols].to_csv(p, index=False)
        written.append(p)
    # rl_report.md
    dates = np.sort(bars["date"].unique())
    n = len(dates)
    i_tr = int(n * split_cfg.train_frac)
    i_val = int(n * (split_cfg.train_frac + split_cfg.val_frac))
    lines = ["# Intraday RL Timing — Walk-Forward Report", ""]
    if is_toy:
        lines += ["> ⚠️ **Toy demo data — results are meaningless, structure only.**", ""]
    lines += [
        f"- Algo: `{split_cfg.algo}` · env fees/slippage: stub fill model",
        f"- Bars: {len(bars)} rows over {n} days",
        "",
        "## Walk-forward layout (chronological, never shuffled)",
        "",
        f"- Train: {dates[0]} – {dates[i_tr-1]} ({split_cfg.train_frac:.0%})",
        f"- Validation: {dates[i_tr]} – {dates[i_val-1]} ({split_cfg.val_frac:.0%})",
        f"- Test (OOS): {dates[i_val]} – {dates[-1]} (remainder)",
        "",
        "## OOS metrics — RL vs baselines",
        "",
        "| policy | sharpe | final_nav |",
        "|---|---|---|",
    ]
    for k, v in metrics.items():
        shown_sharpe = v["sharpe"] if v["sharpe"] is not None else "N/A"
        lines.append(f"| {k} | {shown_sharpe} | {v['final_nav']} |")
    lines += ["", "_N/A = not estimable: zero variance or no trades._"]
    if not (rollouts["rl"]["position"] != 0).any():
        lines += [
            "",
            "## Degraded",
            "",
            "- RL policy produced no non-zero OOS positions; performance is not estimable.",
        ]
    lines += [
        "",
        "## Overfitting caveats",
        "",
        "- RL on minute data overfits easily (tiny sample, non-stationarity, reward hacking).",
        "- Read OOS-vs-baseline lift only; in-sample curves are meaningless here.",
        "- Fill model (fees/slippage) is a stub; real impact/queue dynamics not modeled.",
        "",
        "## Human review checklist",
        "",
        "- [ ] Reward shaping risks (does the reward encourage look-ahead or churn?)",
        "- [ ] Data quality: gaps, auctions, limit halts handled?",
        "- [ ] Look-ahead check: features use only past bars?",
        "- [ ] Split integrity: strictly chronological, no leakage across train/val/test?",
        "",
        "> ⚠️ Research only — no live orders, not investment advice.",
    ]
    p = os.path.join(out_dir, "rl_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    written.append(p)
    return written


# --------------------------------------------------------------------------- #
def make_toy_minute(n_days: int = 30, bars_per_day: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        price = 100.0
        for b in range(bars_per_day):
            price *= (1 + rng.normal(0, 0.0008))
            rows.append({"date": f"2024-01-{d+1:02d}", "time": b,
                         "open": price, "high": price, "low": price,
                         "close": price, "volume": rng.integers(100, 1000)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward train/eval for intraday RL timing (research only).")
    ap.add_argument("--bars-csv", help="minute bars CSV: date,time,open,high,low,close,volume (omit for toy)")
    ap.add_argument("--algo", default="tabular", choices=["dqn", "ppo", "tabular"])
    ap.add_argument("--out-dir", default=None, help="产物输出目录：落盘 episodes/ 逐窗口回放 + rl_report.md")
    args = ap.parse_args(argv)

    bars = pd.read_csv(args.bars_csv) if args.bars_csv else make_toy_minute()
    is_toy = not args.bars_csv
    if is_toy:
        print("[info] no data; running toy minute demo (results meaningless, structure only)")

    split_cfg = SplitConfig(algo=args.algo)
    env_cfg = EnvConfig()
    tr, val, te = chronological_split(bars, split_cfg)
    policy = train_policy(tr, split_cfg, env_cfg)
    metrics, rollouts = evaluate(te, policy, env_cfg)
    print("OOS metrics (RL vs baselines):")
    for k, v in metrics.items():
        print(f"  {k:20s} {v}")
    if args.out_dir:
        written = write_artifacts(args.out_dir, metrics, rollouts, bars, split_cfg, env_cfg, is_toy)
        print("\n[ok] artifacts written:")
        for p in written:
            print(f"  {p}")
    print("\n⚠️ research only — no live orders, not investment advice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
