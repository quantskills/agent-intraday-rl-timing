"""Smoke tests for the intraday RL lab skeleton.

Run: python scripts/test_intraday.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from baselines import BASELINES, run_policy
from intraday_env import EnvConfig, IntradayEnv, engineer_features
from train_eval import (
    SplitConfig,
    _train_tabular_q,
    chronological_split,
    evaluate,
    main,
    make_toy_minute,
    train_policy,
)


def test_features_are_leak_free_shape():
    bars = make_toy_minute(n_days=3, bars_per_day=50)
    feats = engineer_features(bars)
    for c in ("ret_1", "vwap_dist", "vol_z", "mins_to_close"):
        assert c in feats.columns
    assert feats["mins_to_close"].between(0, 1).all()


def test_baselines_roll():
    bars = make_toy_minute(n_days=2, bars_per_day=60)
    env = IntradayEnv(bars, EnvConfig())
    df = run_policy(env, BASELINES["intraday_momentum"])
    assert "nav" in df.columns and len(df) > 0


def test_walkforward_eval_runs():
    bars = make_toy_minute(n_days=20, bars_per_day=60)
    cfg = SplitConfig(algo="tabular")
    tr, val, te = chronological_split(bars, cfg)
    # chronological: train dates strictly before test dates
    assert tr["date"].max() < te["date"].min()
    policy = train_policy(tr, cfg, EnvConfig())
    metrics, rollouts = evaluate(te, policy, EnvConfig())
    assert "rl" in metrics and "twap" in metrics
    assert "rl" in rollouts and len(rollouts["rl"]) > 0


def make_predictable_minute(n_days=12, bars_per_day=50):
    rows = []
    for day in range(n_days):
        price = 100.0
        for bar in range(bars_per_day):
            price *= 1.001
            rows.append({
                "date": f"2024-02-{day + 1:02d}", "time": bar,
                "open": price, "high": price, "low": price,
                "close": price, "volume": 1000 + bar,
            })
    return pd.DataFrame(rows)


def test_tabular_encoder_has_multidimensional_state_space():
    bars = make_toy_minute(n_days=10, bars_per_day=60)
    policy = train_policy(bars, SplitConfig(algo="tabular"), EnvConfig())
    assert policy.encoder.n_states > 3
    assert policy.q.shape == (policy.encoder.n_states, 3)


def test_tabular_policy_trades_on_learnable_signal():
    bars = make_predictable_minute()
    policy = _train_tabular_q(bars, EnvConfig(), episodes=30)
    _, rollouts = evaluate(bars, policy, EnvConfig())
    assert (rollouts["rl"]["position"] != 0).any()


def test_out_dir_writes_declared_artifacts_and_explains_nan():
    with tempfile.TemporaryDirectory() as tmp:
        code = main(["--algo", "tabular", "--out-dir", tmp])
        assert code == 0
        report = Path(tmp) / "rl_report.md"
        assert report.stat().st_size > 0
        report_text = report.read_text(encoding="utf-8")
        assert "not estimable" in report_text
        for name in ("rl", "twap", "intraday_momentum", "intraday_reversal"):
            path = Path(tmp) / "episodes" / f"{name}.csv"
            assert path.stat().st_size > 0
            assert {"position", "reward", "nav"} <= set(pd.read_csv(path).columns)


if __name__ == "__main__":
    test_features_are_leak_free_shape()
    test_baselines_roll()
    test_walkforward_eval_runs()
    test_tabular_encoder_has_multidimensional_state_space()
    test_tabular_policy_trades_on_learnable_signal()
    test_out_dir_writes_declared_artifacts_and_explains_nan()
    print("all smoke tests passed")
