# 🤖 Intraday RL Timing · 日内强化学习研究实验台

**简体中文** | [English](README.en.md)

> 分钟数据建 Gym 环境 + 基线策略 + 防泄漏 walk-forward 训练评估 —— 社区第一个 RL 与日内研究资产。

![type](https://img.shields.io/badge/type-agent-purple)
![license](https://img.shields.io/badge/license-GPLv3-blue)
![validation](https://img.shields.io/badge/validation-Listed-lightgrey)
![scope](https://img.shields.io/badge/scope-research--only-red)

---

## 📖 这是什么

社区有完整的分钟数据接口（`get_stock_min` / `get_future_min` / `get_index_min`），
但没有任何日内或强化学习能力。本 agent 填补这块，定位为**纯研究实验台**：

- `IntradayEnv`：Gymnasium 式日内交易环境（状态=日内特征+持仓+距收盘；动作=flat/long/short；
  奖励=下一 bar 盈亏扣手续费/滑点）。
- **基线策略**：TWAP / 日内动量 / 日内反转 —— RL 必须打过的标尺。
- **训练评估**：`stable-baselines3`（DQN/PPO）封装，缺失时回退内置 tabular Q-learning；
  严格时序 walk-forward。

> ⚠️ **仅研究用途**。本 agent 不连接券商、**不下单**，符合 QUANTSKILLS「不自动下单」规则。
> 所有输出是研究产物，不构成投资建议。

## 🚀 快速开始

```bash
cp -r agent-intraday-rl-timing ~/.claude/agents/agent-intraday-rl-timing
pip install -r requirements.txt
# 玩具分钟数据自检（结果无意义，仅验证结构/可跑；无需凭证与 SB3）
python scripts/train_eval.py --algo tabular --out-dir /tmp/intraday-rl-toy
python scripts/test_intraday.py
```

`--out-dir` 会实际生成 `episodes/{rl,twap,intraday_momentum,intraday_reversal}.csv`
和 `rl_report.md`；stdout 不是产物替代品。toy 数据只验证结构，Sharpe 无法估计时报告显示
`N/A (not estimable)`。

Pandadata 分钟数据调用（参数名是 `frequency`，不是 `period`）：

```python
panda_data.get_stock_min(
    symbol="600519.SH",
    start_date="20260401",
    end_date="20260724",
    frequency="1m",
)
```

导出为 CSV 后运行：

```bash
python scripts/train_eval.py --bars-csv rb_minute.csv --algo tabular \
    --out-dir /tmp/intraday-rl-real
```

```text
触发示例 prompt：
「在螺纹钢分钟数据上训一个日内择时 agent，对比 TWAP 基线的样本外夏普」
「跑 walk-forward 评估，告诉我 RL 是否真的打过日内动量基线」
```

## 📦 目录结构

```text
agent-intraday-rl-timing/
├── AGENTS.md
├── requirements.txt
├── references/
│   ├── env-and-reward.md        # 环境/奖励/成交模型 + walk-forward + 防过拟合
│   └── source_boundary.md
├── scripts/
│   ├── intraday_env.py          # Gym 式环境
│   ├── baselines.py             # TWAP / 动量 / 反转 基线
│   ├── train_eval.py            # 时序切分 + 训练 + OOS 评估
│   ├── test_intraday.py
│   └── login_pandadata.py
└── agents/
    └── openai.yaml
```

## 📐 核心约束

| 约束 | 说明 |
| --- | --- |
| 🚫 不实盘 | 不连券商、不下单，纯历史数据研究 |
| 📉 极易过拟合 | 分钟样本小、非平稳，必须看 OOS vs 基线，单次漂亮曲线不可信 |
| ⏱️ 时序切分 | train/val/test 按日期切，绝不打乱 bar |
| 🧱 成交是桩 | 手续费/滑点为简化桩，未建模真实冲击与容量 |
| 🚫 只述不荐 | 不构成任何投资建议 |

## ✅ 真实数据测试结论（2026-07-27）

- **smoke test**：`python scripts/test_intraday.py` 全部通过（特征无泄漏、baseline 策略滚动、walk-forward 评估）。
- **数据依赖**：分钟线 `get_stock_min` / `get_future_min`（支持 1m/5m/15m/60m），此前已验证真实可得。
- **tabular fallback**：状态由训练集拟合的 `ret_1 × vwap_dist × 时段 × 当前持仓`
  共 81 个离散状态组成；分箱边界不读取验证/测试集。策略仍可能理性选择空仓，报告会显式诊断。
- 结论：代码与数据源可用于研究；不保证 RL 在任意真实窗口中胜过基线。

## ⚠️ 免责声明

本仓库仅提供日内强化学习的研究环境与代码骨架，不连接任何交易账户、不下单、不验证任何收益声明、
不构成任何投资建议。默认 Community Project；请结合引用的数据与本地审核要求复核输出。

## 📜 License

GPL-3.0-only，详见 [LICENSE](LICENSE)。

## 🐼 PandaAI / QUANTSKILLS 社群

<div align="center">
  <img src="https://raw.githubusercontent.com/quantskills/.github/main/profile/assets/pandaai-community-qr.jpg" alt="PandaAI 社群二维码" width="220">
  <br>
  <sub>扫码加入 PandaAI 社群，交流 QUANTSKILLS 技能、Agent 工作流与量化研究实践。</sub>
</div>
