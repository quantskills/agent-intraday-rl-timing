# Env & Reward Spec — Intraday RL Timing

环境、奖励/成交模型、walk-forward 与防过拟合规则。供 Agent 对齐 `scripts/`。

> 全程研究用途。本 agent 不连接券商、不下单。

## 环境（IntradayEnv）

- **状态**：日内特征（上一 bar 收益、VWAP 距离、成交量 z 分、距收盘比例）+ 当前持仓。
  所有特征只用「当前及过去 bar」信息，禁止用到未来 bar。
- **动作**：离散 `{0=flat, 1=long, 2=short}`，目标仓位裁剪到 `[-max_position, max_position]`。
- **奖励**：`target_position × next_bar_return − trade_cost`。
  即「持仓带入下一 bar 的盈亏」减「换手成本」。

## 成交/成本模型（桩）

- 每次换手扣 `(fee_bp + slippage_bp)/1e4 × |Δposition|`。
- 这是简化桩：**未建模**真实盘口冲击、排队、撤单、容量。结论只用于相对比较。
- 非交易 bar（集合竞价、停牌、涨跌停封死）必须可处理为不可成交，跳过或强制 flat。

## Walk-forward（铁律）

- 按日期**时序**切分 train / val / test，**绝不打乱 bar**。
- 训练只在 train 窗，模型选择只看 val，最终指标只报 test 的 OOS。
- 报告必须把 in-sample 指标明确标注为 in-sample。

## 防过拟合自查

- 金融分钟数据有效样本极小、非平稳，RL 极易过拟合 + reward hacking。
- **必须**对比三条基线（TWAP / 日内动量 / 日内反转）；RL 打不过基线即视为无效。
- 警惕：用全样本标准化特征、用未来 VWAP、奖励里隐含未来信息、超参在 test 上调。
- 多次随机种子复跑，看 OOS 指标方差；单次漂亮曲线不可信。

## 数据接口

- `get_stock_min` / `get_future_min` / `get_index_min`，或用户提供的分钟导出。

> RL 策略与回测均为研究产物，不构成投资建议；本 agent 不做实盘交易。
