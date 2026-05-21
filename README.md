# Sell-Only Exit Model

这是一个纯净版卖出模型项目。

它不研究买入，不接入市场状态入场系统，也不把 buy and hold 放在主比较位。当前项目只回答一个问题：

```text
已经买入以后，什么时候卖出更好？
```

## 项目定位

本项目是一个智能离场规则研究框架。

核心目标不是证明模型能长期跑赢持有指数，而是比较不同离场规则的交易质量：

- RL Sell Model
- Tuned Trailing Stop
- Tuned Fixed TP/SL

## 模型结构

每笔交易开始时默认已经买入，模型每天只能选择：

```text
hold: 继续持有
sell: 卖出退出
```

卖出模型由三个 DQN specialist 组成：

```text
bull agent
bear agent
sideways agent
```

每个 agent 都是一个 `hold / sell` 二分类动作模型。真实决策时，soft router 会把三个 agent 的 Q 值加权：

```text
weighted_q =
w_bull * q_bull
+ w_bear * q_bear
+ w_sideways * q_sideways
```

然后选择 Q 值更高的动作。

## 状态特征

模型使用 6 个输入因子：

```text
今日涨跌幅
当前回撤
已持有天数比例
近期波动率
当前浮动收益
短期趋势
```

## 奖励机制

```text
hold:
reward = -holding_penalty
```

含义：继续持有有轻微时间成本 / 机会成本。

```text
sell:
trade_return = price / entry_price - 1.0
drawdown = price / peak_price - 1.0
reward = trade_return + drawdown_penalty * drawdown
```

含义：卖出收益越高越好；从最高点回撤越大，奖励越低。

## 实验设置

当前默认设置：

```text
交易窗口: 30 个交易日
最长持有: 约 29 个交易日（入口日为 day 0）
交易成本: 5 bps
现金利率: 3.68% annualized
数据: 16 个指数 ETF 代理
默认 seed: 11
```

时间切分：

```text
train: 2017-01-03 到 2022-08-15
validation: 2022-08-16 到 2024-07-02
test: 2024-07-03 到 2026-05-20
```

## 运行

```bash
python -m src.main
```

也可以显式指定参数：

```bash
python -m src.main \
  --data-dir data/multi_index_2018_2026_yahoo \
  --output-dir outputs/sell_only_retrained_30d \
  --seed 11 \
  --cost-bps 5 \
  --annual-cash-rate 0.0368
```

输出会写入：

```text
outputs/sell_only_retrained_30d
```

主要输出文件：

```text
summary.json
validation_candidates.csv
test_metrics_aggregate.csv
test_metrics_by_asset.csv
models/<candidate>/*.pt
```

默认流程会做 6 组候选参数搜索，并用 validation score 选择测试集模型。当前默认最佳候选：

```text
sell30_lr1e-3_g95_h64
```

## 项目优势

当前模型的优势不是挑战 buy and hold，而是作为离场规则优于机械止损。

在同一维度比较下，RL Sell Model 相比 tuned trailing stop 和 tuned fixed TP/SL：

- Avg PnL 更高
- Win Rate 更高
- P/L Ratio 更高
- Sharpe 更高
- 更少被短期波动洗出去
- 更能保留上涨趋势

## 当前局限

- 不包含买入模型
- 不负责选股或择时入场
- 不保证跑赢长期持有
- 最大回撤不一定低于机械止损
- 当前奖励机制还没有显式学习“卖出后未来是否继续下跌”的反事实信息

## 最新回测结果

主比较对象：离场规则，不包含 buy and hold。

以下结果由默认命令 `python -m src.main` 重新生成，输出目录为 `outputs/sell_only_retrained_30d`，默认 seed 为 11。

| 策略 | 总收益 | CAGR | Sharpe | 最大回撤 | 在场时间 | Avg PnL | Win Rate | P/L Ratio | 平均持仓 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RL Sell Model | 33.33% | 16.46% | 0.899 | -20.19% | 88.47% | 1.95% | 65.63% | 1.57 | 27.10 天 |
| Tuned Fixed TP/SL | 13.83% | 6.91% | 0.507 | -14.27% | 65.93% | 0.84% | 50.78% | 1.44 | 20.40 天 |
| Tuned Trailing Stop | 3.43% | 1.67% | 0.188 | -15.71% | 66.45% | 0.25% | 44.92% | 1.43 | 20.57 天 |

逐资产胜率：

```text
Avg PnL:
RL Sell Model > Tuned Trailing Stop: 16 / 16
RL Sell Model > Tuned Fixed TP/SL: 15 / 16

Win Rate:
RL Sell Model > Tuned Trailing Stop: 16 / 16
RL Sell Model > Tuned Fixed TP/SL: 16 / 16
```

一句话总结：

```text
这是一个纯卖出模型项目。
它的价值在于做出比传统 trailing stop / 固定止盈止损更聪明的离场判断。
```
