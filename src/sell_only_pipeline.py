from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random

import numpy as np
import pandas as pd
import torch

from .daily_backtest import DailyBacktestResult, daily_agent_backtest, daily_buy_and_hold, daily_rule_backtest
from .data_loader import split_prices
from .dqn import DQNConfig, load_agent
from .router import EnsembleAgent
from .train_router import RouterTrainConfig, train_router_on_price_sets
from .train_router import load_router
from .train_specialists import train_specialists
from .visualize import write_research_charts


@dataclass(frozen=True)
class Candidate:
    name: str
    dqn_lr: float
    gamma: float
    hidden_dim: int
    specialist_episodes: int
    router_lr: float
    router_episodes: int
    episode_length: int


@dataclass(frozen=True)
class SellOnlySummary:
    tickers: list[str]
    best_candidate: dict
    validation_score: float
    aggregate_test_metrics: dict[str, dict]
    charts: dict[str, str]
    output_dir: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metrics(results: list[DailyBacktestResult]) -> dict[str, float]:
    keys = results[0].metrics.keys()
    return {key: float(np.mean([result.metrics[key] for result in results])) for key in keys}


def _trade_metrics(result: DailyBacktestResult) -> dict[str, float]:
    returns = np.asarray([float(trade["return"]) for trade in result.trades if "return" in trade], dtype=np.float64)
    holding_days = np.asarray([float(trade.get("holding_days", 0.0)) for trade in result.trades], dtype=np.float64)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    return {
        "avg_pnl": float(np.mean(returns)) if returns.size else 0.0,
        "median_pnl": float(np.median(returns)) if returns.size else 0.0,
        "win_rate": float(np.mean(returns > 0.0)) if returns.size else 0.0,
        "avg_win": float(np.mean(wins)) if wins.size else 0.0,
        "avg_loss": float(np.mean(losses)) if losses.size else 0.0,
        "pl_ratio": float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else 0.0,
        "avg_holding_days": float(np.mean(holding_days)) if holding_days.size else 0.0,
    }


def _mean_rows(rows: list[dict], keys: list[str]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _score(rows: list[dict]) -> float:
    if not rows:
        return -1e9
    sharpe = np.asarray([row["sharpe_ratio"] for row in rows], dtype=np.float32)
    cagr = np.asarray([row["cagr"] for row in rows], dtype=np.float32)
    maxdd = np.asarray([row["max_drawdown"] for row in rows], dtype=np.float32)
    time_in_market = np.asarray([row["time_in_market"] for row in rows], dtype=np.float32)
    exposure_penalty = np.maximum(time_in_market - 0.85, 0.0)
    return float(
        1.0 * np.mean(sharpe)
        + 2.0 * np.mean(cagr)
        + 0.6 * np.mean(maxdd)
        - 0.5 * np.mean(exposure_penalty)
        - 0.25 * np.std(sharpe)
    )


def _candidate_grid() -> list[Candidate]:
    return [
        Candidate("sell30_lr1e-3_g95_h64", 1e-3, 0.95, 64, 80, 1e-3, 90, 30),
        Candidate("sell30_lr1e-3_g97_h64", 1e-3, 0.97, 64, 80, 1e-3, 90, 30),
        Candidate("sell30_lr5e-4_g97_h64", 5e-4, 0.97, 64, 80, 5e-4, 90, 30),
        Candidate("sell30_lr1e-3_g99_h64", 1e-3, 0.99, 64, 80, 1e-3, 90, 30),
        Candidate("sell30_lr1e-3_g97_h128", 1e-3, 0.97, 128, 80, 1e-3, 90, 30),
        Candidate("sell30_lr5e-4_g99_h128", 5e-4, 0.99, 128, 80, 5e-4, 90, 30),
    ]


def _load_ensemble(model_dir: str, router_path: str, candidate: Candidate, device: str) -> EnsembleAgent:
    cfg = DQNConfig(device=device, lr=candidate.dqn_lr, gamma=candidate.gamma, hidden_dim=candidate.hidden_dim)
    model_path = Path(model_dir)
    return EnsembleAgent(
        load_agent(str(model_path / "bull_agent.pt"), config=cfg, map_location=device),
        load_agent(str(model_path / "bear_agent.pt"), config=cfg, map_location=device),
        load_agent(str(model_path / "sideways_agent.pt"), config=cfg, map_location=device),
        load_router(router_path, device=device),
        device=device,
    )


def _load_local_price_sets(data_dir: Path) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        close = pd.to_numeric(df["Close"], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if close.size >= 500:
            series[path.stem.replace("_us", "").upper()] = close
    if len(series) < 3:
        raise RuntimeError(f"Need at least three usable CSV files in {data_dir}.")
    return series


def run_sell_only_research(
    data_dir: str = "data/multi_index_2018_2026_yahoo",
    output_dir: str = "outputs/sell_only_retrained_30d",
    seed: int = 11,
    device: str = "cpu",
    cost_bps: float = 5.0,
    annual_cash_rate: float = 0.0368,
) -> SellOnlySummary:
    _set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    series_by_ticker = _load_local_price_sets(Path(data_dir))
    splits_by_ticker = {ticker: split_prices(prices) for ticker, prices in series_by_ticker.items()}
    train_sets = [split.train for split in splits_by_ticker.values()]
    validation_sets = {ticker: split.validation for ticker, split in splits_by_ticker.items()}
    test_sets = {ticker: split.test for ticker, split in splits_by_ticker.items()}

    validation_rows: list[dict] = []
    best: tuple[float, Candidate, str, str] | None = None
    for idx, candidate in enumerate(_candidate_grid()):
        candidate_seed = seed + idx * 1000
        _set_seed(candidate_seed)
        model_dir = str(out / "models" / candidate.name)
        router_path = str(out / "models" / candidate.name / "router.pt")
        dqn_cfg = DQNConfig(
            device=device,
            lr=candidate.dqn_lr,
            gamma=candidate.gamma,
            hidden_dim=candidate.hidden_dim,
        )
        train_specialists(model_dir, candidate.specialist_episodes, candidate.episode_length, candidate_seed, device, config=dqn_cfg)
        train_router_on_price_sets(
            train_sets,
            model_dir,
            router_path,
            RouterTrainConfig(
                episodes=candidate.router_episodes,
                gamma=candidate.gamma,
                lr=candidate.router_lr,
                device=device,
                episode_length=candidate.episode_length,
            ),
            seed=candidate_seed,
        )
        agent = _load_ensemble(model_dir, router_path, candidate, device)
        metric_rows = []
        validation_results = []
        for ticker, prices in validation_sets.items():
            del ticker
            result = daily_agent_backtest(
                prices,
                agent,
                episode_length=candidate.episode_length,
                cost_bps=cost_bps,
                annual_cash_rate=annual_cash_rate,
            )
            metric_rows.append(result.metrics)
            validation_results.append(result)
        score = _score(metric_rows)
        validation_rows.append({"candidate": candidate.name, "score": score, **asdict(candidate), **_mean_metrics(validation_results)})
        if best is None or score > best[0]:
            best = (score, candidate, model_dir, router_path)

    if best is None:
        raise RuntimeError("No candidate completed validation.")

    best_score, best_candidate, best_model_dir, best_router_path = best
    best_agent = _load_ensemble(best_model_dir, best_router_path, best_candidate, device)
    methods: dict[str, list[DailyBacktestResult]] = {
        "sell_model": [],
        "tuned_trailing_stop_0.04": [],
        "tuned_fixed_tp0.08_sl-0.03": [],
    }
    by_asset_rows: list[dict] = []
    for ticker, prices in test_sets.items():
        results = {
            "sell_model": daily_agent_backtest(
                prices,
                best_agent,
                episode_length=best_candidate.episode_length,
                cost_bps=cost_bps,
                annual_cash_rate=annual_cash_rate,
            ),
            "tuned_trailing_stop_0.04": daily_rule_backtest(
                prices,
                "trailing_stop",
                episode_length=best_candidate.episode_length,
                trail=0.04,
                cost_bps=cost_bps,
                annual_cash_rate=annual_cash_rate,
            ),
            "tuned_fixed_tp0.08_sl-0.03": daily_rule_backtest(
                prices,
                "fixed_tp_sl",
                episode_length=best_candidate.episode_length,
                take_profit=0.08,
                stop_loss=-0.03,
                cost_bps=cost_bps,
                annual_cash_rate=annual_cash_rate,
            ),
        }
        for method, result in results.items():
            methods[method].append(result)
            by_asset_rows.append({"ticker": ticker, "method": method, **result.metrics, **_trade_metrics(result)})

    trade_keys = ["avg_pnl", "median_pnl", "win_rate", "avg_win", "avg_loss", "pl_ratio", "avg_holding_days"]
    aggregate = {}
    for method, results in methods.items():
        method_rows = [row for row in by_asset_rows if row["method"] == method]
        aggregate[method] = {**_mean_metrics(results), **_mean_rows(method_rows, trade_keys)}
    if "SPY" not in test_sets:
        raise RuntimeError("SPY data is required to build the S&P 500 benchmark visualizations.")
    benchmark_result = daily_buy_and_hold(
        test_sets["SPY"],
        cost_bps=cost_bps,
        annual_cash_rate=annual_cash_rate,
    )
    aggregate["sp500_buy_hold"] = {**benchmark_result.metrics, **_trade_metrics(benchmark_result)}
    charts = write_research_charts(methods, benchmark_result, out)
    summary = SellOnlySummary(
        tickers=list(splits_by_ticker),
        best_candidate=asdict(best_candidate),
        validation_score=best_score,
        aggregate_test_metrics=aggregate,
        charts=charts,
        output_dir=str(out),
    )
    _write_csv(out / "validation_candidates.csv", validation_rows)
    _write_csv(out / "test_metrics_by_asset.csv", by_asset_rows)
    _write_csv(out / "test_metrics_aggregate.csv", [{"method": method, **metrics} for method, metrics in aggregate.items()])
    (out / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
