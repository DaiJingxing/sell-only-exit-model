from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sell-only-exit-model-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .daily_backtest import DailyBacktestResult


METHOD_LABELS = {
    "sell_model": "RL Sell Model",
    "tuned_fixed_tp0.08_sl-0.03": "Tuned Fixed TP/SL",
    "tuned_trailing_stop_0.04": "Tuned Trailing Stop",
}

METHOD_COLORS = {
    "sell_model": "#2563eb",
    "tuned_fixed_tp0.08_sl-0.03": "#16a34a",
    "tuned_trailing_stop_0.04": "#f97316",
    "sp500_buy_hold": "#111827",
}


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#525252")


def _mean_nav(results: list[DailyBacktestResult]) -> np.ndarray:
    if not results:
        return np.asarray([1.0], dtype=np.float64)
    min_len = min(len(result.equity_curve) for result in results)
    curves = np.asarray([result.equity_curve[:min_len] for result in results], dtype=np.float64)
    return np.mean(curves, axis=0)


def _daily_returns(nav: np.ndarray) -> np.ndarray:
    nav = np.asarray(nav, dtype=np.float64)
    if nav.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(nav) / np.maximum(nav[:-1], 1e-12)


def _aligned_strategy_and_benchmark(
    method_results: dict[str, list[DailyBacktestResult]],
    benchmark_result: DailyBacktestResult,
) -> tuple[np.ndarray, np.ndarray]:
    strategy_nav = _mean_nav(method_results["sell_model"])
    benchmark_nav = np.asarray(benchmark_result.equity_curve, dtype=np.float64)
    n = min(strategy_nav.size, benchmark_nav.size)
    if n < 2:
        return np.asarray([1.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)
    return strategy_nav[:n], benchmark_nav[:n]


def _beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    n = min(strategy_returns.size, benchmark_returns.size)
    if n < 3:
        return 0.0
    strategy = strategy_returns[:n]
    benchmark = benchmark_returns[:n]
    benchmark_var = float(np.var(benchmark))
    if benchmark_var <= 1e-12:
        return 0.0
    return float(np.cov(strategy, benchmark, ddof=0)[0, 1] / benchmark_var)


def _rolling_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, window: int = 63) -> np.ndarray:
    n = min(strategy_returns.size, benchmark_returns.size)
    values = np.full(n, np.nan, dtype=np.float64)
    for idx in range(window - 1, n):
        values[idx] = _beta(strategy_returns[idx - window + 1 : idx + 1], benchmark_returns[idx - window + 1 : idx + 1])
    return values


def _rolling_volatility(returns: np.ndarray, window: int = 21) -> np.ndarray:
    values = np.full(returns.size, np.nan, dtype=np.float64)
    for idx in range(window - 1, returns.size):
        values[idx] = float(np.std(returns[idx - window + 1 : idx + 1]) * np.sqrt(252.0))
    return values


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_nav_vs_sp500(
    method_results: dict[str, list[DailyBacktestResult]],
    benchmark_result: DailyBacktestResult,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategy_nav, benchmark_nav = _aligned_strategy_and_benchmark(method_results, benchmark_result)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(strategy_nav, label="RL Sell Model", color=_color("sell_model"), linewidth=2.4)
    ax.plot(benchmark_nav, label="S&P 500 Buy & Hold (SPY)", color=_color("sp500_buy_hold"), linewidth=2.0)
    ax.set_title("NAV vs S&P 500 Buy & Hold", fontsize=14, weight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Net asset value")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_beta_vs_sp500(
    method_results: dict[str, list[DailyBacktestResult]],
    benchmark_result: DailyBacktestResult,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategy_nav, benchmark_nav = _aligned_strategy_and_benchmark(method_results, benchmark_result)
    strategy_returns = _daily_returns(strategy_nav)
    benchmark_returns = _daily_returns(benchmark_nav)
    rolling = _rolling_beta(strategy_returns, benchmark_returns)
    full_beta = _beta(strategy_returns, benchmark_returns)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(rolling, color=_color("sell_model"), linewidth=2.2, label="63-day rolling beta")
    ax.axhline(full_beta, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Full-period beta: {full_beta:.2f}")
    ax.axhline(1.0, color="#9ca3af", linestyle=":", linewidth=1.4, label="Market beta = 1.00")
    ax.set_title("Beta vs S&P 500", fontsize=14, weight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Beta")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_outperformance_by_volatility(
    method_results: dict[str, list[DailyBacktestResult]],
    benchmark_result: DailyBacktestResult,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategy_nav, benchmark_nav = _aligned_strategy_and_benchmark(method_results, benchmark_result)
    strategy_returns = _daily_returns(strategy_nav)
    benchmark_returns = _daily_returns(benchmark_nav)
    n = min(strategy_returns.size, benchmark_returns.size)
    excess = strategy_returns[:n] - benchmark_returns[:n]
    market_vol = _rolling_volatility(benchmark_returns[:n])
    mask = np.isfinite(market_vol)
    excess = excess[mask]
    market_vol = market_vol[mask]

    if market_vol.size:
        low_cut, high_cut = np.quantile(market_vol, [1.0 / 3.0, 2.0 / 3.0])
        buckets = {
            "Calm market": market_vol <= low_cut,
            "Normal market": (market_vol > low_cut) & (market_vol <= high_cut),
            "Volatile market": market_vol > high_cut,
        }
    else:
        buckets = {"Calm market": np.asarray([], dtype=bool), "Normal market": np.asarray([], dtype=bool), "Volatile market": np.asarray([], dtype=bool)}

    labels = list(buckets)
    annualized_excess = [float(np.mean(excess[buckets[label]]) * 252.0 * 100.0) if np.any(buckets[label]) else 0.0 for label in labels]
    hit_rates = [float(np.mean(excess[buckets[label]] > 0.0) * 100.0) if np.any(buckets[label]) else 0.0 for label in labels]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(x, annualized_excess, color=["#94a3b8", "#2563eb", "#dc2626"], width=0.58)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_title("When the Model Outperforms S&P 500", fontsize=14, weight="bold")
    ax.set_ylabel("Annualized excess return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    _style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(x, hit_rates, color="#0f766e", marker="o", linewidth=2.0, label="Daily outperformance hit rate")
    ax2.set_ylabel("Hit rate (%)")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    for bar, value in zip(bars, annualized_excess):
        va = "bottom" if value >= 0 else "top"
        offset = 1.0 if value >= 0 else -1.0
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}%", ha="center", va=va, fontsize=10)
    for idx, value in enumerate(hit_rates):
        ax2.text(idx, value + 2.0, f"{value:.0f}%", ha="center", va="bottom", color="#0f766e", fontsize=10)

    lines, line_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, line_labels, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_research_charts(
    method_results: dict[str, list[DailyBacktestResult]],
    benchmark_result: DailyBacktestResult,
    output_dir: str | Path,
) -> dict[str, str]:
    figure_dir = Path(output_dir) / "figures"
    paths = {
        "nav_vs_sp500": figure_dir / "nav_vs_sp500.png",
        "beta_vs_sp500": figure_dir / "beta_vs_sp500.png",
        "outperformance_by_volatility": figure_dir / "outperformance_by_volatility.png",
    }
    plot_nav_vs_sp500(method_results, benchmark_result, paths["nav_vs_sp500"])
    plot_beta_vs_sp500(method_results, benchmark_result, paths["beta_vs_sp500"])
    plot_outperformance_by_volatility(method_results, benchmark_result, paths["outperformance_by_volatility"])
    return {name: str(path) for name, path in paths.items()}
