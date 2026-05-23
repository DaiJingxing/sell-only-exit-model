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


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_nav(method_results: dict[str, list[DailyBacktestResult]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method, results in method_results.items():
        nav = _mean_nav(results)
        ax.plot(nav, label=_label(method), color=_color(method), linewidth=2.2)
    ax.set_title("Average NAV on Test Set", fontsize=14, weight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Net asset value")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_returns(aggregate_metrics: dict[str, dict], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = list(aggregate_metrics)
    labels = [_label(method) for method in methods]
    total_returns = [aggregate_metrics[method]["total_return"] * 100.0 for method in methods]
    avg_pnls = [aggregate_metrics[method]["avg_pnl"] * 100.0 for method in methods]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(x - width / 2, total_returns, width, label="Total return", color="#2563eb")
    ax.bar(x + width / 2, avg_pnls, width, label="Avg trade PnL", color="#14b8a6")
    ax.set_title("Return Comparison on Test Set", fontsize=14, weight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_sharpe(aggregate_metrics: dict[str, dict], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = list(aggregate_metrics)
    labels = [_label(method) for method in methods]
    values = [aggregate_metrics[method]["sharpe_ratio"] for method in methods]
    colors = [_color(method) for method in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(x, values, color=colors, width=0.55)
    ax.set_title("Sharpe Ratio on Test Set", fontsize=14, weight="bold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    _style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_research_charts(
    method_results: dict[str, list[DailyBacktestResult]],
    aggregate_metrics: dict[str, dict],
    output_dir: str | Path,
) -> dict[str, str]:
    figure_dir = Path(output_dir) / "figures"
    paths = {
        "nav": figure_dir / "nav_test.png",
        "returns": figure_dir / "returns_test.png",
        "sharpe": figure_dir / "sharpe_test.png",
    }
    plot_nav(method_results, paths["nav"])
    plot_returns(aggregate_metrics, paths["returns"])
    plot_sharpe(aggregate_metrics, paths["sharpe"])
    return {name: str(path) for name, path in paths.items()}
