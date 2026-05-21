from __future__ import annotations

import argparse

import torch

from .sell_only_pipeline import run_sell_only_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and backtest a pure sell-only exit model.")
    parser.add_argument("--data-dir", default="data/multi_index_2018_2026_yahoo")
    parser.add_argument("--output-dir", default="outputs/sell_only_retrained_30d")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--annual-cash-rate", type=float, default=0.0368)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_sell_only_research(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        cost_bps=args.cost_bps,
        annual_cash_rate=args.annual_cash_rate,
    )
    print(f"tickers={','.join(summary.tickers)}")
    print(f"best_candidate={summary.best_candidate}")
    print(f"validation_score={summary.validation_score:.6f}")
    print(f"outputs={summary.output_dir}")


if __name__ == "__main__":
    main()
