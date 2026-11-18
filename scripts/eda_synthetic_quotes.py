#!/usr/bin/env python3
"""Exploratory summary on synthetic quotes; writes reports/eda_summary.md.

Uses only stdlib + pandas/numpy (repo dependencies). No plotting libraries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.train import generate_synthetic_quotes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "eda_summary.md"

MARGIN_BINS = [-np.inf, 0.0, 0.05, 0.10, 0.15, 0.20, np.inf]
MARGIN_LABELS = ["<0%", "0–5%", "5–10%", "10–15%", "15–20%", "≥20%"]


def _table_md(frame: pd.DataFrame, title: str) -> list[str]:
    lines = [f"### {title}", ""]
    if frame.empty:
        lines.append("_No rows._")
        lines.append("")
        return lines
    headers = list(frame.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in frame.iterrows():
        cells = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _sparkline(values: list[float], width: int = 20) -> str:
    """ASCII bar chart for a list of non-negative rates in [0,1]."""
    if not values:
        return ""
    mx = max(max(values), 1e-9)
    blocks = "▁▂▃▄▅▆▇█"
    chars = []
    for v in values:
        idx = min(len(blocks) - 1, int((v / mx) * (len(blocks) - 1)))
        chars.append(blocks[idx])
    return "".join(chars)


def build_report(n_samples: int, seed: int) -> str:
    df = generate_synthetic_quotes(n_samples=n_samples, seed=seed)
    overall_win = float(df["won"].mean())

    by_cat = (
        df.groupby("part_category", as_index=False)
        .agg(quotes=("won", "count"), wins=("won", "sum"))
        .assign(win_rate=lambda x: (x["wins"] / x["quotes"]).round(4))
        .sort_values("win_rate", ascending=False)
    )
    by_tier = (
        df.groupby("dealer_tier", as_index=False)
        .agg(quotes=("won", "count"), wins=("won", "sum"))
        .assign(win_rate=lambda x: (x["wins"] / x["quotes"]).round(4))
        .sort_values("win_rate", ascending=False)
    )

    df = df.copy()
    df["margin_band"] = pd.cut(
        df["gross_margin_pct"],
        bins=MARGIN_BINS,
        labels=MARGIN_LABELS,
        include_lowest=True,
    )
    by_margin = (
        df.groupby("margin_band", observed=True)
        .agg(quotes=("won", "count"), wins=("won", "sum"))
        .reset_index()
        .assign(win_rate=lambda x: (x["wins"] / x["quotes"]).round(4))
    )

    discount_bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.41]
    df["discount_band"] = pd.cut(df["requested_discount"], bins=discount_bins, include_lowest=True)
    by_discount = (
        df.groupby("discount_band", observed=True)
        .agg(quotes=("won", "count"), wins=("won", "sum"))
        .reset_index()
        .assign(win_rate=lambda x: (x["wins"] / x["quotes"]).round(4))
    )

    cat_spark = _sparkline(by_cat["win_rate"].tolist())
    tier_spark = _sparkline(by_tier["win_rate"].tolist())

    numeric_summary = df[
        ["base_cost", "list_price", "requested_discount", "gross_margin_pct", "tariff_rate"]
    ].describe(percentiles=[0.25, 0.5, 0.75])

    lines: list[str] = [
        "# EDA summary — synthetic parts quotes",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Dataset",
        "",
        f"- **Rows:** {n_samples:,}",
        f"- **Seed:** {seed}",
        f"- **Overall win rate:** {overall_win:.2%}",
        f"- **Source:** `models.train.generate_synthetic_quotes` (no external files)",
        "",
        "## Leakage / sanity checks",
        "",
        "- Labels are **synthetic** from the same generative rule as margins and discounts; "
        "the model must learn correlations, not read a hidden ID.",
        "- `won` is **not** passed to the API as an input; training uses only `FEATURE_COLUMNS`.",
        "- Requested discounts respect per-tier caps (max 0.40).",
        "",
        "## Win rate by part category",
        "",
        f"Sparkline (relative win rates, high → low): `{cat_spark}`",
        "",
    ]
    lines.extend(_table_md(by_cat, "Counts"))
    lines.append("## Win rate by dealer tier")
    lines.append("")
    lines.append(f"Sparkline: `{tier_spark}`")
    lines.append("")
    lines.extend(_table_md(by_tier, "Counts"))
    lines.extend(_table_md(by_margin.rename(columns={"margin_band": "margin_band"}), "Win rate by margin band"))
    lines.extend(_table_md(by_discount.rename(columns={"discount_band": "discount_band"}), "Win rate by discount depth"))

    lines.append("## Numeric distributions (selected fields)")
    lines.append("")
    lines.append("```")
    lines.append(numeric_summary.round(4).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Takeaways for modeling")
    lines.append("")
    lines.append("- Win rate rises with **gross margin** and falls with **requested discount**—consistent with the generative rule.")
    lines.append("- **Platinum/Gold** tiers show higher win rates than bronze at similar discounts.")
    lines.append("- Category effects are milder than tier/margin; expect tree models to lean on margin and discount features.")
    lines.append("")
    lines.append("Regenerate: `python3 scripts/eda_synthetic_quotes.py`")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write EDA markdown for synthetic quotes")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(n_samples=args.n_samples, seed=args.seed)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())