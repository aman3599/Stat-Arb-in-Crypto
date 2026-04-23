"""
diagnose_pairs.py — Diagnostic tool to understand why pairs fail screening.

Run this locally:
    python diagnose_pairs.py

It prints a full breakdown of every pair showing WHICH filter kills it,
plus a "relaxation ladder" showing how many pairs survive at progressively
looser thresholds. Use this to find the right thresholds empirically.
"""

import sys
import numpy as np
import pandas as pd

from data import load_prices, INTERVAL_1D
from cointegration import (
    compute_hedge_ratio, compute_spread, compute_halflife,
    johansen_test,
)
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations

TICKERS    = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "AVAX", "LINK"]
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-12-31"
BAR_HOURS   = 24.0   # daily bars


def diagnose(prices: pd.DataFrame):
    assets = list(prices.columns)
    pairs  = list(combinations(assets, 2))

    rows = []
    print(f"\nRunning diagnostics on {len(pairs)} pairs ({TRAIN_START} → {TRAIN_END})...\n")

    for a, b in pairs:
        pa = prices[a].dropna()
        pb = prices[b].dropna()
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]

        try:
            _, eg_p, _      = coint(np.log(pa), np.log(pb))
            joh_t, joh_c, joh_pass = johansen_test(pa, pb)
            hedge           = compute_hedge_ratio(pa, pb)
            spread          = compute_spread(pa, pb, hedge)
            hl              = compute_halflife(spread, bar_hours=BAR_HOURS)
            adf_p           = float(adfuller(spread.dropna(), autolag="AIC")[1])
        except Exception as e:
            print(f"  ERROR {a}/{b}: {e}")
            continue

        # Which filter(s) kill this pair at the current thresholds?
        fails = []
        if eg_p  >= 0.20:  fails.append(f"EG p={eg_p:.3f} ≥ 0.20")
        if not joh_pass:   fails.append(f"Johansen ({joh_t:.1f} < {joh_c:.1f})")
        if hl == np.inf:   fails.append("HL=inf (not mean-reverting)")
        elif hl > 120:     fails.append(f"HL={hl:.0f}d > 120")
        elif hl < 1:       fails.append(f"HL={hl:.1f}d < 1")
        if adf_p >= 0.15:  fails.append(f"ADF p={adf_p:.3f} ≥ 0.15")

        rows.append({
            "Pair":    f"{a}/{b}",
            "EG p":    round(eg_p,  4),
            "Joh":     "✓" if joh_pass else "✗",
            "Hedge":   round(hedge,  3),
            "HL (d)":  round(hl, 1) if hl != np.inf else "∞",
            "ADF p":   round(adf_p, 4),
            "Fails":   " | ".join(fails) if fails else "— PASSES ALL —",
        })

    df = pd.DataFrame(rows).sort_values("EG p")

    # ── Full diagnostic table ─────────────────────────────────────────────
    print("=" * 100)
    print(f"{'FULL DIAGNOSTIC TABLE':^100}")
    print("=" * 100)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    # ── Summary by failure mode ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FAILURE MODE BREAKDOWN")
    print("=" * 60)
    total = len(df)
    passes      = df["Fails"].str.contains("PASSES").sum()
    fail_eg     = df["Fails"].str.contains("EG").sum()
    fail_joh    = df["Fails"].str.contains("Johansen").sum()
    fail_hl     = df["Fails"].str.contains("HL").sum()
    fail_adf    = df["Fails"].str.contains("ADF").sum()
    fail_multi  = (df["Fails"].str.count(r"\|") >= 1).sum()

    print(f"  Total pairs:           {total}")
    print(f"  Pass all filters:      {passes}")
    print(f"  Fail EG only (or+):    {fail_eg}")
    print(f"  Fail Johansen (or+):   {fail_joh}")
    print(f"  Fail half-life (or+):  {fail_hl}")
    print(f"  Fail ADF (or+):        {fail_adf}")
    print(f"  Fail 2+ filters:       {fail_multi}")

    # ── Relaxation ladder ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RELAXATION LADDER — survivors at each threshold combination")
    print("=" * 60)

    raw = []
    for a, b in combinations(assets, 2):
        pa = prices[a].dropna(); pb = prices[b].dropna()
        common = pa.index.intersection(pb.index); pa, pb = pa.loc[common], pb.loc[common]
        try:
            _, eg_p, _          = coint(np.log(pa), np.log(pb))
            _, _, joh_pass      = johansen_test(pa, pb)
            hedge               = compute_hedge_ratio(pa, pb)
            spread              = compute_spread(pa, pb, hedge)
            hl                  = compute_halflife(spread, bar_hours=BAR_HOURS)
            adf_p               = float(adfuller(spread.dropna(), autolag="AIC")[1])
            raw.append((a, b, eg_p, joh_pass, hl, adf_p))
        except:
            pass

    configs = [
        ("Current  (eg≤0.20, hl≤120, adf≤0.15, Joh)",  0.20, 120, 0.15, True),
        ("Relaxed1 (eg≤0.25, hl≤150, adf≤0.20, Joh)",  0.25, 150, 0.20, True),
        ("Relaxed2 (eg≤0.30, hl≤180, adf≤0.25, Joh)",  0.30, 180, 0.25, True),
        ("No Joh   (eg≤0.20, hl≤120, adf≤0.15, no J)", 0.20, 120, 0.15, False),
        ("EG only  (eg≤0.05, no other filters)",        0.05, 9999, 1.0, False),
        ("EG only  (eg≤0.20, no other filters)",        0.20, 9999, 1.0, False),
    ]

    for label, eg_thr, hl_thr, adf_thr, req_joh in configs:
        survivors = [
            f"{a}/{b}" for a, b, eg_p, joh_pass, hl, adf_p in raw
            if eg_p < eg_thr
            and (not req_joh or joh_pass)
            and (hl != np.inf and hl <= hl_thr)
            and adf_p < adf_thr
        ]
        print(f"  {label:<50}  {len(survivors):>2} pairs  {survivors}")

    print("\nDone. Use the table above to pick thresholds where 3–7 pairs survive.")


if __name__ == "__main__":
    print(f"Loading prices from {TRAIN_START} (will slice to {TRAIN_END} after load)...")
    prices_full = load_prices(tickers=TICKERS, start=TRAIN_START, interval=INTERVAL_1D)

    # Debug: show exactly what came back before any slicing
    print(f"\nDEBUG prices_full shape: {prices_full.shape}")
    print(f"DEBUG index dtype: {prices_full.index.dtype}")
    print(f"DEBUG index tz: {prices_full.index.tz}")
    print(f"DEBUG first 3 index values: {list(prices_full.index[:3])}")
    print(f"DEBUG last  3 index values: {list(prices_full.index[-3:])}")
    print(f"DEBUG TRAIN_END value: '{TRAIN_END}'")

    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    print(f"DEBUG train_end_ts: {train_end_ts}")
    mask = prices_full.index <= train_end_ts
    print(f"DEBUG mask sum (rows <= train_end_ts): {mask.sum()}")

    prices = prices_full[mask]
    print(f"\nFull loaded range:  {prices_full.index[0].date()} → {prices_full.index[-1].date()}  ({len(prices_full):,} bars)")
    if prices.empty:
        print("ERROR: training slice is empty — see DEBUG output above.")
        sys.exit(1)
    print(f"Training window:    {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices):,} bars)")
    print(f"Tickers: {list(prices.columns)}\n")
    diagnose(prices)