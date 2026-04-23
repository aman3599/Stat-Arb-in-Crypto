"""
cointegration.py — Pairs screening via Johansen and Engle-Granger tests.
Also computes Ornstein-Uhlenbeck half-life for mean-reversion filtering.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from typing import Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm


@dataclass
class PairResult:
    asset_a: str
    asset_b: str
    eg_pvalue: float          # Engle-Granger p-value
    johansen_trace_stat: float
    johansen_crit_95: float   # 95% critical value
    johansen_cointegrated: bool
    hedge_ratio: float        # OLS beta (b in: a = beta*b + spread)
    half_life_days: float     # OU mean-reversion half-life in calendar days
    adf_pvalue: float         # ADF on residuals
    is_valid: bool            # passes all filters


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS regression: log(A) ~ beta * log(B) + c. Returns beta."""
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    X = sm.add_constant(log_b)
    model = sm.OLS(log_a, X).fit()
    return float(model.params.iloc[1])


def compute_spread(
    price_a: pd.Series, price_b: pd.Series, hedge_ratio: float
) -> pd.Series:
    """Log-price spread: log(A) - hedge_ratio * log(B)."""
    return np.log(price_a) - hedge_ratio * np.log(price_b)


def compute_halflife(spread: pd.Series, bar_hours: float = 24.0) -> float:
    """
    Ornstein-Uhlenbeck half-life via OLS on spread differences.
    Estimated by: delta_S ~ theta * S_lag + c
    half_life = -log(2) / theta  (in bars), converted to calendar days via bar_hours.
    Returns np.inf if the spread is not mean-reverting (theta >= 0).
    """
    spread_lag = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    common = spread_lag.index.intersection(delta_spread.index)
    X = sm.add_constant(spread_lag.loc[common])
    y = delta_spread.loc[common]
    model = sm.OLS(y, X).fit()
    theta = model.params.iloc[1]
    if theta >= 0:
        return np.inf  # not mean-reverting

    half_life_bars = -np.log(2) / theta
    half_life_days = (half_life_bars * bar_hours) / 24.0

    return half_life_days


def johansen_test(price_a: pd.Series, price_b: pd.Series) -> tuple:
    """
    Johansen trace test for cointegration.
    Returns (trace_stat, critical_value_95, is_cointegrated).
    """
    log_prices = pd.concat([np.log(price_a), np.log(price_b)], axis=1).dropna()
    result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
    trace_stat = float(result.lr1[0])        # trace stat for r=0
    crit_95 = float(result.cvt[0, 1])        # 95% critical value
    return trace_stat, crit_95, trace_stat > crit_95


def screen_pairs(
    prices: pd.DataFrame,
    eg_pvalue_threshold: float = 0.05,
    min_halflife_days: float = 3.0,
    max_halflife_days: float = 30.0,
    adf_pvalue_threshold: float = 0.05,
    require_johansen: bool = True,
    bar_hours: float = 24.0,
) -> list:
    """
    Screen all pairs from prices DataFrame.
    Filters: EG p-value, Johansen trace (optional), OU half-life, ADF on residuals.
    Returns list of PairResult sorted by EG p-value.
    """
    assets = list(prices.columns)
    pairs = list(combinations(assets, 2))
    results = []

    for a, b in pairs:
        try:
            pa = prices[a].dropna()
            pb = prices[b].dropna()
            common = pa.index.intersection(pb.index)
            pa, pb = pa.loc[common], pb.loc[common]

            if len(pa) < 60:
                continue

            eg_stat, eg_pvalue, _ = coint(np.log(pa), np.log(pb))
            joh_trace, joh_crit95, joh_cointegrated = johansen_test(pa, pb)
            hedge_ratio = compute_hedge_ratio(pa, pb)
            spread = compute_spread(pa, pb, hedge_ratio)
            hl = compute_halflife(spread, bar_hours=bar_hours)
            adf_res = adfuller(spread.dropna(), autolag="AIC")
            adf_pvalue = float(adf_res[1])

            is_valid = (
                eg_pvalue < eg_pvalue_threshold
                and (not require_johansen or joh_cointegrated)
                and min_halflife_days <= hl <= max_halflife_days
                and adf_pvalue < adf_pvalue_threshold
            )

            results.append(PairResult(
                asset_a=a, asset_b=b,
                eg_pvalue=round(eg_pvalue, 4),
                johansen_trace_stat=round(joh_trace, 3),
                johansen_crit_95=round(joh_crit95, 3),
                johansen_cointegrated=joh_cointegrated,
                hedge_ratio=round(hedge_ratio, 4),
                half_life_days=round(hl, 2),
                adf_pvalue=round(adf_pvalue, 4),
                is_valid=is_valid
            ))

        except Exception as e:
            print(f"  Skipping {a}/{b}: {e}")
            continue

    results.sort(key=lambda r: r.eg_pvalue)
    return results


def print_screening_report(results: list) -> None:
    valid = [r for r in results if r.is_valid]
    print(f"\n{'='*60}")
    print(f"COINTEGRATION SCREENING REPORT")
    print(f"{'='*60}")
    print(f"Pairs screened: {len(results)}  |  Valid pairs: {len(valid)}")
    print(f"\n{'Pair':<18} {'EG p-val':>9} {'Johansen':>9} {'Hedge':>7} {'HalfLife':>9} {'ADF p':>8} {'Valid':>6}")
    print("-"*60)
    for r in results:
        joh = "✓" if r.johansen_cointegrated else "✗"
        valid_mark = "✓" if r.is_valid else ""
        pair = f"{r.asset_a[:6]}/{r.asset_b[:6]}"
        print(f"{pair:<18} {r.eg_pvalue:>9.4f} {joh:>9} {r.hedge_ratio:>7.3f} {r.half_life_days:>8.1f}d {r.adf_pvalue:>8.4f} {valid_mark:>6}")
    print(f"{'='*60}\n")
