"""
sizing.py — Kelly-adjusted position sizing for crypto stat-arb.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SizingConfig:
    capital: float = 100_000        # total capital in USD
    max_position_pct: float = 0.20  # max 20% of capital per pair
    kelly_fraction: float = 0.25    # fractional Kelly (conservative)
    min_vol_lookback: int = 30      # days for vol estimation


def kelly_size(
    signals_df: pd.DataFrame,
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    config: SizingConfig = SizingConfig(),
) -> pd.DataFrame:
    """
    Compute dollar position sizes using fractional Kelly criterion.

    Kelly f* = (mu / sigma^2) for a continuous strategy.
    We estimate mu and sigma from the spread's recent returns,
    then apply fractional Kelly with position caps.

    Returns DataFrame with: pos_a_usd, pos_b_usd, leverage
    """
    spread = signals_df["spread"]
    position = signals_df["position"]

    spread_returns = spread.diff().dropna()
    rolling_vol = spread_returns.rolling(config.min_vol_lookback).std()
    rolling_mu = spread_returns.rolling(config.min_vol_lookback).mean()

    kelly_f = config.kelly_fraction * (rolling_mu / (rolling_vol ** 2 + 1e-9))
    kelly_f = kelly_f.clip(lower=0.0, upper=config.max_position_pct)

    dollar_exposure = kelly_f * config.capital
    pos_a_usd = dollar_exposure * position
    pos_b_usd = -dollar_exposure * hedge_ratio * position

    sizing_df = pd.DataFrame({
        "pos_a_usd": pos_a_usd.round(2),
        "pos_b_usd": pos_b_usd.round(2),
        "dollar_exposure": dollar_exposure.round(2),
        "kelly_f": kelly_f.round(4),
        "rolling_vol": rolling_vol.round(6),
    }, index=signals_df.index)

    return sizing_df
