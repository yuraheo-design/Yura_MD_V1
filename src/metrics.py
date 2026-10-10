"""오차 지표: MAPE / RMSE / Bias"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """실제 판매량이 0인 날은 (분모가 0이라) 표준 MAPE 정의상 제외하고 계산한다.
    편의점/저매출 SKU처럼 0 판매일이 잦은 데이터에서, eps로 나누면 오차율이
    수십만%로 폭발해 지표가 무의미해지기 때문. 0 판매일 비중은 별도로 봐야 한다
    (WAPE를 함께 보는 것을 권장)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def wape(actual: pd.Series, forecast: pd.Series) -> float:
    """Weighted MAPE = sum(|오차|) / sum(실제값). 0 판매일이 많은 저매출 SKU에서도
    안정적으로 계산되는 집계 오차율이라 그룹/SKU 간 비교에 더 적합하다."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - forecast)) / denom * 100)


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(forecast - actual))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def metrics_by_group(df: pd.DataFrame, group_col: str, actual_col: str, forecast_col: str) -> pd.DataFrame:
    """SKU별/그룹별 MAPE·RMSE·Bias 요약표."""
    out = (
        df.groupby(group_col)
        .apply(
            lambda g: pd.Series(
                {
                    "MAPE": mape(g[actual_col], g[forecast_col]),
                    "WAPE": wape(g[actual_col], g[forecast_col]),
                    "RMSE": rmse(g[actual_col], g[forecast_col]),
                    "Bias": bias(g[actual_col], g[forecast_col]),
                    "n_days": len(g),
                }
            )
        )
        .reset_index()
    )
    return out
