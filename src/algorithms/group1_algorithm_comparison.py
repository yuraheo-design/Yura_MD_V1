"""
그룹 1. 예측 알고리즘 비교
  1-1 선형추세(Linear Trend) 단독 성능 측정
  1-2 이동평균(7일) 성능 측정
  1-3 이동평균(14일) 성능 측정
  1-4 지수평활(Exponential Smoothing) 성능 측정
  1-5 알고리즘 종합 비교 및 SKU별 최적 알고리즘 매칭
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse, bias


class Group1AlgorithmComparison:
    def __init__(self, df: pd.DataFrame):
        """df: data.generate_sample_sales() 또는 load_sales_csv()의 결과."""
        self.df = df

    # ---- 공통 유틸 -----------------------------------------------------
    def _train_test(self, train_start, train_end, test_start, test_end):
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        test = self.df[(self.df["일"] >= test_start) & (self.df["일"] <= test_end)]
        return train, test

    def _eval(self, test: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
        rows = []
        for sku, g in test.groupby("SKU_ID"):
            rows.append(
                {
                    "SKU_ID": sku,
                    "MAPE": mape(g["판매_수량"], g[forecast_col]),
                    "RMSE": rmse(g["판매_수량"], g[forecast_col]),
                    "Bias": bias(g["판매_수량"], g[forecast_col]),
                }
            )
        return pd.DataFrame(rows)

    # ---- 1-1 선형추세 ----------------------------------------------------
    def exp_1_1_linear_trend(self, train_start="2026-04-01", train_end="2026-08-31",
                              test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 SKU별 일별 판매 이력, 학습/검증 구간
        OUTPUT: SKU별 선형추세 예측 시계열 + MAPE/RMSE/Bias"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test = test.copy()
        test["forecast"] = np.nan

        for sku, g_train in train.groupby("SKU_ID"):
            x = (g_train["일"] - g_train["일"].min()).dt.days.values
            y = g_train["판매_수량"].values
            if len(x) < 2:
                slope, intercept = 0.0, y.mean() if len(y) else 0.0
            else:
                slope, intercept = np.polyfit(x, y, 1)
            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            pred = np.clip(intercept + slope * x_test, 0, None)
            test.loc[mask, "forecast"] = pred

        metrics = self._eval(test, "forecast")
        return {"forecast": test, "metrics": metrics}

    # ---- 1-2 / 1-3 이동평균 ------------------------------------------------
    def _moving_average(self, window: int, train_start, train_end, test_start, test_end):
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test = test.copy()
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            last_n = g_train.sort_values("일")["판매_수량"].tail(window)
            level = last_n.mean() if len(last_n) else 0.0
            test.loc[test["SKU_ID"] == sku, "forecast"] = level
        metrics = self._eval(test, "forecast")
        return {"forecast": test, "metrics": metrics}

    def exp_1_2_moving_average_7d(self, train_start="2026-04-01", train_end="2026-08-31",
                                   test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: SKU별 일별 판매량, 윈도우 7일 / OUTPUT: 7일 이동평균 예측치 + 오차"""
        return self._moving_average(7, train_start, train_end, test_start, test_end)

    def exp_1_3_moving_average_14d(self, train_start="2026-04-01", train_end="2026-08-31",
                                    test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: SKU별 일별 판매량, 윈도우 14일 / OUTPUT: 14일 이동평균 예측치 + 오차"""
        return self._moving_average(14, train_start, train_end, test_start, test_end)

    # ---- 1-4 지수평활 -------------------------------------------------
    def exp_1_4_exponential_smoothing(self, train_start="2026-04-01", train_end="2026-08-31",
                                       test_start="2026-09-01", test_end="2026-09-30",
                                       alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        """INPUT: SKU별 일별 판매량, 평활계수 alpha 후보값
        OUTPUT: alpha별 MAPE/RMSE, SKU별 최적 alpha, 지수평활 예측치"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test = test.copy()
        test["forecast"] = np.nan
        best_alpha_per_sku = {}

        for sku, g_train in train.sort_values("일").groupby("SKU_ID"):
            y = g_train["판매_수량"].values
            best = None
            for a in alphas:
                level = y[0] if len(y) else 0.0
                for v in y[1:]:
                    level = a * v + (1 - a) * level
                # 학습 구간 마지막 레벨을 검증기간 상수 예측치로 사용, 학습 자체 재현오차로 alpha 선택
                in_sample_pred = []
                lvl = y[0] if len(y) else 0.0
                for v in y:
                    in_sample_pred.append(lvl)
                    lvl = a * v + (1 - a) * lvl
                score = mape(y, np.array(in_sample_pred))
                if best is None or score < best[0]:
                    best = (score, a, level)
            best_alpha_per_sku[sku] = best[1]
            test.loc[test["SKU_ID"] == sku, "forecast"] = best[2]

        metrics = self._eval(test, "forecast")
        metrics["best_alpha"] = metrics["SKU_ID"].map(best_alpha_per_sku)
        return {"forecast": test, "metrics": metrics, "best_alpha": best_alpha_per_sku}

    # ---- 1-5 종합 비교 ----------------------------------------------
    def exp_1_5_compare_all(self, **kwargs):
        """INPUT: 1-1~1-4 결과 / OUTPUT: 알고리즘별 종합 비교표 + SKU 특성별 최적 알고리즘 매핑"""
        r1 = self.exp_1_1_linear_trend(**kwargs)["metrics"].assign(algorithm="linear_trend")
        r2 = self.exp_1_2_moving_average_7d(**kwargs)["metrics"].assign(algorithm="ma_7d")
        r3 = self.exp_1_3_moving_average_14d(**kwargs)["metrics"].assign(algorithm="ma_14d")
        r4 = self.exp_1_4_exponential_smoothing(**kwargs)["metrics"][["SKU_ID", "MAPE", "RMSE", "Bias"]].assign(
            algorithm="exp_smoothing"
        )
        combined = pd.concat([r1, r2, r3, r4], ignore_index=True)

        best = combined.loc[combined.groupby("SKU_ID")["MAPE"].idxmin()][["SKU_ID", "algorithm", "MAPE"]]
        best = best.rename(columns={"algorithm": "best_algorithm", "MAPE": "best_MAPE"})

        sku_stats = (
            self.df.groupby("SKU_ID")["판매_수량"]
            .agg(mean_qty="mean", std_qty="std")
            .reset_index()
        )
        sku_stats["cv"] = sku_stats["std_qty"] / sku_stats["mean_qty"].replace(0, np.nan)
        mapping = best.merge(sku_stats, on="SKU_ID", how="left")
        return {"combined": combined, "best_per_sku": mapping}
