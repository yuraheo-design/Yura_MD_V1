"""
그룹 6. 학습기간(윈도우) 비교
  6-1 최근 1개월(8월)만 사용한 예측
  6-2 최근 2개월(7~8월) 사용한 예측
  6-3 최근 3개월(6~8월) 사용한 예측
  6-4 전체 5개월(4~8월) 사용한 예측
  6-5 학습기간별 오차 추세 분석
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse


class Group6TrainingWindow:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _forecast_with_window(self, train_start, train_end, test_start, test_end):
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        test = self.df[(self.df["일"] >= test_start) & (self.df["일"] <= test_end)].copy()
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            x = (g_train["일"] - g_train["일"].min()).dt.days.values
            y = g_train["판매_수량"].values
            if len(x) < 2:
                slope, intercept = 0.0, (y.mean() if len(y) else 0.0)
            else:
                slope, intercept = np.polyfit(x, y, 1)
            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            test.loc[mask, "forecast"] = np.clip(intercept + slope * x_test, 0, None)
        return test

    def _eval(self, test: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sku, g in test.groupby("SKU_ID"):
            rows.append({"SKU_ID": sku, "MAPE": mape(g["판매_수량"], g["forecast"]), "RMSE": rmse(g["판매_수량"], g["forecast"])})
        return pd.DataFrame(rows)

    # ---- 6-1 ~ 6-4 : 학습기간 길이별 ------------------------------------
    def exp_6_1_one_month(self, test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 8월 1개월치 데이터 / OUTPUT: 1개월 학습 예측치와 MAPE"""
        f = self._forecast_with_window("2026-08-01", "2026-08-31", test_start, test_end)
        return {"forecast": f, "metrics": self._eval(f)}

    def exp_6_2_two_months(self, test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 7~8월 2개월치 데이터 / OUTPUT: 2개월 학습 예측치와 MAPE, 1개월 대비 개선폭"""
        f = self._forecast_with_window("2026-07-01", "2026-08-31", test_start, test_end)
        base = self.exp_6_1_one_month(test_start, test_end)["metrics"]["MAPE"].mean()
        cur = self._eval(f)
        return {"forecast": f, "metrics": cur, "improvement_vs_1m": base - cur["MAPE"].mean()}

    def exp_6_3_three_months(self, test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 6~8월 3개월치 데이터 / OUTPUT: 3개월 학습 예측치와 MAPE, 2개월 대비 개선폭"""
        f = self._forecast_with_window("2026-06-01", "2026-08-31", test_start, test_end)
        base = self.exp_6_2_two_months(test_start, test_end)["metrics"]["MAPE"].mean()
        cur = self._eval(f)
        return {"forecast": f, "metrics": cur, "improvement_vs_2m": base - cur["MAPE"].mean()}

    def exp_6_4_five_months(self, test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 전체 5개월 데이터 / OUTPUT: 전체기간 학습 예측치와 MAPE, 성능 상한"""
        f = self._forecast_with_window("2026-04-01", "2026-08-31", test_start, test_end)
        base = self.exp_6_3_three_months(test_start, test_end)["metrics"]["MAPE"].mean()
        cur = self._eval(f)
        return {"forecast": f, "metrics": cur, "improvement_vs_3m": base - cur["MAPE"].mean()}

    # ---- 6-5 학습기간별 오차 추세 ----------------------------------------
    def exp_6_5_window_trend(self, test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 실험 6-1~6-4 MAPE 결과 / OUTPUT: 학습기간-오차 곡선, 최소 권장 기간"""
        m1 = self.exp_6_1_one_month(test_start, test_end)["metrics"]["MAPE"].mean()
        m2 = self.exp_6_2_two_months(test_start, test_end)["metrics"]["MAPE"].mean()
        m3 = self.exp_6_3_three_months(test_start, test_end)["metrics"]["MAPE"].mean()
        m4 = self.exp_6_4_five_months(test_start, test_end)["metrics"]["MAPE"].mean()

        curve = pd.DataFrame(
            [
                {"학습기간(개월)": 1, "MAPE": m1},
                {"학습기간(개월)": 2, "MAPE": m2},
                {"학습기간(개월)": 3, "MAPE": m3},
                {"학습기간(개월)": 5, "MAPE": m4},
            ]
        )
        # 직전 대비 개선폭이 가장 먼저 완만해지는(diff가 작아지는) 지점을 권장 최소 기간으로 추정
        curve["개선폭"] = -curve["MAPE"].diff()
        recommended_idx = curve["개선폭"].iloc[1:].idxmin() if len(curve) > 1 else 0
        recommended_months = curve.loc[recommended_idx, "학습기간(개월)"] if len(curve) else None
        return {"curve": curve, "recommended_min_months": recommended_months}
