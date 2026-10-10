"""
그룹 4. 요일/월 계절성 반영 비교
  4-1 요일 가중치 미적용
  4-2 요일 가중치 적용(현재 방식)
  4-3 주중/주말 이진 구분만 적용
  4-4 월별 계절성 추가 적용
  4-5 요일+월 계절성 결합 버전과 단순 버전 비교
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse


class Group4Seasonality:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _train_test(self, train_start, train_end, test_start, test_end):
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        test = self.df[(self.df["일"] >= test_start) & (self.df["일"] <= test_end)]
        return train.copy(), test.copy()

    def _eval(self, test: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
        rows = []
        for sku, g in test.groupby("SKU_ID"):
            rows.append({"SKU_ID": sku, "MAPE": mape(g["판매_수량"], g[forecast_col]), "RMSE": rmse(g["판매_수량"], g[forecast_col])})
        return pd.DataFrame(rows)

    def _base_trend(self, g_train: pd.DataFrame):
        x = (g_train["일"] - g_train["일"].min()).dt.days.values
        y = g_train["판매_수량"].values
        if len(x) < 2:
            return 0.0, (y.mean() if len(y) else 0.0)
        return tuple(np.polyfit(x, y, 1))

    # ---- 4-1 요일 미적용(순수 추세) -------------------------------------
    def exp_4_1_no_weekday_weight(self, train_start="2026-04-01", train_end="2026-08-31",
                                   test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 일별 판매량(요일 정보 미사용) / OUTPUT: 순수 추세 예측치, MAPE/RMSE, 요일별 잔차"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            slope, intercept = self._base_trend(g_train)
            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            test.loc[mask, "forecast"] = np.clip(intercept + slope * x_test, 0, None)

        test["residual"] = test["판매_수량"] - test["forecast"]
        test["weekday"] = test["일"].dt.weekday
        resid_by_weekday = test.groupby("weekday")["residual"].mean()
        metrics = self._eval(test, "forecast")
        return {"forecast": test, "metrics": metrics, "residual_by_weekday": resid_by_weekday}

    # ---- 4-2 요일 가중치 적용(현재 방식, 7단계) --------------------------
    def exp_4_2_weekday_weight(self, train_start="2026-04-01", train_end="2026-08-31",
                                test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 일별 판매량, SKU별 요일 계수(7단계) / OUTPUT: 요일 보정 예측치, MAPE/RMSE, 개선폭"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        base = self.exp_4_1_no_weekday_weight(train_start, train_end, test_start, test_end)
        test["forecast_trend"] = base["forecast"]["forecast"].values

        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            slope, intercept = self._base_trend(g_train)
            g_train = g_train.copy()
            x = (g_train["일"] - g_train["일"].min()).dt.days.values
            g_train["trend_pred"] = np.clip(intercept + slope * x, 0, None)
            g_train["weekday"] = g_train["일"].dt.weekday
            wd_factor = (
                g_train.groupby("weekday").apply(
                    lambda g: (g["판매_수량"] / g["trend_pred"].replace(0, np.nan)).mean()
                )
            ).fillna(1.0)

            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            trend_pred = np.clip(intercept + slope * x_test, 0, None)
            wd_test = test.loc[mask, "일"].dt.weekday.map(wd_factor).fillna(1.0).values
            test.loc[mask, "forecast"] = trend_pred * wd_test

        metrics = self._eval(test, "forecast")
        improvement = base["metrics"]["MAPE"].mean() - metrics["MAPE"].mean()
        return {"forecast": test, "metrics": metrics, "improvement_vs_no_weekday": improvement}

    # ---- 4-3 주중/주말 2단계 -------------------------------------------
    def exp_4_3_weekday_weekend_binary(self, train_start="2026-04-01", train_end="2026-08-31",
                                        test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 일별 판매량, 주중/주말 이진 라벨 / OUTPUT: 2단계 계수 예측치, 7단계 대비 오차 차이"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            slope, intercept = self._base_trend(g_train)
            g_train = g_train.copy()
            x = (g_train["일"] - g_train["일"].min()).dt.days.values
            g_train["trend_pred"] = np.clip(intercept + slope * x, 0, None)
            g_train["is_weekend"] = g_train["일"].dt.weekday >= 5
            factor = (
                g_train.groupby("is_weekend").apply(
                    lambda g: (g["판매_수량"] / g["trend_pred"].replace(0, np.nan)).mean()
                )
            ).fillna(1.0)

            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            trend_pred = np.clip(intercept + slope * x_test, 0, None)
            is_we = (test.loc[mask, "일"].dt.weekday >= 5)
            f_test = is_we.map(factor).fillna(1.0).values
            test.loc[mask, "forecast"] = trend_pred * f_test

        metrics = self._eval(test, "forecast")
        seven_stage = self.exp_4_2_weekday_weight(train_start, train_end, test_start, test_end)["metrics"]
        diff = metrics["MAPE"].mean() - seven_stage["MAPE"].mean()
        return {"forecast": test, "metrics": metrics, "mape_diff_vs_7stage": diff}

    # ---- 4-4 월별 계절성 추가 ------------------------------------------
    def exp_4_4_add_monthly(self, train_start="2026-04-01", train_end="2026-08-31",
                             test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 일별 판매량, 요일 계수, 월별 계수 / OUTPUT: 요일+월 보정 예측치, MAPE, 월별 계수 유의성"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        weekday_result = self.exp_4_2_weekday_weight(train_start, train_end, test_start, test_end)
        test = weekday_result["forecast"].copy()

        monthly_factor_per_sku = {}
        for sku, g_train in train.groupby("SKU_ID"):
            g_train = g_train.copy()
            monthly_mean = g_train.groupby(g_train["일"].dt.month)["판매_수량"].mean()
            overall_mean = g_train["판매_수량"].mean() or 1e-6
            monthly_factor = (monthly_mean / overall_mean).to_dict()
            monthly_factor_per_sku[sku] = monthly_factor

            mask = test["SKU_ID"] == sku
            months = test.loc[mask, "일"].dt.month
            factor = months.map(monthly_factor).fillna(1.0).values
            test.loc[mask, "forecast"] = test.loc[mask, "forecast"] * factor

        metrics = self._eval(test, "forecast")
        return {"forecast": test, "metrics": metrics, "monthly_factor_per_sku": monthly_factor_per_sku}

    # ---- 4-5 종합 비교 ----------------------------------------------
    def exp_4_5_compare(self, **kwargs):
        """INPUT: 실험 4-1~4-4 오차 결과 / OUTPUT: 계절성 수준별 종합 비교표, 최적 조합 선정"""
        m1 = self.exp_4_1_no_weekday_weight(**kwargs)["metrics"]["MAPE"].mean()
        m2 = self.exp_4_2_weekday_weight(**kwargs)["metrics"]["MAPE"].mean()
        m3 = self.exp_4_3_weekday_weekend_binary(**kwargs)["metrics"]["MAPE"].mean()
        m4 = self.exp_4_4_add_monthly(**kwargs)["metrics"]["MAPE"].mean()

        summary = pd.DataFrame(
            [
                {"구성": "요일 미적용", "MAPE": m1},
                {"구성": "요일 7단계", "MAPE": m2},
                {"구성": "주중/주말 2단계", "MAPE": m3},
                {"구성": "요일7단계+월별", "MAPE": m4},
            ]
        ).sort_values("MAPE")
        return {"summary": summary, "recommended": summary.iloc[0]["구성"]}
