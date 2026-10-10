"""
그룹 3. 집계 단위 비교 (SKU vs Category vs Sub Category)
  3-1 SKU 단위 개별 예측(현재 방식)
  3-2 Category 단위 예측 후 재분배
  3-3 Sub Category 단위 예측 후 재분배
  3-4 저매출 SKU만 별도로 카테고리 단위 적용 (하이브리드)
  3-5 매출 규모 구간별 최적 집계 단위 도출
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse


class Group3AggregationLevel:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _train_test(self, train_start, train_end, test_start, test_end):
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        test = self.df[(self.df["일"] >= test_start) & (self.df["일"] <= test_end)]
        return train.copy(), test.copy()

    def _fit_trend_forecast(self, g_train: pd.DataFrame, test_dates: pd.Series):
        x = (g_train["일"] - g_train["일"].min()).dt.days.values
        y = g_train["판매_수량"].values
        if len(x) < 2:
            slope, intercept = 0.0, (y.mean() if len(y) else 0.0)
        else:
            slope, intercept = np.polyfit(x, y, 1)
        x_test = (test_dates - g_train["일"].min()).dt.days.values
        return np.clip(intercept + slope * x_test, 0, None)

    def _sku_mape_table(self, test: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
        rows = []
        for sku, g in test.groupby("SKU_ID"):
            rows.append({"SKU_ID": sku, "MAPE": mape(g["판매_수량"], g[forecast_col]), "RMSE": rmse(g["판매_수량"], g[forecast_col])})
        return pd.DataFrame(rows)

    # ---- 3-1 SKU 단위(현재 방식) --------------------------------------
    def exp_3_1_sku_level(self, train_start="2026-04-01", train_end="2026-08-31",
                           test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: SKU별 일별 판매량(개별 시계열) / OUTPUT: SKU별 개별 예측치, MAPE, 판매량 구간별 오차 분포"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            mask = test["SKU_ID"] == sku
            test.loc[mask, "forecast"] = self._fit_trend_forecast(g_train, test.loc[mask, "일"])
        table = self._sku_mape_table(test, "forecast")
        return {"forecast": test, "metrics": table}

    # ---- 3-2 / 3-3 상위 단위 예측 후 재분배 -----------------------------
    def _top_down(self, group_col: str, train_start, train_end, test_start, test_end):
        train, test = self._train_test(train_start, train_end, test_start, test_end)

        # 상위 단위(Category or Sub_Category) 합산 시계열 학습/예측
        group_forecast = {}
        for grp, g_train in train.groupby(group_col):
            daily = g_train.groupby("일")["판매_수량"].sum().reset_index()
            x = (daily["일"] - daily["일"].min()).dt.days.values
            y = daily["판매_수량"].values
            slope, intercept = (np.polyfit(x, y, 1) if len(x) >= 2 else (0.0, y.mean() if len(y) else 0.0))
            test_dates = test.loc[test[group_col] == grp, "일"].drop_duplicates().sort_values()
            x_test = (test_dates - daily["일"].min()).dt.days.values
            group_forecast[grp] = pd.Series(np.clip(intercept + slope * x_test, 0, None), index=test_dates)

        # SKU별 과거 판매비중(그룹 내)으로 재분배
        share = (
            train.groupby([group_col, "SKU_ID"])["판매_수량"].sum()
            / train.groupby(group_col)["판매_수량"].sum()
        ).rename("share").reset_index()

        test = test.merge(share, on=[group_col, "SKU_ID"], how="left")
        test["share"] = test["share"].fillna(1.0 / test.groupby(group_col)["SKU_ID"].transform("nunique"))

        def lookup(row):
            gf = group_forecast.get(row[group_col])
            grp_val = gf.get(row["일"], np.nan) if gf is not None else np.nan
            return grp_val * row["share"]

        test["forecast"] = test.apply(lookup, axis=1)
        table = self._sku_mape_table(test, "forecast")
        return {"forecast": test, "metrics": table}

    def exp_3_2_category_level(self, train_start="2026-04-01", train_end="2026-08-31",
                                test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: Category 단위 합산 일별 판매량, 과거 SKU별 판매비중
        OUTPUT: Category 예측치, 비중 분배 후 SKU별 예측치, SKU별 MAPE"""
        return self._top_down("Category", train_start, train_end, test_start, test_end)

    def exp_3_3_sub_category_level(self, train_start="2026-04-01", train_end="2026-08-31",
                                    test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: Sub Category 단위 합산 일별 판매량, 그룹 내 SKU 판매비중
        OUTPUT: Sub Category 예측치, 재분배 SKU 예측치, MAPE(Category 방식 대비 차이 포함)"""
        return self._top_down("Sub_Category", train_start, train_end, test_start, test_end)

    # ---- 3-4 저매출 SKU만 하이브리드 적용 --------------------------------
    def exp_3_4_hybrid_low_volume(self, low_volume_threshold: float = 1.0, **kwargs):
        """INPUT: 일평균 판매량 임계값 기준 저매출 SKU 목록, 두 방식 예측치
        OUTPUT: 하이브리드 적용 전후 MAPE, 임계값별 개선폭"""
        sku_sku = self.exp_3_1_sku_level(**kwargs)
        sku_cat = self.exp_3_2_category_level(**kwargs)

        daily_mean = self.df.groupby("SKU_ID")["판매_수량"].mean()
        low_volume_skus = daily_mean[daily_mean <= low_volume_threshold].index.tolist()

        sku_level_metrics = sku_sku["metrics"].set_index("SKU_ID")["MAPE"]
        cat_level_metrics = sku_cat["metrics"].set_index("SKU_ID")["MAPE"]

        hybrid_mape = sku_level_metrics.copy()
        hybrid_mape.loc[hybrid_mape.index.isin(low_volume_skus)] = cat_level_metrics.loc[
            cat_level_metrics.index.isin(low_volume_skus)
        ]

        return {
            "low_volume_skus": low_volume_skus,
            "sku_level_overall_MAPE": sku_level_metrics.mean(),
            "hybrid_overall_MAPE": hybrid_mape.mean(),
            "improvement": sku_level_metrics.mean() - hybrid_mape.mean(),
        }

    # ---- 3-5 매출 구간별 최적 집계 단위 -------------------------------
    def exp_3_5_best_unit_by_volume(self, **kwargs):
        """INPUT: 실험 3-1~3-4 결과, SKU별 매출 구간(고/중/저)
        OUTPUT: 매출 구간 x 집계 단위 MAPE 교차표, 구간별 권장 단위"""
        sku_level = self.exp_3_1_sku_level(**kwargs)["metrics"].set_index("SKU_ID")["MAPE"]
        cat_level = self.exp_3_2_category_level(**kwargs)["metrics"].set_index("SKU_ID")["MAPE"]
        subcat_level = self.exp_3_3_sub_category_level(**kwargs)["metrics"].set_index("SKU_ID")["MAPE"]

        daily_mean = self.df.groupby("SKU_ID")["판매_수량"].mean()
        tiers = pd.qcut(daily_mean, q=3, labels=["저매출", "중매출", "고매출"])

        combo = pd.DataFrame(
            {"SKU": sku_level, "Category": cat_level, "Sub_Category": subcat_level, "tier": tiers}
        ).dropna()
        cross = combo.groupby("tier")[["SKU", "Category", "Sub_Category"]].mean()
        recommendation = cross.idxmin(axis=1)
        return {"cross_table": cross, "recommendation": recommendation}
