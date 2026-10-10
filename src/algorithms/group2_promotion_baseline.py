"""
그룹 2. 행사/비행사 구분 반영 여부 비교
  2-1 행사일 포함 전체 데이터로 추세선 학습(현재 단순 버전)
  2-2 비행사(NONE) 데이터만으로 추세선 학습
  2-3 행사등급별(SS/S/A) 업리프트 배수 적용
  2-4 행사등급 세분화(SS/S/A) vs 단순 행사여부(Y/N) 비교
  2-5 종합 비교 및 최종 베이스라인 방식 채택
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse, bias


class Group2PromotionBaseline:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _train_test(self, train_start, train_end, test_start, test_end):
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        test = self.df[(self.df["일"] >= test_start) & (self.df["일"] <= test_end)]
        return train.copy(), test.copy()

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

    def _fit_trend(self, g_train: pd.DataFrame):
        x = (g_train["일"] - g_train["일"].min()).dt.days.values
        y = g_train["판매_수량"].values
        if len(x) < 2:
            return 0.0, (y.mean() if len(y) else 0.0)
        return tuple(np.polyfit(x, y, 1))

    # ---- 2-1 행사 미구분(전체 포함) --------------------------------------
    def exp_2_1_include_events(self, train_start="2026-04-01", train_end="2026-08-31",
                                test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 4~8월 전체 일별 판매량(행사 구분 없음)
        OUTPUT: 행사 미구분 추세선 예측치, MAPE/RMSE, 행사일/비행사일 오차 분리"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        test["forecast"] = np.nan
        for sku, g_train in train.groupby("SKU_ID"):
            slope, intercept = self._fit_trend(g_train)
            mask = test["SKU_ID"] == sku
            x_test = (test.loc[mask, "일"] - g_train["일"].min()).dt.days.values
            test.loc[mask, "forecast"] = np.clip(intercept + slope * x_test, 0, None)

        metrics = self._eval(test, "forecast")
        by_event = (
            test.assign(is_event=test["event_grade"] != "NONE")
            .groupby("is_event")
            .apply(lambda g: pd.Series({"MAPE": mape(g["판매_수량"], g["forecast"])}))
            .reset_index()
        )
        return {"forecast": test, "metrics": metrics, "metrics_by_event_flag": by_event}

    # ---- 2-2 비행사만 학습 ------------------------------------------------
    def exp_2_2_non_event_only(self, train_start="2026-04-01", train_end="2026-08-31",
                                test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 행사등급 = NONE 인 날짜만 필터링한 데이터
        OUTPUT: 비행사 기반 베이스라인, 비행사일 오차, 행사일 과소예측 폭"""
        train, test = self._train_test(train_start, train_end, test_start, test_end)
        train_non = train[train["event_grade"] == "NONE"]
        test["forecast"] = np.nan
        for sku, g_train in train_non.groupby("SKU_ID"):
            slope, intercept = self._fit_trend(g_train)
            mask = test["SKU_ID"] == sku
            base_date = g_train["일"].min()
            x_test = (test.loc[mask, "일"] - base_date).dt.days.values
            test.loc[mask, "forecast"] = np.clip(intercept + slope * x_test, 0, None)

        metrics = self._eval(test, "forecast")
        event_days = test[test["event_grade"] != "NONE"]
        underforecast = (event_days["판매_수량"] - event_days["forecast"]).mean() if len(event_days) else np.nan
        return {"forecast": test, "metrics": metrics, "event_day_underforecast": underforecast}

    # ---- 2-3 등급별 업리프트 -------------------------------------------
    def exp_2_3_grade_uplift(self, train_start="2026-04-01", train_end="2026-08-31",
                              test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 비행사 베이스라인, 등급별(SS/S/A) 과거 평균 증폭 배수
        OUTPUT: 등급별 업리프트 적용 예측치, 행사일 오차율, 등급별 배수 추정값"""
        baseline = self.exp_2_2_non_event_only(train_start, train_end, test_start, test_end)
        train, _ = self._train_test(train_start, train_end, test_start, test_end)

        # SKU x 등급별 과거 업리프트 배수 = 행사일 평균 판매량 / 비행사일 평균 판매량
        uplift_table = {}
        for sku, g in train.groupby("SKU_ID"):
            non_event_mean = g.loc[g["event_grade"] == "NONE", "판매_수량"].mean() or 1e-6
            uplift_table[sku] = {}
            for grade in ["A", "S", "SS"]:
                grade_mean = g.loc[g["event_grade"] == grade, "판매_수량"].mean()
                uplift_table[sku][grade] = (grade_mean / non_event_mean) if pd.notna(grade_mean) else 1.0

        test = baseline["forecast"].copy()
        def apply_uplift(row):
            if row["event_grade"] == "NONE":
                return row["forecast"]
            mult = uplift_table.get(row["SKU_ID"], {}).get(row["event_grade"], 1.0)
            return row["forecast"] * mult

        test["forecast_uplifted"] = test.apply(apply_uplift, axis=1)
        metrics = self._eval(test.rename(columns={"forecast": "_base"}).rename(
            columns={"forecast_uplifted": "forecast"}
        ), "forecast")
        return {"forecast": test, "metrics": metrics, "uplift_table": uplift_table}

    # ---- 2-4 세분화 vs 단순 Y/N -------------------------------------
    def exp_2_4_grade_vs_binary(self, train_start="2026-04-01", train_end="2026-08-31",
                                 test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 등급 세분화 예측결과, 단순 Y/N 예측결과
        OUTPUT: 두 방식 MAPE 차이, 등급별 오차 분해"""
        graded = self.exp_2_3_grade_uplift(train_start, train_end, test_start, test_end)
        graded_mape = mape(graded["forecast"]["판매_수량"], graded["forecast"]["forecast_uplifted"])

        # 단순 Y/N: 행사여부만으로 단일 업리프트 배수 적용
        train, _ = self._train_test(train_start, train_end, test_start, test_end)
        binary_uplift = {}
        for sku, g in train.groupby("SKU_ID"):
            non_event_mean = g.loc[g["event_grade"] == "NONE", "판매_수량"].mean() or 1e-6
            event_mean = g.loc[g["event_grade"] != "NONE", "판매_수량"].mean()
            binary_uplift[sku] = (event_mean / non_event_mean) if pd.notna(event_mean) else 1.0

        base = self.exp_2_2_non_event_only(train_start, train_end, test_start, test_end)["forecast"].copy()
        base["forecast_binary"] = base.apply(
            lambda r: r["forecast"] * binary_uplift.get(r["SKU_ID"], 1.0) if r["event_grade"] != "NONE" else r["forecast"],
            axis=1,
        )
        binary_mape = mape(base["판매_수량"], base["forecast_binary"])

        return {
            "graded_MAPE": graded_mape,
            "binary_MAPE": binary_mape,
            "diff": graded_mape - binary_mape,
        }

    # ---- 2-5 종합 비교 ----------------------------------------------
    def exp_2_5_final_baseline_choice(self, **kwargs):
        """INPUT: 실험 2-1~2-4 오차 결과 / OUTPUT: 4개 방식 종합 비교표, 최종 채택 방식"""
        m1 = mape_from(self.exp_2_1_include_events(**kwargs)["forecast"])
        m2 = mape_from(self.exp_2_2_non_event_only(**kwargs)["forecast"])
        graded = self.exp_2_3_grade_uplift(**kwargs)
        m3 = mape(graded["forecast"]["판매_수량"], graded["forecast"]["forecast_uplifted"])
        bv = self.exp_2_4_grade_vs_binary(**kwargs)

        summary = pd.DataFrame(
            [
                {"방식": "행사 포함 전체학습", "MAPE": m1},
                {"방식": "비행사만 학습", "MAPE": m2},
                {"방식": "비행사+등급별 업리프트", "MAPE": m3},
                {"방식": "비행사+단순 Y/N 업리프트", "MAPE": bv["binary_MAPE"]},
            ]
        ).sort_values("MAPE")
        return {"summary": summary, "recommended": summary.iloc[0]["방식"]}


def mape_from(df: pd.DataFrame) -> float:
    return mape(df["판매_수량"], df["forecast"])
