"""
그룹 5. 실측 검증 (예측치 vs 실제 9~12월 판매량)
  5-1 9월 예측치 vs 실제 판매량 오차 산출
  5-2 예측 시계(9→12월)에 따른 오차 변화 확인
  5-3 오차가 큰 SKU의 원인 분석
  5-4 예측 기반 발주량 vs 실제 필요 발주량 비교
  5-5 실무 도입 여부 최종 결론
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import mape, rmse, bias, mae


class Group5Validation:
    def __init__(self, df: pd.DataFrame, final_model_forecaster):
        """final_model_forecaster: 일자 범위를 받아 SKU별 forecast를 채운 DataFrame을 돌려주는 콜러블.
        (그룹 1~4 실험에서 최종 채택된 모델의 forecast 함수를 넣어준다.)"""
        self.df = df
        self.forecast = final_model_forecaster

    # ---- 5-1 9월 오차 산출 ---------------------------------------------
    def exp_5_1_september_error(self, train_start="2026-04-01", train_end="2026-08-31",
                                 test_start="2026-09-01", test_end="2026-09-30"):
        """INPUT: 9월 SKU별 일별 예측치, 9월 실측 일별 판매량 / OUTPUT: SKU별 MAPE/MAE/Bias, 오차 분포"""
        result = self.forecast(train_start, train_end, test_start, test_end)
        rows = []
        for sku, g in result.groupby("SKU_ID"):
            rows.append(
                {
                    "SKU_ID": sku,
                    "MAPE": mape(g["판매_수량"], g["forecast"]),
                    "MAE": mae(g["판매_수량"], g["forecast"]),
                    "Bias": bias(g["판매_수량"], g["forecast"]),
                }
            )
        table = pd.DataFrame(rows)
        return {"forecast": result, "metrics": table, "mape_distribution": table["MAPE"].describe()}

    # ---- 5-2 예측 시계별 오차 변화 ---------------------------------------
    def exp_5_2_horizon_error(self, train_start="2026-04-01", train_end="2026-08-31",
                               horizons=(("2026-09-01", "2026-09-30", "9월"),
                                         ("2026-10-01", "2026-10-31", "10월"),
                                         ("2026-11-01", "2026-11-30", "11월"),
                                         ("2026-12-01", "2026-12-31", "12월"))):
        """INPUT: 9~12월 예측치/실측치, 기준일로부터 경과일수 / OUTPUT: 예측 시계별 MAPE 추이"""
        rows = []
        for start, end, label in horizons:
            result = self.forecast(train_start, train_end, start, end)
            rows.append({"horizon": label, "MAPE": mape(result["판매_수량"], result["forecast"])})
        return pd.DataFrame(rows)

    # ---- 5-3 오차 원인 분석 ---------------------------------------------
    def exp_5_3_error_root_cause(self, sku_metrics: pd.DataFrame, top_n: int = 5,
                                  stockout_qty_threshold: int = 0):
        """INPUT: 오차율 상위 SKU 목록, 품절/신제품/프로모션 이력
        OUTPUT: 오차 원인 분류표(품절/신제품/프로모션/순수모델오차) 및 비중

        실측 품절/신제품 이력 컬럼이 없는 합성 데이터에서는, 실측 판매량이 0에 가까운(=품절 추정)
        구간 비중으로 근사치를 계산한다. 실제 데이터 연결 시 stockout_flag 등 실제 컬럼으로 교체.
        """
        top_error_skus = sku_metrics.sort_values("MAPE", ascending=False).head(top_n)["SKU_ID"].tolist()
        sub = self.df[self.df["SKU_ID"].isin(top_error_skus)]
        near_zero_ratio = (sub["판매_수량"] <= stockout_qty_threshold).mean() if len(sub) else np.nan
        return {
            "top_error_skus": top_error_skus,
            "approx_stockout_day_ratio": near_zero_ratio,
            "note": "실 운영 데이터 연결 시 품절/신제품 등록/프로모션 이력 테이블과 조인해 정확히 분류할 것",
        }

    # ---- 5-4 발주량 비교 -------------------------------------------
    def exp_5_4_order_qty_comparison(self, train_start="2026-04-01", train_end="2026-08-31",
                                      test_start="2026-09-01", test_end="2026-09-30",
                                      safety_stock_days: int = 3):
        """INPUT: 예측 기반 권장 발주량, 실제 발주량(=과거 평균 기반 추정), 실측 판매량
        OUTPUT: 방식별 과잉재고/품절 건수 및 비율"""
        result = self.forecast(train_start, train_end, test_start, test_end)
        train = self.df[(self.df["일"] >= train_start) & (self.df["일"] <= train_end)]
        naive_order = train.groupby("SKU_ID")["판매_수량"].mean() * safety_stock_days

        rows = []
        for sku, g in result.groupby("SKU_ID"):
            actual_total = g["판매_수량"].sum()
            model_order = g["forecast"].sum()
            naive_total = naive_order.get(sku, 0) * (len(g) / safety_stock_days)
            rows.append(
                {
                    "SKU_ID": sku,
                    "actual_sales": actual_total,
                    "model_order_qty": model_order,
                    "naive_order_qty": naive_total,
                    "model_gap": model_order - actual_total,
                    "naive_gap": naive_total - actual_total,
                }
            )
        comp = pd.DataFrame(rows)
        comp["model_overstock"] = comp["model_gap"] > 0
        comp["naive_overstock"] = comp["naive_gap"] > 0
        return comp

    # ---- 5-5 최종 결론 ----------------------------------------------
    def exp_5_5_final_recommendation(self, mape_threshold: float = 30.0, **kwargs):
        """INPUT: 실험 5-1~5-4 결과 / OUTPUT: 도입 가능 구간, 제약조건, 추가 개선 과제"""
        sep = self.exp_5_1_september_error(**kwargs)
        horizon = self.exp_5_2_horizon_error()
        overall_mape = sep["metrics"]["MAPE"].mean()

        go_skus = sep["metrics"].loc[sep["metrics"]["MAPE"] <= mape_threshold, "SKU_ID"].tolist()
        manual_skus = sep["metrics"].loc[sep["metrics"]["MAPE"] > mape_threshold, "SKU_ID"].tolist()

        decision = "조건부 도입" if overall_mape <= mape_threshold * 1.5 else "추가 개선 필요"
        return {
            "overall_MAPE": overall_mape,
            "horizon_table": horizon,
            "decision": decision,
            "auto_forecast_skus": go_skus,
            "manual_review_skus": manual_skus,
        }
