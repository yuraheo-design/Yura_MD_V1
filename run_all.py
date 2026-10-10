"""
30개 function(그룹1~6, 각 1~5)을 전부 실행해 결과를 콘솔에 요약 출력하는 엔트리 포인트.

- 실제 판매 데이터가 없는 동안은 src/data.py의 generate_sample_sales()로 만든
  샘플 CSV(sample_data/sample_sales.csv)를 사용한다.
- 실제 데이터가 생기면 data.load_sales_csv("실제파일.csv")로 교체하면 된다
  (컬럼 스펙: index, Brand, Category, Sub_Category, SKU_ID, Desc_KR, 일, 판매_수량, 주문_금액[, event_grade]).

실행:
    python run_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import generate_sample_sales, save_sample_csv
from src.algorithms.group1_algorithm_comparison import Group1AlgorithmComparison
from src.algorithms.group2_promotion_baseline import Group2PromotionBaseline
from src.algorithms.group3_aggregation_level import Group3AggregationLevel
from src.algorithms.group4_seasonality import Group4Seasonality
from src.algorithms.group5_validation import Group5Validation
from src.algorithms.group6_training_window import Group6TrainingWindow


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    sample_csv_path = Path(__file__).parent / "sample_data" / "sample_sales.csv"
    sample_csv_path.parent.mkdir(exist_ok=True)
    df = save_sample_csv(str(sample_csv_path))
    print(f"[Sample CSV] {sample_csv_path}  ({len(df)} rows, {df['SKU_ID'].nunique()} SKUs)")

    kwargs = dict(train_start="2026-04-01", train_end="2026-08-31",
                  test_start="2026-09-01", test_end="2026-09-30")

    # ---- 그룹 1: 알고리즘 비교 -------------------------------------
    section("그룹 1. 예측 알고리즘 비교 (1-1 ~ 1-5)")
    g1 = Group1AlgorithmComparison(df)
    r11 = g1.exp_1_1_linear_trend(**kwargs)
    print("1-1 선형추세 평균 MAPE:", round(r11["metrics"]["MAPE"].mean(), 2))
    r12 = g1.exp_1_2_moving_average_7d(**kwargs)
    print("1-2 이동평균(7일) 평균 MAPE:", round(r12["metrics"]["MAPE"].mean(), 2))
    r13 = g1.exp_1_3_moving_average_14d(**kwargs)
    print("1-3 이동평균(14일) 평균 MAPE:", round(r13["metrics"]["MAPE"].mean(), 2))
    r14 = g1.exp_1_4_exponential_smoothing(**kwargs)
    print("1-4 지수평활 평균 MAPE:", round(r14["metrics"]["MAPE"].mean(), 2))
    r15 = g1.exp_1_5_compare_all(**kwargs)
    print("1-5 SKU별 최적 알고리즘 분포:\n", r15["best_per_sku"]["best_algorithm"].value_counts())

    # ---- 그룹 2: 행사 처리 ---------------------------------------
    section("그룹 2. 행사/비행사 처리 비교 (2-1 ~ 2-5)")
    g2 = Group2PromotionBaseline(df)
    r25 = g2.exp_2_5_final_baseline_choice(**kwargs)
    print(r25["summary"].to_string(index=False))
    print("권장 방식:", r25["recommended"])

    # ---- 그룹 3: 집계 단위 ---------------------------------------
    section("그룹 3. 집계 단위 비교 (3-1 ~ 3-5)")
    g3 = Group3AggregationLevel(df)
    r35 = g3.exp_3_5_best_unit_by_volume(**kwargs)
    print(r35["cross_table"])
    print("구간별 권장 단위:\n", r35["recommendation"])

    # ---- 그룹 4: 계절성 ----------------------------------------
    section("그룹 4. 계절성 반영 비교 (4-1 ~ 4-5)")
    g4 = Group4Seasonality(df)
    r45 = g4.exp_4_5_compare(**kwargs)
    print(r45["summary"].to_string(index=False))
    print("권장 구성:", r45["recommended"])

    # ---- 그룹 5: 실측 검증 (그룹4 최종 모델: 요일가중치 적용 버전 사용) ----
    section("그룹 5. 실측 검증 (5-1 ~ 5-5)")
    g5 = Group5Validation(df, lambda *a, **k: g4.exp_4_2_weekday_weight(*a, **k)["forecast"])
    r55 = g5.exp_5_5_final_recommendation(**kwargs)
    print("전체 평균 MAPE:", round(r55["overall_MAPE"], 2))
    print("도입 판단:", r55["decision"])
    print("자동 예측 적용 가능 SKU 수:", len(r55["auto_forecast_skus"]),
          "/ 수작업 병행 SKU 수:", len(r55["manual_review_skus"]))

    # ---- 그룹 6: 학습기간 --------------------------------------
    section("그룹 6. 학습기간(윈도우) 비교 (6-1 ~ 6-5)")
    g6 = Group6TrainingWindow(df)
    r65 = g6.exp_6_5_window_trend(test_start="2026-09-01", test_end="2026-09-30")
    print(r65["curve"].to_string(index=False))
    print("권장 최소 학습기간(개월):", r65["recommended_min_months"])

    section("완료 — 30개 function 전체 실행 성공")


if __name__ == "__main__":
    main()
