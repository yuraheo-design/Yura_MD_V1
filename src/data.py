"""
데이터 로딩 / 샘플 데이터 생성 모듈

컬럼 스펙은 요청하신 양식과 동일하게 맞춤:
index number | Brand | Category | Sub Category | SKU ID# | Desc_KR | 일 | 판매 수량 | 주문 금액
(+ 행사 처리 실험(그룹2)을 위한 event_grade 컬럼 추가: NONE / A / S / SS)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    "Motorized": ["Blender", "Kitchen System"],
    "Floorcare": ["Stick Vacuum", "Robot Vacuum"],
    "Cooking": ["Air Fryer", "Multi Cooker"],
}

EVENT_GRADES = ["NONE", "A", "S", "SS"]
EVENT_UPLIFT = {"NONE": 1.0, "A": 1.8, "S": 2.6, "SS": 4.0}


def generate_sample_sales(
    n_skus: int = 24,
    start: str = "2026-04-01",
    end: str = "2026-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """실측 데이터가 없을 때 코드 실행/검증용으로 쓰는 합성 판매 데이터.

    SKU마다 일평균 판매량, 추세 기울기, 요일 패턴, 행사 민감도를 다르게 줘서
    실험(알고리즘/집계단위/계절성 등) 결과가 SKU별로 갈리도록 설계했다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    cats = list(CATEGORIES.keys())

    rows = []
    for i in range(1, n_skus + 1):
        sku_id = f"SKU{i:03d}"
        category = cats[i % len(cats)]
        sub_category = CATEGORIES[category][i % len(CATEGORIES[category])]
        brand = "SharkNinja"
        desc_kr = f"{category} {sub_category} 모델 {i:03d}"

        base_level = rng.uniform(0.5, 40)  # 저매출~고매출 SKU 혼재
        trend_per_day = rng.uniform(-0.01, 0.03) * (base_level / 10)
        weekday_effect = rng.normal(0, 0.15, size=7)
        weekday_effect -= weekday_effect.mean()
        promo_sensitivity = rng.uniform(0.6, 1.4)
        noise_scale = max(0.3, base_level * 0.25)

        # 이 SKU가 참여하는 행사일(약 8%) 배정, 등급은 랜덤
        is_event_day = rng.random(len(dates)) < 0.08
        event_grade = np.where(
            is_event_day,
            rng.choice(["A", "S", "SS"], size=len(dates), p=[0.6, 0.3, 0.1]),
            "NONE",
        )

        order_unit_price = rng.uniform(15000, 120000)

        for d_idx, (dt, grade) in enumerate(zip(dates, event_grade)):
            trend = base_level + trend_per_day * d_idx
            wd = weekday_effect[dt.weekday()]
            month_effect = 0.1 * np.sin((dt.month / 12) * 2 * np.pi)
            uplift = 1.0 + (EVENT_UPLIFT[grade] - 1.0) * promo_sensitivity
            mean_qty = max(0.0, trend * (1 + wd + month_effect) * uplift)
            qty = max(0, int(round(rng.normal(mean_qty, noise_scale * 0.5 + 0.3))))

            rows.append(
                {
                    "Brand": brand,
                    "Category": category,
                    "Sub_Category": sub_category,
                    "SKU_ID": sku_id,
                    "Desc_KR": desc_kr,
                    "일": dt.strftime("%Y-%m-%d"),
                    "판매_수량": qty,
                    "주문_금액": int(qty * order_unit_price),
                    "event_grade": grade,
                }
            )

    df = pd.DataFrame(rows)
    df.insert(0, "index", range(1, len(df) + 1))
    df["일"] = pd.to_datetime(df["일"])
    return df


def save_sample_csv(path: str, **kwargs) -> pd.DataFrame:
    df = generate_sample_sales(**kwargs)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def load_sales_csv(path: str) -> pd.DataFrame:
    """실제 판매 데이터 CSV 로더. 컬럼명은 generate_sample_sales와 동일해야 한다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["일"] = pd.to_datetime(df["일"])
    return df


def split_by_date(df: pd.DataFrame, train_start, train_end, test_start, test_end):
    train = df[(df["일"] >= train_start) & (df["일"] <= train_end)].copy()
    test = df[(df["일"] >= test_start) & (df["일"] <= test_end)].copy()
    return train, test
