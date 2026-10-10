# Yura_MD_V1

SKU 판매 예측(발주량 산출) 실험 설계를 위한 30개 function 명세 MD 파일 + 각 function을 구현한 Python 코드.

## 구성

```
functions/        - 30개 MD 명세 파일 (INPUT/OUTPUT/PURPOSE/Background/Intent/Result/HARNESS)
src/
  data.py          - 샘플 판매 데이터 생성기 / CSV 로더
  metrics.py       - MAPE, WAPE, RMSE, Bias 등 오차 지표
  algorithms/
    group1_algorithm_comparison.py   - 1-1~1-5 예측 알고리즘 비교 (선형추세/이동평균/지수평활)
    group2_promotion_baseline.py     - 2-1~2-5 행사/비행사 구분 반영 비교
    group3_aggregation_level.py      - 3-1~3-5 SKU/Category/Sub Category 집계 단위 비교
    group4_seasonality.py            - 4-1~4-5 요일/월 계절성 반영 비교
    group5_validation.py             - 5-1~5-5 실측(9~12월) 검증 및 실무 도입 판단
    group6_training_window.py        - 6-1~6-5 학습기간(윈도우 길이) 비교
run_all.py         - 30개 function을 전부 실행해 결과를 콘솔에 요약 출력하는 엔트리 포인트
sample_data/       - run_all.py 실행 시 생성되는 샘플 판매 CSV
requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt
python run_all.py
```

실제 판매 데이터가 아직 없어서, 지금은 `src/data.py`의 `generate_sample_sales()`가 만드는
합성 데이터로 전체 파이프라인이 끝까지 도는지 검증한 상태입니다. 실제 데이터가 준비되면
`src/data.load_sales_csv("실제파일.csv")`로 교체하면 됩니다. CSV 컬럼 스펙:

```
index | Brand | Category | Sub_Category | SKU_ID | Desc_KR | 일 | 판매_수량 | 주문_금액 [| event_grade]
```
(`event_grade`: NONE / A / S / SS — 행사 처리 실험(그룹2)에 사용)

## 함수 목록

- **그룹 1 (1-1~1-5)**: 선형추세 / 이동평균(7일·14일) / 지수평활 성능 비교, SKU별 최적 알고리즘 매칭
- **그룹 2 (2-1~2-5)**: 행사일 포함 전체학습 vs 비행사만 학습 vs 등급별(SS/S/A) 업리프트 vs 단순 Y/N, 최종 베이스라인 채택
- **그룹 3 (3-1~3-5)**: SKU 단위 vs Category/Sub Category 단위 top-down 예측 후 재분배, 저매출 SKU 하이브리드 적용, 매출 구간별 권장 집계 단위
- **그룹 4 (4-1~4-5)**: 요일 가중치 미적용/적용(7단계)/주중·주말 2단계/월별 계절성 추가, 최적 조합 선정
- **그룹 5 (5-1~5-5)**: 9월 실측 오차 산출, 예측 시계(9→12월)별 오차 추이, 오차 원인 분석, 발주량 비교, 실무 도입 최종 결론
- **그룹 6 (6-1~6-5)**: 학습기간 1/2/3/5개월 비교, 최소 권장 학습기간 도출

각 함수의 docstring에 해당 MD 파일의 INPUT/OUTPUT 스펙을 그대로 매핑해 두었습니다.

## 참고

- 지표는 MAPE(0 판매일은 정의상 제외)와 WAPE(저매출 SKU에도 안정적인 집계 오차율)를 함께 제공합니다.
- 코드는 실제 데이터 연결을 전제로 한 뼈대(working scaffold)입니다. 품절/신제품 이력처럼
  합성 데이터에 없는 정보가 필요한 부분(그룹 5-3 등)은 TODO로 표시해 두었으니, 실제 운영
  테이블과 연결 시 보완이 필요합니다.
