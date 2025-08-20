# datathon

# 📊 Price Grading & Scoring Pipeline

상품의 브랜드 / 카테고리 / 상태를 기준으로 등급화하고, 학습된 회귀 가중치를 기반으로 최종 가격 점수(grade_score)를 만드는 파이프라인입니다.

## 🔹 1단계: 베이스셋 준비
설명

이상치(outlier) 제거 후 price > 0 조건으로 데이터 필터링

원본 인덱스(orig_index)를 따로 보관 → 나중에 원본 데이터(data4)와 다시 합칠 때 사용

log_price : 가격을 로그 변환해 분포 안정화

condition_score : item_condition_id가 클수록 상태가 나쁜 구조라 반대로 점수화(6 - id)

코드
df = data4.drop(index=outlier_indices, errors="ignore").copy()
df = df[df["price"] > 0].copy()
df["orig_index"] = df.index

df["log_price"] = np.log1p(df["price"])
df["condition_score"] = 6 - df["item_condition_id"].astype(int)

## 🔹 2단계: 희소 레벨 묶기 (other)
설명

브랜드/카테고리별로 등장 빈도가 너무 낮은 값들은 통계적으로 신뢰도가 부족 → "other"로 묶음

이후 분석 시 "other"는 제거 (대표성 없는 잡음 처리 목적)

코드
def lump(s, min_count=50, other="other"):
    vc = s.value_counts()
    return s.where(~s.isin(vc[vc < min_count].index), other)

df["brand_grp"] = lump(df["brand_name"].fillna("no_brand"), 50)
df["cat2_grp"]  = lump(df["cat2"], 50)

# 'other' 제거
df = df[(df["brand_grp"] != "other") & (df["cat2_grp"] != "other")].copy()

## 🔹 3단계: 등급화 (분위 기반)
설명

log_price 기준으로 그룹별 평균을 계산 → 분위(percentile rank)로 등급 나누기

5개 구간으로 나눔:

1️⃣ 상위

2️⃣ 중상

3️⃣ 중간

4️⃣ 중하

5️⃣ 하위

표본 수가 적은 그룹은 스무딩 처리 (global mean과 혼합)

최종적으로 brand_grade, category_grade, condition_grade 생성

코드
def make_grade_pct(df, by, target="log_price", k=20, out="grade", min_n=None):
    gmean = df[target].mean()
    grp = df.groupby(by, dropna=False)[target].agg(count="count", mean="mean").reset_index()

    # 스무딩 평균
    grp["sm"] = (grp["count"]*grp["mean"] + k*gmean) / (grp["count"] + k)

    # 분위 기반 rank → 구간화
    grp["prank"] = grp["sm"].rank(pct=True, ascending=True, method="average")
    grp[out] = pd.cut(
        grp["prank"],
        bins=[0, 0.25, 0.50, 0.75, 0.90, 1.00],
        labels=[5, 4, 3, 2, 1],  # 1=상위, 5=하위
        include_lowest=True,
        right=True
    ).astype("Int64")

    if min_n is not None:
        grp.loc[grp["count"] < min_n, out] = 3  # 표본 적으면 중간값

    grp[out] = grp[out].astype(int)
    return df.merge(grp[by + [out]], on=by, how="left", validate="m:1")

df = make_grade_pct(df, ["brand_grp"],       out="brand_grade",     k=20, min_n=20)
df = make_grade_pct(df, ["cat2_grp"],        out="category_grade",  k=20, min_n=10)
df = make_grade_pct(df, ["condition_score"], out="condition_grade", k=10, min_n=None)

## 🔹 4단계: 가중치 학습 → grade_score 계산
설명

brand_grade, category_grade, condition_grade → log_price를 설명하는 정도(회귀계수) 학습

회귀계수를 정규화 후 스케일 조정 (가장 작은 값을 1로 맞춤)

최종 산식:

grade_score = w1 * brand_grade + w2 * category_grade + w3 * condition_grade

코드
X = df[["brand_grade", "category_grade", "condition_grade"]]
y = df["log_price"]

coefs = LinearRegression().fit(X, y).coef_

# 회귀계수 → 절대값 비율 → 스케일 조정
weights = np.abs(coefs) / np.abs(coefs).sum()
scaled_weights = weights / weights.min()
w1, w2, w3 = scaled_weights

df["grade_score"] = (
    w1*df["brand_grade"] + w2*df["category_grade"] + w3*df["condition_grade"]
).astype(float)



✅ 최종 생성되는 컬럼

log_price : 로그 변환 가격

condition_score : 상태 점수 (좋음=5 → 나쁨=1)

brand_grp, cat2_grp : 희소 묶기 그룹

brand_grade, category_grade, condition_grade : 5등급화된 점수

grade_score : 회귀계수 기반 가중합 점수

📌 활용 포인트

왜 other 제거?
→ 극소수 브랜드/카테고리는 대표성이 없어 평균 왜곡. 신뢰도 높은 Top-N 그룹 중심 분석

해석 가능성
→ 가중치(w1, w2, w3)를 통해 브랜드/카테고리/상태 중 어느 요인이 가격에 가장 기여하는지 분석 가능

모델 입력
→ grade_score 자체를 머신러닝 특성으로 사용하거나, 각 grade 변수를 독립적으로 사용 가능
