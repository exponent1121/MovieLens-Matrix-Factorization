# 🎞️ MovieLens-100k Matrix Factorization Recommender from Scratch

### 📌 Project Overview
본 프로젝트는 **외부 딥러닝 프레임워크(PyTorch, TensorFlow 등)를 전혀 사용하지 않고**, 오직 **NumPy와 선형대수 수식만을 활용**하여 사용자-아이템 협업 필터링 알고리즘인 **Biased Matrix Factorization**을 바닥부터(From Scratch) 직접 구현한 추천 시스템입니다. 

`argparse`를 통한 유연한 CLI 환경을 지원하며, MovieLens 100K 데이터셋 환경에서 하이퍼파라미터 최적화를 통해 목표 성능 RMSE 0.9127을 성공적으로 달성했습니다.

---

### 🧠 Core Algorithm (핵심 알고리즘)

#### 1. Biased Matrix Factorization 구현
단순한 행렬 분해(Matrix Factorization)는 평점을 관대하게 주거나 짜게 주는 유저/아이템 고유의 성향을 반영하지 못합니다. <br>
이를 보완하기 위해 <strong>Global Bias(전체 평균), User Bias(유저 편향), Item Bias(아이템 편향)</strong>를 수식에 통합하여 <br> 예측 정밀도를 극대화했습니다. <br><br>

##### 🎯 평점 예측 수식
$$\hat{r}_{ui} = \mu + b_u + b_i + P_u Q_i^T$$

* $\mu$: 전체 데이터셋의 평균 평점 (Global Bias)
* $b_u$: 특정 유저 $u$의 평점 성향 편향 (User Bias)
* $b_i$: 특정 아이템 $i$의 고유 특징 편향 (Item Bias)
* $P_u, Q_i$: 유저와 아이템의 잠재 요인 행렬 (Latent Factor Matrices)
<br><br>

#### 2. Robust Cold-Start Handling (강력한 예외 처리)
추천 시스템의 고질적인 문제인 <strong>Cold-Start(학습 데이터에 없는 유저/아이템이 테스트에 등장하는 현상)</strong>를 안전하게 방어합니다. <br>
`predict` 단계에서 유저와 아이템의 인덱스 범위를 검사하여 다음과 같이 유연하게 대응합니다.
* 유저와 아이템 모두 인지된 경우: $\mu + b_u + b_i + P_u Q_i^T$
* 유저만 인지된 경우 (새로운 아이템): $\mu + b_u$ (유저 편향 반영)
* 아이템만 인지된 경우 (새로운 유저): $\mu + b_i$ (아이템 편향 반영)
* 모두 처음 보는 경우: $\mu$ (전체 평균 평점 반환)

---

#### 📉 최적화 알고리즘
오차 함수를 최소화하기 위해 <strong>확률적 경사 하강법(SGD, Stochastic Gradient Descent)</strong>을 적용하여 <br> 에포크마다 가중치를 실시간으로 업데이트. <br>
과적합을 방지하기 위해 $L_2$ Regularization 패널티 수식에 포함.


---

### 📊 Performance Evaluation (성능 평가 결과)

MovieLens에서 제공하는 `u1`부터 `u5`까지의 5-Fold dataset을 바탕으로 독립 실행한 최종 RMSE 결과.

* **최적 하이퍼파라미터 세팅:** $k=40, lr=0.014, reg=0.10, \text{epochs}=30$, seed=3

| Validation Set | Total Evaluated Pairs | FINAL RMSE |
| :--- | :---: | :---: |
| **u1.test** | 20,000 | 0.923729 |
| **u2.test** | 20,000 | 0.914145 |
| **u3.test** | 20,000 | 0.905936 |
| **u4.test** | 20,000 | 0.908046 |
| **u5.test** | 20,000 | 0.911844 |
| 📊 **Average** | **100,000** | **0.912738** |

> **Result Summary:** <br>
5개 데이터셋의 평균 RMSE 결과 **0.912738**을 기록하며, 목표 베이스라인이었던 **0.9130**을 성공적으로 돌파.

---

### 🛠️ Troubleshooting & Tuning (트러블슈팅 여정)

#### 1. 난수 초기화로 인한 성능 재현성(Reproducibility) 문제 해결
* **문제:** 실행할 때마다 잠재 행렬 $P, Q$의 초기 난수 상태와 `shuffle` 순서가 달라져 동일한 파라미터 환경에서도 RMSE 결과가 불규칙하게 변동하는 현상 발생.
* **해결:** 구형 방식인 `np.random.seed` 대신, 최신 NumPy 표준인 **`np.random.default_rng(seed)`** 독립 제너레이터 객체를 도입. <br>
이를 통해 객체 지향적인 난수 상태 관리가 가능해졌으며, 어떤 환경에서 실행하더라도 완벽히 동일한 성능 수치가 도출되도록 재현성 확보.

#### 2. 높은 학습률(Learning Rate) 하에서의 오차 발산 방지
* **문제:** SGD 학습 과정에서 잠재 요인의 차원($k=40$)이 커짐에 따라 모델이 훈련 데이터의 노이즈까지 과하게 학습하거나 오차가 발산하여 RMSE가 0.95 이상으로 급격히 치솟는 현상 관측.
* **해결:** 모델의 숨통을 조이지 않으면서도 과적합을 강하게 억제할 수 있도록 $L_2$ Regularization Term(정규화 패널티)의 가중치를 <strong>`reg=0.10`</strong>으로 상향 조정. <br>
패널티 규제를 맞춘 결과, 별도의 학습률 감쇠(LR Decay) 로직 없이도 30 에포크 이내에 안정적으로 최적점에 수렴.

---

### 💻 How to Run (실행 방법)

`argparse` 라이브러리를 통해 터미널 창에서 자유롭게 하이퍼파라미터를 입력하여 실행할 수 있습니다.

#### 1. 추천 모델 학습 및 예측 프로그램 실행 (`recommender.py`)
```bash
python recommender.py u1.base u1.test
```
* 실행 완료 시 `u1.base_prediction.txt` 파일이 자동으로 생성됩니다.


#### 2. 자체 검증 스크립트를 통한 RMSE 확인 (`test_rs.py`)
: 생성된 예측 파일과 원본 테스트 파일을 비교하여 최종 RMSE를 계산
```bash
python test_rs.py u1.test u1.base_prediction.txt
```
python test_rs.py u1.test u1.base_prediction.txt ```
