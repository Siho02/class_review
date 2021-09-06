'''
로지스틱 회귀(Logistic Regression)
    - 선형 회귀 알고리즘을 이용한 이진 분류 모델
    - sample이 특정 클래스에 속할 확률을 추정

    1. 확률 추정 
        - 선형 회귀처럼 입력특성(feature)에 가중치 합을 계산한 값을 로지스틱 함수를 적용해 확률을 계산

        1) 로지스틱 함수
            - 0과 1 사이의 실수를 반환
            - S자 형태의 결과를 내는 시그모이드 함수이다.
                𝜎(𝑥)= 1 / (1 + 𝐞^(-x))
            - 샘플 x가 양성에 속할 확률
                y_hat = 0 (p_hat < 0.5)
                        1 (p_hat < 0.5) 
    
    2. 손실 함수(loss function)
        - 손실 함수 L(W) = -(1/m) * ∑[y_i * log(p_i_hat) + (1 - y_i) * log(1 - p_i_hat)]
        - y(실제값)이 1인 경우 y_i * log(p_i_hat)이 손실을 계산
        - y가 0인 경우 (1 - y_i) * log(1 - p_i_hat)이 손실을 계산
        - p_hat(예측확률)이 클수록 반환값은 작아지고, 작을 수록 값이 커진다
    
    3. 최적화
        - 위 손실을 가장 적게하는 가중치(W)를 찾는다
        - 로그 손실함수는 최소값을 찾는 정규방정식이 없음 >> Logistic Regression은 경사하강법을 이용해 최적화
        - 로그 손실을 W로 미분 >> (1 / m) * ∑(σ(W^T * x_i) - y_i)x_ij   
    
    4. 주요 하이퍼파라미터
        - penalty : 과적합을 줄이기 위한 규제 방식
            - l1, l2(기본값), elasticnet, none
        - C : 규제강도(기본값=1), 작을 수록 규제가 강하다
        - max_iter(기본값=100) : 경사하강법 반복 횟수
'''
## 로지스틱 함수
import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(-10, 10, 100)      #shape = (100, )
sigmoid = 1 / (1 + np.exp(-xx))     #shape = (100, )

plt.figure(figsize=(12, 6))
plt.plot(xx, sigmoid, color='b', linewidth = 2)
plt.plot([-10, 10], [0, 0], color = 'k', linestyle = '-')
plt.plot([-10, 10], [0.5, 0.5], color='r', linestyle=':', label='y=0.5')
plt.xlabel("x")
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()

## 손실함수
print('log계산')
print('-log(1), -log(0.99),\t-log(0.7),\t-log(0.51),\t-log(0.5),\t-log(0.4),\t-log(0.2),\t-log(0.0000001)')
print(-np.log(1), -np.log(0.99), -np.log(0.7), -np.log(0.51), -np.log(0.5), -np.log(0.4), -np.log(0.2), -np.log(0.0000001))
print('\nnp.log(1), np.log(0.7), np.log(0.5), np.log(0.1), np.log(0.0000000001)')
print(np.log(1), np.log(0.7), np.log(0.5), np.log(0.1), np.log(0.0000000001))

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

#데이터 load, split
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성 + 학습
lr = LogisticRegression(random_state=0)
lr.fit(X_train_scaled, y_train)

# 평가
pred_train = lr.predict(X_train_scaled)
pred_test = lr.predict(X_test_scaled)
accuracy_score(y_train, pred_train),  accuracy_score(y_test, pred_test)

# GridSearchCV 파라미터 - penalty, C 
param = {
    'penalty':['l1', 'l2'], 
    'C':[0.001, 0.01, 0.1, 1, 10]
}

gs = GridSearchCV(LogisticRegression(random_state=0), 
                  param,
                  cv=5, 
                  scoring='accuracy', 
                  n_jobs=-1)

gs.fit(X_train_scaled, y_train)

result = pd.DataFrame(gs.cv_results_)
print(result.sort_values('rank_test_score').head())
print(gs.best_params_)

best_model = gs.best_estimator_
pred_test = best_model.predict(X_test_scaled)
print(accuracy_score(y_test, pred_test))