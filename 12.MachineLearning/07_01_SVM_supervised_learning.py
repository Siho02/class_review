'''
Support Vector Machine(SVM)

1. Linear SVM
    - 딥러닝 이전에 분류에서 뛰어는 성능으로 많이 사용되었던 분류 모델
    - 하나의 분류 그룹을 다른 그룹과 분리하는 최적의 경계를 찾아내는 알고리즘
    - 중간 크기의 데이터셋과 특성(feature)이 많은 복잡한 데이터셋에서 성능이 좋은 것으로 알려져있다.

    1) 목표
        - support vector 간의 가장 넓은 margin을 가지는 초평면(결정경계)를 찾는다
        - margin이 넓은 결정 경계를 만드는 함수를 찾는 것
        
        (1) 초평면 : 데이터가 존재하는 공간보다 1차원 낮은 부분 공간
            - 데이터 : n차원  >> 초평면 : n-1 차원
            - 공간을 나누기 위해 초평면을 사용
            - 1차원:점 / 2차원:선 / 3차원:평면 / 4차원이상:초평면
        (2) Support Vector
            - 경계를 찾아내는데 기준이 되는 데이터 포인트
            - 초평면(결정 경계)에 가장 가까이 있는 vector(데이터포인트)를 말한다.
        (3) margin
            - 두 support vector간의 너비
            ● Hard Margin, Soft Margin
                - Overfitting(과적합)을 방지하기 위해 어느정도 오차를 허용하는 방식을 Soft margin이라고 한다. 
                - 반대로 오차를 허용하지 않는 방식을 Hard Margin이라고 한다.
                - 모든 데이터 셋이 완전히 분류 되지는 않음
                - 노이즈가 있는 데이터나 선형적으로 분리 되지 않는 경우 하이퍼파마미터인 'C' 조정해 마진을 변경
                    - C의 기본값 : 1
                    - C를 크게 주면 >> 마진폭 좁아짐(마진오류 ↓) >> overfitting 가능성↑
                    - C를 작게 주면 >> 마진폭 넓어짐(마진오류 ↑) & 훈련 데이터 성능↓ but 테스트 데이터 성능↑(일반화↑) >> underfitting 가능성↑
'''

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

#SVM은 선형 기반 모델 >> Feature Scaling 전처리
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

#SVC 학습 및 평가
#C의 변화에 따른 평가 결과 변환
c_params = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

train_acc_lst = []
test_acc_lst = []

for c in c_params:
    svc = SVC(kernel='linear', C = c, random_state=0)
    svc.fit(X_train_scale, y_train)

    pred_train = svc.predict(X_train_scale)
    pred_test = svc.predict(X_test_scale)

    train_acc_lst.append(accuracy_score(y_train, pred_train))
    test_acc_lst.append(accuracy_score(y_test, pred_test))

result_df = pd.DataFrame({
    'C' : c_params,
    'Train' : train_acc_lst,
    'Test' : test_acc_lst
})
'''
          C     Train      Test
0    0.0001  0.643192  0.643357
1    0.0010  0.936620  0.944056
2    0.0100  0.978873  0.958042
3    0.1000  0.990610  0.951049
4    1.0000  0.990610  0.958042
5   10.0000  0.992958  0.944056
6  100.0000  1.000000  0.930070
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(c_params, train_acc_lst, marker='o', label='Train')
plt.plot(c_params, test_acc_lst, marker='o', label='Test')
plt.ylim(0.9, 1)
plt.xlim(0, 1)
plt.legend()
plt.title('accuracy for C')
plt.show()


'''
커널 서포트 벡터 머신(Kernal Support Vector Machine)

1. 비선형 데이터 셋에 SVM 적용
    1) 선형으로 분리가 안되는 경우(데이터들이 x축 또는 y축에 평행한 경우)
        - 다항식 특성을 추가하여 차원을 늘려 선형 분리가 되도록 변환(2차원으로 변환 x3 = (x1)^2)
        - 차원을 늘리는 경우
            - 다항식 특성을 추가하는 방법
                - 낮은 차원의 데이터 패턴 >> 과소적합 가능성
                - 높은 차원의 데이터 패턴 >> 과대적합 가능성 및 모델 속도 저하

    2) 커털 트릭(Kernal Trick)
        - 다항식을 만들기 위한 특성을 추가하지 않으면서 수학적 기교를 적용
        
        (1) 방사기저(radial base function - RBF) 함수
            - 커널 서포트 벡터 머신의 기본 커널 함수
            - 기준점들이 되는 위치를 지정하고 각 샘플이 그 기준점들과 얼마나 떨어졌는지 계산 >> 유사도(거리)
            - 기준점 별 유사도를 계산한 값 = 원래 값보다 차원↑, 선형적으로 구분될 가능성↑
            - Φ(x, l) = exp(γ||x-l||^2)
                - x:샘플 / l:기준값 / γ:규제파라미터
            - 하이퍼 파라미터
                - C : 오차 허용기준, 작은 값일 수록 많이 허용(큰 값일 수록 과적합 가능 높음)
                - gamma(γ): 큰 값일 수록 과적합 가능성 높음 / 작은 값일 수록 과소적합 가능성 증가

'''

# C는 고정한 후 gamma를 변화
gamma_params = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

train_acc_lst = [] 
test_acc_lst = []

for gamma in gamma_params:
    svc = SVC(kernel='rbf', C=1, gamma=gamma, random_state=0)
    svc.fit(X_train_scale, y_train)

    pred_train = svc.predict(X_train_scale)
    pred_test = svc.predict(X_test_scale)

    train_acc_lst.append(accuracy_score(y_train, pred_train))
    test_acc_lst.append(accuracy_score(y_test, pred_test))

result_df = pd.DataFrame({
    'gamma' : gamma_params,
    'train' : train_acc_lst,
    'test' : test_acc_lst
})
'''
      gamma     train      test
0    0.0001  0.791080  0.804196
1    0.0010  0.950704  0.958042
2    0.0100  0.985915  0.965035
3    0.1000  0.995305  0.923077
4    1.0000  1.000000  0.636364
5   10.0000  1.000000  0.629371
6  100.0000  1.000000  0.629371
'''

# svc 모델에서 predict_prob() 함수를 사용하기 위해서는 probability 매개변수의 값을 True로 설정해야 한다.
svc = SVC(C=1, gamma=0.001, random_state=0, probability=True)
svc.fit(X_train_scale, y_train)

prob_test = svc.predict_proba(X_test_scale)

from sklearn.metrics import roc_auc_score, average_precision_score
print(roc_auc_score(y_test, prob_test[:, 1]), average_precision_score(y_test, prob_test[:,1]))    
    #0.9849056603773585 0.9905770103513604
    