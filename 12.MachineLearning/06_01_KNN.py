'''
최근접 이웃(K-Nearest Neighbors, KNN)
1. KNN
    - 예측하려는 데이터와 input 데이터들 간의 거리를 측정해 가장 가까운 K개의 data set의 레이블을 참조해 분류/예측한다.
    - 학습시 단순 input 데이터들을 저장만 하며 예측 시점에 거리를 계산
        - 학습은 빠르지만 예측시 시간이 많이 소요
    - 분류(classificaion)와 회귀(regression)을 모두 지원

2. 분류
    - KNN에서 K는 새로운 데이터 포인트를 분류할 때, 확인할 데이터 포인트의 개수를 지정하는 hyper parameter
    - K가 너무 작은 경우 >> 과적합 발생 >> K 값을 더 크게 잡음
    - K가 너무 큰 경우 >> 과소적합 발생 >> K 값을 더 작게 잡음

    1) 주요 hyper parameter
        (1) 이웃 수 
            - n_neighbors = K
            - K가 작을 수록 모델이 복잡해져 과적합이 일어나고 너무 크면 성능이 떨어짐
            - n_neighbors는 feature 수의 제곱근 정도를 지정할 때 성능이 좋은 것으로 알려져 있다.
        (2) 거리 측정 방법
            - p = 2 : 유클리디안 거리(기본값)
                - Euclidian_distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
            - p = 1 : 맨하탄 거리
                - Manhattan_distance = |x2 - x1| + |y2 - y1|
    
    2) 요약
        - K-NN은 이해하기 쉬운 모델이며 튜닝할 하이퍼파라미터의 수가 적어 빠르게 만들 수있다.
        - K-NN은 서비스할 모델을 구현할때 보다는 복잡한 알고리즘을 적용해 보기 전에 확인용 또는 base line을 잡기 위한 모델로 사용한다.
        - 훈련세트가 너무 큰 경우(Feature나 관측치의 개수가 많은 경우) 거리를 계산하는 양이 늘어나 예측이 느려진다.
        - Feature간의 값의 단위가 다르면 작은 단위의 Feature에 영향을 많이 받게 되므로 전처리로 Scaling작업이 필요하다.
        - Feature가 너무 많은 경우와 대부분의 값이 0으로 구성된(희소-sparse) 데이터셋에서 성능이 아주 나쁘다
'''

# 유방암 데이터를 활용한 암환자 분류
# K값 변화에 따른 성능 평가 : malignant(악성), benign(양성)

import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

#feature간의 값의 단위가 다르므로 scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

# knn모델을 이용한 train, 평가
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_param = range(1,10)   # 1 ~ 9
train_acc_lst = []
test_acc_lst = []


for k in k_param:
    knn = KNeighborsClassifier(n_neighbors=k)       #모델 생성
    knn.fit(X_train_scale, y_train)                 #학습
    
    pred_train = knn.predict(X_train_scale)
    pred_test = knn.predict(X_test_scale)           #평가

    train_acc_lst.append(accuracy_score(y_train, pred_train))
    test_acc_lst.append(accuracy_score(y_test, pred_test))

df = pd.DataFrame({
    'K' : k_param,
    'train' : train_acc_lst,
    'test' : test_acc_lst
})

'''
   K     train      test
0  1  1.000000  0.888112
1  2  0.985915  0.867133
2  3  0.995305  0.923077
3  4  0.990610  0.902098
4  5  0.988263  0.930070
5  6  0.983568  0.895105
6  7  0.983568  0.923077
7  8  0.981221  0.916084
8  9  0.983568  0.923077
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(k_param, train_acc_lst, marker='o', label='train')
plt.plot(k_param, test_acc_lst, marker='x', label='test')
plt.legend()
plt.grid(True)
plt.xlabel('K value')
plt.title('accuracy for K')
plt.show()

# K=5로 모델 생성
best_model = KNeighborsClassifier(n_neighbors=5)
best_model.fit(X_train_scale, y_train)

print(accuracy_score(y_test, best_model.predict(X_test_scale)))
    #0.9300699300699301

print(X_test_scale[0])


## GridSearch / Pipeline을 이용한 구현
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# pipeline(scaler , knn)
order = [
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier()) 
]

pipeline = Pipeline(order, verbose=True)

# GridSearchCV
param = {
    'knn__n_neighbors' : range(1,11),
    'knn__p' : [1,2]                    # 1:맨해튼거리, 2:유클리드거리
}

gs = GridSearchCV(pipeline, param, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

result_df = pd.DataFrame(gs.cv_results_)
print(result_df[result_df.columns[6:]].sort_values('rank_test_score').head())

'''
                                  params  split0_test_score  split1_test_score  ...  mean_test_score  std_test_score  rank_test_score
4   {'knn__n_neighbors': 3, 'knn__p': 1}           0.953488           0.988235  ...         0.983639        0.015749                1
13  {'knn__n_neighbors': 7, 'knn__p': 2}           0.941860           0.988235  ...         0.978960        0.019987                2
6   {'knn__n_neighbors': 4, 'knn__p': 1}           0.953488           0.988235  ...         0.978933        0.017127                3
12  {'knn__n_neighbors': 7, 'knn__p': 1}           0.930233           0.988235  ...         0.976635        0.023201                4
15  {'knn__n_neighbors': 8, 'knn__p': 2}           0.953488           0.976471  ...         0.976580        0.014709                5
'''

best_model = gs.best_estimator_
print(accuracy_score(y_test, best_model.predict(X_test)))
    #0.951048951048951

