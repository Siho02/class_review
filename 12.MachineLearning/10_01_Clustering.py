'''
Clustering(군집)

1. 정의 : 비지도 학습으로 비슷한 특성을 가지는 데이터들끼리 그룹으로 묶는다.

2. 적용
    1) 고객 분류 : 고객 데이터를 바탕으로 비슷한 특징의 고객들을 묶어 성향을 파악할 수 있다.
    2) 이상치 탐지 : 모든 군집에 묶이지 않는 데이터는 이상치일 가능성이 높다
    3) 준지도 학습 : 레이블이 없는 데이터셋에 군집을 이용해 Label을 생성해 분류 지도학습을 할 수 있다. 또는 레이블을 좀더 세분화 할 수 있다.

3. K-mean(K평균)
    1) 설명
        - 가장 널리 사용되는 군집 알고리즘 중 하나.
        - 데이터셋을 K의 군집으로 나눈다. K는 하이퍼파라미터로 사용자가 지정한다.
        - 군집의 중심이 될 것 같은 임의의 지점(Centroid)을 선택해 해당 중심에 가장 가까운 포인드들을 선택하는 기법.
    
    2) 특징
        - K-mean은 군집을 원 모양으로 간주 한다.
        - 모든 특성은 동일한 Scale을 가져야 한다.
        - 이상치에 취약하다.
    
    3) 적정 군집수 판단(Inertia value)
        - 군집 내 데이터들과 중심간의 거리의 합 >> 군집의 응집도를 나타내는 값
        - 값이 클수록 군집화가 높게 되어 있다
        - 조회 : KMean의 inertia_ 속성
        - 적정 군집수 판단 : 군집 단위 별로 inertia 값을 조회 후 급격히 값이 떨어지는 곳
    
    4) 평가 : 실루엣 지표
        (1) 실루엣 계수(silhouette coefficient)
            - 개별 관측치가 해당 군집 내의 데이터와 얼마나 가깝고 가장 가까운 다른 군집과 얼마나 먼지를 나타내는 지표
            - 1에 가까울 수록 좋은 지표
            - 0에 가까울 수록 다른 군집과 가까움
        (2) 속성
            - silhouette_samples() : 개별 관측치의 실루엣 계수 반환
            - silhouette_score() : 실루엣 계수들을의 평균
        (3) 좋은 군집화
            - 실루엣 계수의 평균이 1에 가까움 >> 좋은 군집화
            - 실루엣 계수의 평균과 개별 군집의 실루엣 계수 평균의 편차가 작음 >> 좋은 군집화


'''
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data = load_iris()
X = data['data']
y = data['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

print('kmeans.labels_')
print(kmeans.labels_.shape)
print(kmeans.labels_)

new_X = [[4.1, 4.5, 1.7, 0.6]]
new_X_scaled = scaler.transform(new_X)
lb = kmeans.predict(new_X_scaled)
print('lb = kmeans.predict(new_X_scaled)',lb)

df = pd.DataFrame(X, columns=data['feature_names'])
df['cluster'] = kmeans.labels_
df['Y_label'] = y

print('-'*20)
print(df['cluster'].value_counts())
    #1    61    
    #0    50
    #2    39

print(kmeans.inertia_)
    # 6.98 

k_list = [2, 3, 4, 5, 6, 7]
inertia_list = []
for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

print(inertia_list)
    #[12.12779075053819, 6.982216473785234, 5.516933472040375, 4.580323115230181, 3.9613933830568673, 3.4898066373652736]

import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.plot(k_list, inertia_list)
plt.title('k & inertia')
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()


### 실루엣 계수### 
from sklearn.metrics import silhouette_samples, silhouette_score
s_coef = silhouette_samples(X_scaled, kmeans.labels_) #X, labels
    #array([0.79347482, 0.70973136, 0.76237793, 0.7257315 , 0.78692945,
    #    0.66218237, 0.75897165, 0.79472986, 0.64055474, 0.73946025,
    #    0.73540441, 0.78351441, 0.71021692, 0.66415815, 0.61666756,
    #    0.51541913, 0.67837458, 0.78627881, 0.64045393, 0.73938586,
    #    ...    
    #    0.55285186, 0.52038649, 0.22686175, 0.58208661, 0.53658431,
    #    0.5019764 , 0.08745778, 0.365267  , 0.39819986, 0.17273361])

np.mean(s_coef) #0.5047687565398589
silhouette_score(X_scaled, kmeans.labels_) #0.5047687565398589

df['실루엣계수'] = s_coef   #df에 실루엣 계수열을 만들어 저장
df.tail()
'''
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  cluster  Y_label     실루엣계수
    145                6.7               3.0                5.2               2.3        2        2  0.501976
    146                6.3               2.5                5.0               1.9        1        2  0.087458
    147                6.5               3.0                5.2               2.0        2        2  0.365267
    148                6.2               3.4                5.4               2.3        2        2  0.398200
    149                5.9               3.0                5.1               1.8        1        2  0.172734
'''