'''
연속형(수치형) 데이터 전처리

1. 정규화(feature scaling)
    1) 정의 : 각 피처가 가지는 값들의 숫자 범위(Scale)가 다를 경우 이 값의 범위를 일정한 범위로 맞추는 작업
    2) 사용
        - 트리계열을 제외한 대부분의 머신러닝 알고리즘들이 피처의 스케일에 영향을 받는다
        - 선형모델, SVM(support vector machine) 모델, 신경망 모델 
        - Scaling(정규화)은 train set으로 fitting 한다. test set이나 예측할 새로운 데이터는 train set으로 fitting한 것으로 변환한다.
        - 사용하는 함수
            - fit(): 어떻게 변환할 지 학습
            - transform(): 변환
            - fit_transform(): 학습과 변환을 한번에 처리
    3) 종류
        (1) 표준화
            - 피쳐의 값들이 평균이 0이고 표준편차가 1인 범위(표준정규분포)에 있도록 변환한다.
              (0을 기준으로 모든 데이터들이 모여있게 된다)
            - 특히 SVM이나 선형회귀, 로지스틱 회귀 알고리즘(선형모델)은 데이터셋이 표준정규분포를 따를 때 
              성능이 좋은 모델이기 때문에 표준화를 하면 대부분의 경우 성능이 향상된다.
            - sklearn.preprocessing.StandardScaler 이용
        (2) Min Max Scaling
            - 데이터셋의 모든 값을 0과 1 사이의 값으로 변환한다
            - feature scaling 시점
                - X(input, feature)를 train, test, validation set으로 분할한 후 train set으로 학습한 Scaler를 이용하여 train/test/validataion set을 변환
            - 범주형 전처리(Label, OneHotEncoding) 시점
                - X값 전체를 가지고 처리한 뒤에 train/test/validataion으로 분할
'''

# 1. Feature Scaling (표준화)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 기본 연습
data = np.array([10, 2, 30]).reshape(3,1) # [10, 2, 30]은 1차원 배열이므로 2차원 배열로 변환
sc = StandardScaler()
sc.fit(data)
rv = sc.transform(data)
        #[[-0.33968311]
        # [-1.01904933]
        # [ 1.35873244]]

#load_iris를 이용한 표준화 

iris = load_iris()
X, y = iris['data'], iris['target']

df = pd.DataFrame(X, columns=iris['feature_names'])
#print('평균과 표준편차\n', df.mean(), df.std())

s_scaler = StandardScaler()
s_scaler.fit(df)
s_df = s_scaler.transform(df)
    #[[-9.00681170e-01  1.01900435e+00 -1.34022653e+00 -1.31544430e+00]
    # [-1.14301691e+00 -1.31979479e-01 -1.34022653e+00 -1.31544430e+00]
    # [-1.38535265e+00  3.28414053e-01 -1.39706395e+00 -1.31544430e+00]
    # ...
    # [ 6.86617933e-02 -1.31979479e-01  7.62758269e-01  7.90670654e-01]]

# 열 이름 추가
s_df = pd.DataFrame(s_df, columns = iris['feature_names'])

# 스케일링 훈련 및 변환
rv = s_scaler.fit_transform(df)



#2. MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#minmaxscaler 모델 생성, 훈련 및 변환
mm_scaler = MinMaxScaler()
mm_scaler.fit(df)
rv = mm_scaler.transform(df)


#데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train)

#데이터 스케일링
mm_scaler = MinMaxScaler()
X_train_scaled = mm_scaler.fit_transform(X_train)
X_val_scaled = mm_scaler.transform(X_val)
X_test_scaled = mm_scaler.transform(X_test)