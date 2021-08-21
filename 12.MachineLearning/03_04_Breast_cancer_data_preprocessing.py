#위스콘신 대학교에서 제공한 유방암 진단결과 데이터
#ID, 암측정값들, 진단결과 컬럼들로 구성
#사이킷런에서 제공. load_breast_cancer() 함수 이용

from math import gamma
from scipy.sparse.construct import random
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

#print(data['feature_names'], end='\n')
#print(data['target_names'], end='\n')
#print(data['data'], end='\n')

X, y = data['data'], data['target']

#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)

##1. 스탠다드 스케일링을 이용한 feature scaling
s_scale = StandardScaler()
X_train_standard_scale = s_scale.fit_transform(X_train)
X_val_standard_scale = s_scale.transform(X_val)
X_test_standard_scale = s_scale.transform(X_test)

# 확인
print('X_train의 평균\n',np.mean(X_train, axis=0))
print('X_train_standard_scale의 평균\n',np.mean(X_train_standard_scale, axis=0))

print('X_train의 표준편차\n',np.std(X_train, axis=0))
print('X_train_standard_scale의 표준편차\n',np.std(X_train_standard_scale, axis=0))

print('X_test_standard_scaled의 평균\n',np.mean(X_train_standard_scale, axis=0))
print('X_test_standard_scale의 표준편차\n',np.std(X_test_standard_scale, axis=0))

##2. MinMaxScaling을 이용한 feature scaling
mm_scaler = MinMaxScaler()
X_train_minmax_scale = mm_scaler.fit_transform(X_train)
X_val_minmax_scale = mm_scaler.transform(X_val)
X_test_minmax_scale = mm_scaler.transform(X_test)

print('X_train_minmax_scale의 최대, 최소 배열')
print(np.max(X_train_minmax_scale, axis=0), np.min(X_train_minmax_scale, axis=0))

print('X_test_minmax_scale의 최대, 최소 배열')
print(np.max(X_test_minmax_scale, axis=0), np.min(X_test_minmax_scale, axis=0))

print('X_val_minmax_scale의 최대, 최소 배열')
print(np.max(X_val_minmax_scale, axis=0), np.min(X_val_minmax_scale, axis=0))


##Modeling 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Scaling하지 않는 X_train으로 학습 및 평가
svc = SVC(C=0.1, gamma=0.1, random_state=0)
    #C, gamma >> 성능과 관련 hyper parameter

#학습
svc.fit(X_train, y_train)

pred_train = svc.predict(X_train)
pred_val = svc.predict(X_val)
pred_test = svc.predict(X_test)

print('SVC 학습 시 C와 gamma를 0.1로 설정')
print('train 정화도 = {}, val 정확도 = {}, test 정확도 = {}'.format(accuracy_score(y_train, pred_train), 
        accuracy_score(y_val, pred_val), accuracy_score(y_test, pred_test)))

##Standard Scaling 된 X_train_scale로 학습 및 평가
#standard scaler로 전처리된 데이터셋
svc = SVC(C=0.1, gamma=0.1, random_state=0)
#학습
svc.fit(X_train_standard_scale, y_train)
#평가 / 추론 >> 학습 데이터셋에 처리한 전처리를 똑같이 적용한 뒤 추론
pred_train_scale = svc.predict(X_train_standard_scale)
pred_val_scale = svc.predict(X_val_standard_scale)
pred_test_scale = svc.predict(X_test_standard_scale)
print('standard_scaled_train 정화도 = {}, standard_scaled_val 정확도 = {}, standard_scaled_test 정확도 = {}'.format(accuracy_score(y_train, pred_train_scale), 
        accuracy_score(y_val, pred_val_scale), accuracy_score(y_test, pred_test_scale)))

##Min Max Scaling 된 X_train_scale로 학습 평가
svc = SVC(C=0.1, gamma=0.1, random_state=0)
svc.fit(X_train_minmax_scale, y_train)

pred_train_minmax = svc.predict(X_train_minmax_scale)
pred_val_minmax = svc.predict(X_val_minmax_scale)
pred_test_minmax = svc.predict(X_test_minmax_scale)

print('minmax_scaled_train 정화도 = {}, minmax_scaled_val 정확도 = {}, minmax_scaled_test 정확도 = {}'.format(accuracy_score(y_train, pred_train_minmax), 
        accuracy_score(y_val, pred_val_minmax), accuracy_score(y_test, pred_test_minmax)))
print("일반적으로 Standard Scaler가 효율이 좋음(100%는 아님, 두개 다 해서 비교해볼 것)\n\n")

svc = SVC(random_state=0)
    #C와 gamma에 따로 값을 주지 않음
svc.fit(X_train, y_train)

pred_train = svc.predict(X_train)
pred_val = svc.predict(X_val)
pred_test = svc.predict(X_test)

print('SVC 학습 시 C와 gamma를 디폴트 값으로 지정(따로 설정값을 주지 않음)')
print('train 정화도 = {}, val 정확도 = {}, test 정확도 = {}'.format(accuracy_score(y_train, pred_train), 
        accuracy_score(y_val, pred_val), accuracy_score(y_test, pred_test)))

svc = SVC(random_state=0)
svc.fit(X_train_standard_scale, y_train)

pred_train_scale = svc.predict(X_train_standard_scale)
pred_val_scale = svc.predict(X_val_standard_scale)
pred_test_scale = svc.predict(X_test_standard_scale)
print('standard_scaled_train 정화도 = {}, standard_scaled_val 정확도 = {}, standard_scaled_test 정확도 = {}'.format(accuracy_score(y_train, pred_train_scale), 
        accuracy_score(y_val, pred_val_scale), accuracy_score(y_test, pred_test_scale)))

svc = SVC(random_state=0)
svc.fit(X_train_minmax_scale, y_train)

pred_train_minmax = svc.predict(X_train_minmax_scale)
pred_val_minmax = svc.predict(X_val_minmax_scale)
pred_test_minmax = svc.predict(X_test_minmax_scale)

print('minmax_scaled_train 정화도 = {}, minmax_scaled_val 정확도 = {}, minmax_scaled_test 정확도 = {}'.format(accuracy_score(y_train, pred_train_minmax), 
        accuracy_score(y_val, pred_val_minmax), accuracy_score(y_test, pred_test_minmax)))