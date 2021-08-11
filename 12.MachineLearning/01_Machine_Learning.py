import numpy as np
import pandas as pd
from scipy.sparse.construct import random
from sklearn.datasets import load_iris

iris = load_iris()
print("iris의 데이터 타입 : ", type(iris))
#print(iris)

print("iris['data']의 shape와 데이터 타입 : ", iris['data'].shape,', ' ,type(iris['data']))

print("iris['data']의 일부분",iris['data'][:10])

print("iris 데이터의 feature 데이터 : ",iris['feature_names'])
print("iris 데이터의 target 데이터 : ", iris['target'])

print(np.unique(iris['target'], return_counts=True))
print(iris['target_names'])

#print('-='*30)

#iris 데이터의 설명서
#print(iris['DESCR'])

print("iris데이터를 이용하여 데이터프레임 만들기")
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

#iris_df에 새로운 '품종'열을 만들어 데이터 넣어주기
iris_df['품종'] = iris['target']

#150행 5열의 dataframe
print(iris_df.head())
print(iris_df.shape)

#iris_df의 info
print(iris_df.info())

#iris_df의 요약 정보
print(iris_df.describe().T)


## 결정트리 모델을 이용한 머신 러닝 구현
#0. tree 모델 import
from sklearn.tree import DecisionTreeClassifier

#1. 모델 생성 >> 모델 클래스 객체 생성
tree = DecisionTreeClassifier(random_state=0)

#2. 모델 학습(train) - 모델에 수집한 데이터셋을 전달, 모델이 데이터의 패턴을 찾도록 하는 작업
#2-1. 지도학습 : input data(feature), output data(label)을 나눠서 전달
#ex. 모델.fit(input_data, output_data)
tree.fit(iris['data'], iris['target'])

#iris['data'][:2]의 결과 [5.1, 3.5, 1.4, 0.2], [4.9, 3. , 1.4, 0.2]
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [4.5, 3.7, 6.5, 3.2]])
result = tree.predict(new_data)

print(new_data.shape, result.shape)
print(type(result), result)

print(result, ' >> ',iris['target_names'][result])



## 결과를 예측했지만 이게 정말 맞는 것이지는 알 수 없다 >> 모델의 성능이 좋은지 나쁜지 알 수 없다
## 그러므로 우리는 앞으로 전체 데이터를 학습용 데이터와 평가용 데이터로 나눈다(보통 8:2 또는 7:3으로 나눈다(데이터가 충분히 많을 경우 6:4도 가능하다))
## 데이터셋을 분할하기 위해서 필요한 것 >> 모듈
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris['data']
y = iris['target']

#X는 150행 4열 데이터, y는 150행 데이터
print("X와 y의 행*열",X.shape, y.shape)

#데이터 분할 
# test_size=0.2 >> 학습용, 평가용을 8:2로 나눈다
# shuffle=True >>나누기 전에 섞음
# stratify=y >> 나누기 전과 후의 클래스 비율을 동일하게 한다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
print("분할데이터들의 행*열", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#모델 생성
tree = DecisionTreeClassifier(random_state=0)

#학습용 데이터 모델 학습
tree.fit(X_train, y_train)

#모델을 통한 예측  
pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)

#모델 평가 >> 정확성을 기준으로
print('---------pred_test와 y_test-----------')
print('pred_test : ',pred_test)
print('y_test    : ', y_test)

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)

print('train 정확도 : {}, test 정확도 : {}'.format(train_acc, test_acc))

#------------------------------------------------------------------------------
#혼동 행렬 (confusion matrix)
    # 예측한 것이 실제 무엇이었는지를 표로 구성한 평가 지표
    # 분류의 평가 지표로 사용
    # axis 0 = 실제, axis 1 = 예측
#------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

train_cm = confusion_matrix(y_train, pred_train)
test_cm = confusion_matrix(y_test, pred_test)

print('train 혼동행렬')
print(train_cm)
print('')
print('test 혼동행렬')
print(test_cm)