'''
1. 데이터 셋 분리
    1) train data set (훈련 / 학습 데이터 셋)
        - 모델을 학습 시킬 때 사용하는 데이터 셋

    2) validation data set (검증 데이터 셋)
        - train set으로 학습한 모델의 성능을 측정하기 위한 데이터 셋
    
    3) test data set (평가 데이터 셋)
        - 모델의 성능을 최종적으로 측정하기 위한 데이터셋
        - 모델 성능을 측정하는 용도로 단 한번만 사용한다
            - 학습과 평가를 반복할 시 >> 사용한 데이터 셋에 대해 과적합 되어 새로운 데이터셋에 대한 성능이 떨어진다
            - 그러므로 세개의 데이터 셋으로 나누어 train, validation 데이터 셋으로 모델을 최적화 한 뒤 test 셋으로 최종 평가 진행
'''

## Hold Out
'''
Hold Out 방식의 단점 
    - train, test set의 분할 방법에 따라 결과가 상이
        데이터가 충분히 많은 경우 >> 변동성 흡수에 따라 양호
        데이터가 충분치 않은 경우(~수천건) >> 학습이 제대로 되지 않을 수 있음
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

# 데이터를 train / test set으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
# train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
    #arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
    #test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)
    #train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
    #random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
    #shuffle : 셔플여부설정 (default = True)
    #stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.

# train 데이터 셋을 다시 train / validation 데이터셋으로 분리
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
print("shape of y, y_train, y_val, y_test : ",y.shape, y_train.shape, y_val.shape, y_test.shape)

# 모델 생성
tree = DecisionTreeClassifier(random_state=0, max_depth=1)
# 모델 학습
tree.fit(X_train, y_train)
# validation data set 평가 및 예측
pred_val = tree.predict(X_val)
val_acc = accuracy_score(y_val, pred_val)
print('max_depth = 1, validataion 정확도 : ', val_acc)

# max_depth = 3으로 변경
tree = DecisionTreeClassifier(random_state=0, max_depth=3)
tree.fit(X_train, y_train)
pred_val = tree.predict(X_val)
val_acc = accuracy_score(y_val, pred_val)
print('max_depth = 3, validataion 정확도 : ', val_acc)

# max_depth = 5 으로 변경
tree = DecisionTreeClassifier(random_state=0, max_depth=5)
tree.fit(X_train, y_train)
pred_val = tree.predict(X_val)
val_acc = accuracy_score(y_val, pred_val)
print('max_depth = 5, validataion 정확도 : ', val_acc)



'''
K겹 교차 검증(K-Fold Cross Validation)
    1. 정의 및 방법
        1) data set을 k개로 나눈 뒤, 하나는 검증 세트, 나머지는 훈련세트로 하여 모델을 학습, 평가
        2) k개의 데이터 셋이 모두 한번씩 검증 세트가 되도록 k번 모델 학습 >> 평가 지표들을 평균하여 모델의 성능 평가
    
    2. 종류 
        1) k-fold >> 회귀 문제에 사용
        2) stratified k-fold >> 분류 문제에 사용
'''


## KFold ## 

import numpy as np
from sklearn.model_selection import KFold
X, y = load_iris(return_X_y=True)

#데이터 분리
kfold = KFold(n_splits=5)

#KFold객체.split(데이터셋) : argument로 전달한 데이터셋을 지정한 n_split 개수로 나눠 train/test 폴드별 index를 제공하는 generator를 반환
kfold_gen = kfold.split(X)
print('kfold_gen의 type',type(kfold_gen))

#각 fold 별로 평가한 정확도 점수를 저장할 리스트
acc_train_lst = []
acc_test_lst = []

gen = kfold.split(X)

for train_idx, test_idx in gen:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(y_train.shape, y_test.shape)

    #모델 생성 및 학습
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)

    #평가
    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    acc_train_lst.append(acc_train)
    acc_test_lst.append(acc_test)

print(acc_train_lst, acc_test_lst)
train_acc = np.mean(acc_train_lst)
test_acc = np.mean(acc_test_lst)

print(f"train 정확도 : {train_acc}, test 정확도 : {test_acc}")

## K-fold의 단점  >> 원 데이터셋의 row 순서대로 분할 >> 분균형 문제 발생 가능


'''
Stratified K-Fold
    - 나누어진 fold들의 label들이 같은 비율을 가지도록 구성
    - 분류 문제에 사용하기 좋음
'''
print('-'*15, 'Stratified K-Fold', '-'*15)
print()
from sklearn.model_selection import StratifiedKFold
X, y = load_iris(return_X_y=True)

s_kfold = StratifiedKFold(n_splits=3)
s_kfold_gen = s_kfold.split(X, y)

print(next(s_kfold_gen))

idx1, idx2 = next(s_kfold_gen)
print('dix1의 유니크값과 그 갯수', np.unique(y[idx1], return_counts=True))
print('dix2의 유니크값과 그 갯수', np.unique(y[idx2], return_counts=True))

##### 교차 검증 ####
X, y = load_iris(return_X_y=True)
s_kfold = StratifiedKFold(n_splits=3)
gen = s_kfold.split(X, y)

# 교차검증 평가결과들을 저장할 리스트
acc_train_list = []
acc_test_list = []

for train_idx, test_idx in gen:
    #1. 데이터 분리 작업
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    #2. 모델 생성
    tree = DecisionTreeClassifier(random_state=0)
    
    #3. 학습
    tree.fit(X_train, y_train)
    
    #4. 평가
    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    
    acc_train_list.append(accuracy_score(y_train, pred_train))
    acc_test_list.append(accuracy_score(y_test, pred_test))

print(f'train 정확도 : {np.mean(acc_train_lst)}, test 정확도 : {np.mean(acc_test_lst)}')


''' 
cross_val_score() 함수
    - 데이터셋을 k개로 나누고 k번 반복하면서 평가하는 작업을 처리해주는 함수
    - 주요 매개 변수
        - estimator : 학습할 평가 모델 객체
        - X : feature
        - y : label
        - scoring : 평가 지표
        - cv : 나눌 개수(k)
        - n_jobs : cpu를 몇개 사용할지 (-1 : 모든 cpu 사용)
    - 반환값
        - array : 각 반복마다의 평가 점수
'''
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
tree = DecisionTreeClassifier(random_state=0)
score_lst = cross_val_score(tree, X, y, scoring='accuracy', cv=3, n_jobs=-1)

print('score_lst의 타입 : ', type(score_lst))
print('score_lst = ', score_lst)
print('score_lst의 평균값 : ', np.mean(score_lst))