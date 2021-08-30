'''
1. 일반화, 과대적합, 과소적합 - 모델학습의 세 종류
    1) 일반화(Generalization)
        - 모델이 새로운 데이터셋에 대하여 정확히 예측한 경우
        - 모델이 훈련 데이터로 평가한 결과와 테스트 데이터로 평가한 결과의 차이가 거의 없어 좋은 평가 지표를 보여준다

    2) 과대적합(Overfitting)
        - 모델이 훈련 데이터에 대한 예측 성능은 높지만, 새로운 데이터(test data)에 대해서는 예측 성능이 떨어지는 경우
        - 훈련 데이터의 특성에 과도하게 맞추어 학습하여 새로운 데이터에 대한 성능이 부족한 경우
        - 발생 원인 및 해결
            - 원인 : 학습 데이터 양에 비해 모델이 복잡 >> 모델을 단순하게 제작
            - 해결 
                - 데이터 양을 늘린다 >> 현실적으로 어려움
                - 각 모델의 규제와 관련된 hyper parameter를 조절

    3) 과소적합(Underfitting)
        - 모델이 훈련 데이터와 테스트 데이터에서 모두 예측 성능이 떨어지는 경우
        - 모델이 너무 간단하여 훈련 데이터에 대한 학습이 충분치 못하고 데이터 셋의 패턴을 제대로 찾지 못하는 경우이다
        - 발생 원인 및 해결
            - 원인 : 데이터 양에 비해 모델이 단순 
            - 해결 : 복잡한 모델을 이용


2. Decision Tree의 복잡도 제어
    1) Decision Tree의 복잡도
        - 노드가 너무 많은 경우 복잡도가 상승
        - 적절한 시점에 트리 생성을 중단해야 한다
    2) 모델의 복잡도 관련 hyper parameter
        - max_depth : 트리의 최대 깊이
        - max_leaf_nodes : 리프 노드의 갯수
        - min_samples_leaf : leaf 노드가 되기 위한 최소 샘플 수
        - min_samples_split : 나누는 최소 샘플수
        
3. GridSearch를 통한 hyper parameter tunning
    1) Parameter
        - parameter : 머신 러닝 모델에서 파라미터는 모델이 학습을 통해서 직접 찾아야 하는 값
        - hyper parameter : 모델의 학습에 영향을 미치는 파라미터 값으로 모델 생성 시 사람이 직접 지정
        - hyper parameter tunning : 가장 성능이 좋은 하이퍼 파라미터를 찾는 것

    2) 최적의 하이퍼 파라미터 찾기
        (1) 만족할만 한 하이퍼 파라미터들의 값의 조합을 찾을 때까지 수동으로 조정
        (2) GridSearch 사용
            - GridSearchCV()
                - 시도해볼 하이퍼파라미터들을 지정하면 모든 조합에 대해 교차검증 후 제일 좋은 성능을 내는 하이퍼파라미터 조합을 찾아준다.
                - 적은 수의 조합의 경우는 괜찮지만 시도할 하이퍼파라미터와 값들이 많아지면 너무 많은 시간이 걸린다.

                - 주요 매개 변수
                    - estimator : 모델 객체 지정
                    - paramas : 하이퍼 파라미터 목록을 dictionary로 전달 / '파라미터명' : [파라미터값 리스트] 
                    - scoring : 평가 지표
                        - 평가 지표 문자열 : https://scikit-learn.org/stable/modules/model_evaluation.html
                        - 여러 개일 경우 list로 묶어서 지정
                    - cv : 교차검증시 fold 개수
                    - n_jobs : 사용할 CPU 코어 개수(None : CPU 1개 사용 / -1 : 모든 CPU 코어 사용)
                
                - 메소드
                    - fit(X, y) : 학습
                    - predict(X) : 제일 좋은 성능을 낸 모델로 predict()
                    - predict_proba(X) : 제일 좋은 성능을 낸 모델로 predict_proba() 호출
                
                - 결과 조회 속성
                    - cv_result : 파라미터 조합 별 결과 조회
                    - best_paramas_ : 가장 좋은 성능을 낸 파라미터 조합 조회
                    - best_estimator_ : 가장 좋은 성능을 낸 모델 반환
                    - best_score_ : 가장 좋은 점수 반환

        (3) Random Search 사용
            - RandomizedSearchCV()
                - GridSearch와 동일한 방식으로 사용한다.
                - 모든 조합을 다 시도하지 않고 각 반복마다 임의의 값만 대입해 지정한 횟수만큼만 평가한다.

'''

#위스콘신 유방암 데이터셋
from scipy.sparse.construct import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

tree1 = DecisionTreeClassifier(random_state=0)
tree2 = DecisionTreeClassifier(max_depth=1, random_state=0)
tree3 = DecisionTreeClassifier(max_depth=3, random_state=0)

tree1.fit(X_train, y_train)
tree2.fit(X_train, y_train)
tree3.fit(X_train, y_train)

pred_train1 = tree1.predict(X_train)
pred_test1 = tree1.predict(X_test)
pred_train2 = tree2.predict(X_train)
pred_test2 = tree2.predict(X_test)
pred_train3 = tree3.predict(X_train)
pred_test3 = tree3.predict(X_test)

print('max_depth의 차이에 따른 정확도 차이')
print('max_depth : None')
print("Train 정확도: {}, Test 정확도: {}\n".format(accuracy_score(y_train, pred_train1), accuracy_score(y_test, pred_test1)))
print('max_depth : 1')
print("Train 정확도: {}, Test 정확도: {}\n".format(accuracy_score(y_train, pred_train2), accuracy_score(y_test, pred_test2)))
print('max_depth : 3')
print("Train 정확도: {}, Test 정확도: {}\n".format(accuracy_score(y_train, pred_train3), accuracy_score(y_test, pred_test3)))

#print(X_test)
#print(X_test[0])

from sklearn.tree import export_graphviz
from graphviz import Source

graph = Source(export_graphviz(tree3, out_file=None, feature_names=data['feature_names'], class_names=data['target_names'], rounded=True, filled=True))

graph   #<graphviz.files.Source at 0x1c4cac23850>

#######################################################
# GridSearch
DecisionTreeClassifier(max_depth=1, max_leaf_nodes=5, min_samples_split=20)

max_depth_candidate = [1,2,3,4,5]
max_leaf_nodes = [5,7,10,20]
min_sample_splits = [20,30,40,50]

train_acc_lst = []
test_acc_lst = []

#max_depth를 1,2,3,4,5로 넣어서 accuracy를 구하여 저장하는 반복문
for max_dep in max_depth_candidate:
    tree = DecisionTreeClassifier(max_depth = max_dep, random_state=0)
    tree.fit(X_train, y_train)

    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)

    train_acc_lst.append(accuracy_score(y_train, pred_train))
    test_acc_lst.append(accuracy_score(y_test, pred_test))

import pandas as pd
acc_df = pd.DataFrame({
    'max depth' : max_depth_candidate,
    'train accuracy' : train_acc_lst,
    'test accuracy' : test_acc_lst
})

'''
acc_df

   max depth  train accuracy  test accuracy
0          1        0.929577       0.888112
1          2        0.931925       0.888112
2          3        0.976526       0.916084
3          4        0.985915       0.909091
4          5        1.000000       0.902098
'''

#선 그래프로 변화를 시각화 하기
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

plt.plot(max_depth_candidate, train_acc_lst, marker='o', label = 'Train')
plt.plot(max_depth_candidate, test_acc_lst, marker='o', label = 'Test')

plt.ylabel('accuracy')
plt.xlabel('max depth(model complexibity)')
plt.xticks(range(1,6))
plt.legend()
plt.show()

#### GridSearchCV ####
from sklearn.model_selection import GridSearchCV

# 모델
tree = DecisionTreeClassifier(random_state=0)

# 하이퍼 파라미터 후보 - 딕셔너리 (key:하이퍼파라미터(str), value: 리스트(후보 파라미터 값들))
param = {
    "max_depth":[None, 1, 2, 3, 4, 5], 
    "max_leaf_nodes":[3, 5, 7, 9]  #range(3,10,2)
}

#GridSearchCV 생성
gs = GridSearchCV(tree, param, scoring='accuracy', cv=3, n_jobs=-1)
# tree : GridSearch를 적용할 모델 / param : 하이퍼파라미터 후보 딕셔너리 / scoring : 평가지표 / cv = 3 cross validation 시 3개의 fold로 나눔 / n_jobs = 1 : 모든 cpu 사용)
# 조합 갯수 : 6 * 4 * 3

#모델 학습
gs.fit(X_train, y_train)

#모든 경우 별 결과를 확인
gs.cv_results_
'''
{'mean_fit_time': array([0.00764243, 0.02170674, 0.01405009, 0.03784569, 0.014949  ,
        0.01398651, 0.01967661, 0.01669542, 0.01847196, 0.00365527,
        0.00332387, 0.01031701, 0.00532349, 0.00498096, 0.00431951,
        0.00833996, 0.00383441, 0.00450754, 0.00515652, 0.006977  ,
        0.00397984, 0.00797796, 0.01131129, 0.00664298]),
 'std_fit_time': array([2.85591783e-03, 1.13894422e-02, 5.74118269e-03, 2.57211041e-02,
        1.55411763e-02, 4.21415710e-03, 8.37345925e-03, 8.18630193e-03,
        9.90210280e-03, 9.39312769e-04, 4.70302644e-04, 8.93383620e-03,
        1.24248041e-03, 8.64977289e-06, 4.66931139e-04, 3.17735687e-03,
        6.36860768e-04, 4.06532739e-04, 6.17777918e-04, 3.55345935e-03,
        8.20720105e-04, 3.73303889e-03, 2.05090141e-03, 2.04749180e-03]),
 'mean_score_time': array([0.00099842, 0.00456675, 0.0013291 , 0.00231862, 0.0009923 ,
        0.00806149, 0.00265296, 0.00332022, 0.00549928, 0.0009985 ,
        0.00066503, 0.00099897, 0.00033283, 0.00100287, 0.00066559,
        0.00099889, 0.00066026, 0.00066535, 0.00051149, 0.00066415,
        0.00033323, 0.00066527, 0.00542196, 0.00067043]),
 'std_score_time': array([9.79807218e-07, 2.62548733e-03, 4.69913295e-04, 4.58058413e-04,
        8.64246796e-06, 3.62030847e-03, 2.34212854e-03, 9.39070849e-04,
        5.68117643e-03, 3.37174788e-07, 4.70246478e-04, 3.37174788e-06,
        4.70696004e-04, 8.56170232e-06, 4.70640624e-04, 4.05233662e-07,
        4.66918884e-04, 4.70471382e-04, 4.07552431e-04, 4.69629101e-04,
        4.71257962e-04, 4.70417935e-04, 3.91914100e-03, 4.74111276e-04]),
 'param_max_depth': masked_array(data=[None, None, None, None, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
                    3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False],
        fill_value='?',
             dtype=object),
 'param_max_leaf_nodes': masked_array(data=[3, 5, 7, 9, 3, 5, 7, 9, 3, 5, 7, 9, 3, 5, 7, 9, 3, 5,
                    7, 9, 3, 5, 7, 9],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'max_depth': None, 'max_leaf_nodes': 3},
  {'max_depth': None, 'max_leaf_nodes': 5},
  {'max_depth': None, 'max_leaf_nodes': 7},
  {'max_depth': None, 'max_leaf_nodes': 9},
  {'max_depth': 1, 'max_leaf_nodes': 3},
  {'max_depth': 1, 'max_leaf_nodes': 5},
  {'max_depth': 1, 'max_leaf_nodes': 7},
  {'max_depth': 1, 'max_leaf_nodes': 9},
  {'max_depth': 2, 'max_leaf_nodes': 3},
  {'max_depth': 2, 'max_leaf_nodes': 5},
  {'max_depth': 2, 'max_leaf_nodes': 7},
  {'max_depth': 2, 'max_leaf_nodes': 9},
  {'max_depth': 3, 'max_leaf_nodes': 3},
  {'max_depth': 3, 'max_leaf_nodes': 5},
  {'max_depth': 3, 'max_leaf_nodes': 7},
  {'max_depth': 3, 'max_leaf_nodes': 9},
  {'max_depth': 4, 'max_leaf_nodes': 3},
  {'max_depth': 4, 'max_leaf_nodes': 5},
  {'max_depth': 4, 'max_leaf_nodes': 7},
  {'max_depth': 4, 'max_leaf_nodes': 9},
  {'max_depth': 5, 'max_leaf_nodes': 3},
  {'max_depth': 5, 'max_leaf_nodes': 5},
  {'max_depth': 5, 'max_leaf_nodes': 7},
  {'max_depth': 5, 'max_leaf_nodes': 9}],
 'split0_test_score': array([0.88732394, 0.90140845, 0.8943662 , 0.90140845, 0.86619718,
        0.86619718, 0.86619718, 0.86619718, 0.88732394, 0.90140845,
        0.90140845, 0.90140845, 0.88732394, 0.90140845, 0.90140845,
        0.90140845, 0.88732394, 0.90140845, 0.8943662 , 0.9084507 ,
        0.88732394, 0.90140845, 0.8943662 , 0.90140845]),
 'split1_test_score': array([0.88732394, 0.91549296, 0.91549296, 0.91549296, 0.90140845,
        0.90140845, 0.90140845, 0.90140845, 0.88732394, 0.9084507 ,
        0.9084507 , 0.9084507 , 0.88732394, 0.91549296, 0.91549296,
        0.91549296, 0.88732394, 0.91549296, 0.91549296, 0.91549296,
        0.88732394, 0.91549296, 0.91549296, 0.91549296]),
 'split2_test_score': array([0.9084507 , 0.93661972, 0.92253521, 0.92957746, 0.88732394,
        0.88732394, 0.88732394, 0.88732394, 0.9084507 , 0.93661972,
        0.93661972, 0.93661972, 0.9084507 , 0.93661972, 0.92253521,
        0.92253521, 0.9084507 , 0.93661972, 0.92253521, 0.92253521,
        0.9084507 , 0.93661972, 0.92253521, 0.92957746]),
 'mean_test_score': array([0.8943662 , 0.91784038, 0.91079812, 0.91549296, 0.88497653,
        0.88497653, 0.88497653, 0.88497653, 0.8943662 , 0.91549296,
        0.91549296, 0.91549296, 0.8943662 , 0.91784038, 0.91314554,
        0.91314554, 0.8943662 , 0.91784038, 0.91079812, 0.91549296,
        0.8943662 , 0.91784038, 0.91079812, 0.91549296]),
 'std_test_score': array([0.00995925, 0.01447046, 0.01196953, 0.01149995, 0.01447046,
        0.01447046, 0.01447046, 0.01447046, 0.00995925, 0.01521301,
        0.01521301, 0.01521301, 0.00995925, 0.01447046, 0.00878323,
        0.00878323, 0.00995925, 0.01447046, 0.01196953, 0.00574998,
        0.00995925, 0.01447046, 0.01196953, 0.01149995]),
 'rank_test_score': array([16,  1, 13,  5, 21, 21, 21, 21, 16,  5,  5,  5, 16,  1, 11, 11, 16,
         1, 13,  5, 16,  1, 13,  5])}
'''
#최적의 그리드서치 결과를 이용한 데이터 프레임 생성
result_df = pd.DataFrame(gs.cv_results_)
'''

mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_max_depth	param_max_leaf_nodes	params	split0_test_score	split1_test_score	split2_test_score	mean_test_score	std_test_score	rank_test_score
0	0.007642	0.002856	0.000998	9.798072e-07	None	3	{'max_depth': None, 'max_leaf_nodes': 3}	0.887324	0.887324	0.908451	0.894366	0.009959	16
1	0.021707	0.011389	0.004567	2.625487e-03	None	5	{'max_depth': None, 'max_leaf_nodes': 5}	0.901408	0.915493	0.936620	0.917840	0.014470	1
2	0.014050	0.005741	0.001329	4.699133e-04	None	7	{'max_depth': None, 'max_leaf_nodes': 7}	0.894366	0.915493	0.922535	0.910798	0.011970	13
3	0.037846	0.025721	0.002319	4.580584e-04	None	9	{'max_depth': None, 'max_leaf_nodes': 9}	0.901408	0.915493	0.929577	0.915493	0.011500	5
4	0.014949	0.015541	0.000992	8.642468e-06	1	3	{'max_depth': 1, 'max_leaf_nodes': 3}	0.866197	0.901408	0.887324	0.884977	0.014470	21
5	0.013987	0.004214	0.008061	3.620308e-03	1	5	{'max_depth': 1, 'max_leaf_nodes': 5}	0.866197	0.901408	0.887324	0.884977	0.014470	21
6	0.019677	0.008373	0.002653	2.342129e-03	1	7	{'max_depth': 1, 'max_leaf_nodes': 7}	0.866197	0.901408	0.887324	0.884977	0.014470	21
7	0.016695	0.008186	0.003320	9.390708e-04	1	9	{'max_depth': 1, 'max_leaf_nodes': 9}	0.866197	0.901408	0.887324	0.884977	0.014470	21
8	0.018472	0.009902	0.005499	5.681176e-03	2	3	{'max_depth': 2, 'max_leaf_nodes': 3}	0.887324	0.887324	0.908451	0.894366	0.009959	16
9	0.003655	0.000939	0.000998	3.371748e-07	2	5	{'max_depth': 2, 'max_leaf_nodes': 5}	0.901408	0.908451	0.936620	0.915493	0.015213	5
10	0.003324	0.000470	0.000665	4.702465e-04	2	7	{'max_depth': 2, 'max_leaf_nodes': 7}	0.901408	0.908451	0.936620	0.915493	0.015213	5
11	0.010317	0.008934	0.000999	3.371748e-06	2	9	{'max_depth': 2, 'max_leaf_nodes': 9}	0.901408	0.908451	0.936620	0.915493	0.015213	5
12	0.005323	0.001242	0.000333	4.706960e-04	3	3	{'max_depth': 3, 'max_leaf_nodes': 3}	0.887324	0.887324	0.908451	0.894366	0.009959	16
13	0.004981	0.000009	0.001003	8.561702e-06	3	5	{'max_depth': 3, 'max_leaf_nodes': 5}	0.901408	0.915493	0.936620	0.917840	0.014470	1
14	0.004320	0.000467	0.000666	4.706406e-04	3	7	{'max_depth': 3, 'max_leaf_nodes': 7}	0.901408	0.915493	0.922535	0.913146	0.008783	11
15	0.008340	0.003177	0.000999	4.052337e-07	3	9	{'max_depth': 3, 'max_leaf_nodes': 9}	0.901408	0.915493	0.922535	0.913146	0.008783	11
16	0.003834	0.000637	0.000660	4.669189e-04	4	3	{'max_depth': 4, 'max_leaf_nodes': 3}	0.887324	0.887324	0.908451	0.894366	0.009959	16
17	0.004508	0.000407	0.000665	4.704714e-04	4	5	{'max_depth': 4, 'max_leaf_nodes': 5}	0.901408	0.915493	0.936620	0.917840	0.014470	1
18	0.005157	0.000618	0.000511	4.075524e-04	4	7	{'max_depth': 4, 'max_leaf_nodes': 7}	0.894366	0.915493	0.922535	0.910798	0.011970	13
19	0.006977	0.003553	0.000664	4.696291e-04	4	9	{'max_depth': 4, 'max_leaf_nodes': 9}	0.908451	0.915493	0.922535	0.915493	0.005750	5
20	0.003980	0.000821	0.000333	4.712580e-04	5	3	{'max_depth': 5, 'max_leaf_nodes': 3}	0.887324	0.887324	0.908451	0.894366	0.009959	16
21	0.007978	0.003733	0.000665	4.704179e-04	5	5	{'max_depth': 5, 'max_leaf_nodes': 5}	0.901408	0.915493	0.936620	0.917840	0.014470	1
22	0.011311	0.002051	0.005422	3.919141e-03	5	7	{'max_depth': 5, 'max_leaf_nodes': 7}	0.894366	0.915493	0.922535	0.910798	0.011970	13
23	0.006643	0.002047	0.000670	4.741113e-04	5	9	{'max_depth': 5, 'max_leaf_nodes': 9}	0.901408	0.915493	0.929577	0.915493	0.011500	5

'''
#가장 좋은 모델 선택
best_model = gs.best_estimator_

#선택한 모델의 accuracy 점수
print(accuracy_score(y_test, best_model.predict(X_test)))
    #0.916083916083916

print(accuracy_score(y_test, gs.predict(X_test)))
    #0.916083916083916

## 즉, gs를 하면 알아서 가장 좋은 모델을 선택함

print(gs.best_params_)
    #{'max_depth': None, 'max_leaf_nodes': 5}
print(gs.best_score_)
    #0.9178403755868545


### randomsized search CV

# iris 데이터를 이용하여 랜덤사이즈서치
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = load_iris(return_X_y=True)

#총 조합수 : 10 * 10 * 2
param_iris = {
    'max_depth' : range(1,11),
    'max_leaf_nodes' : range(3, 31, 3),
    'criterion' : ['gini', 'entropy']
}

rs = RandomizedSearchCV(tree, param_iris, n_iter=50, cv=5, n_jobs=-1)
#n_iter : 총 조합의 갯수 중 50개만 랜덤하게 선택해서 테스트

rs.fit(X_train, y_train)

result_dict = rs.cv_results_
#print(result_dict)

result_df = pd.DataFrame(result_dict)
#print(result_df.head())
'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time  ... split4_test_score mean_test_score std_test_score rank_test_score
0       0.009375      0.007655         0.000000        0.000000  ...          0.976471        0.934282       0.041142              13
1       0.012500      0.011693         0.000000        0.000000  ...          0.952941        0.931929       0.032763              31
2       0.009376      0.007655         0.000000        0.000000  ...          0.964706        0.934282       0.034560              13
3       0.008222      0.008167         0.000801        0.001603  ...          0.976471        0.934282       0.041142              13
4       0.013008      0.002531         0.000601        0.000491  ...          0.976471        0.948372       0.019067               9
'''

#가장 좋은 점수 
print(rs.best_score_)
    # rank_test_score가 가장 작은 값의 mean_test_score / 0.95077975376197

print(rs.best_params_) 
    # rank_test_score가 가장 작은 값을 만드는 파라미터값들을 반환 / {'max_leaf_nodes': 9, 'max_depth': 10, 'criterion': 'entropy'}

best_model = rs.best_estimator_
print(best_model)
    # rank_test_score가 가장 작은 값을 만들도록 하는 모델 반환 / DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=9, random_state=0)


print(accuracy_score(y_test, best_model.predict(X_test)), accuracy_score(y_test, rs.predict(X_test)))
    # 0.9230769230769231 0.9230769230769231
    