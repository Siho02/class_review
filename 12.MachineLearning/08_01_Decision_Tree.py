'''
의사 결정 나무(Decision Tree)
0. 개요
    - '스무 고개'와 비슷한 형식의 알고리즘
    - 데이터를 분류할 수 있는 질문을 던져 대상을 좁혀감
    - 분기 해나가는 구조가 나무의 가지와 비슷하여 Decision Tree라고 함
        - 불순도를 최대한 감소시키는 방향으로 조건을 만들어 학습을 진행
            - 순도와 불순도
                -서로 다른 종류의 값들이 섞여 있는 비율
                -한 종류의 클래스(class)가 많을 수록 순도가 높다.
        - 하위 노드는 yes / no 로 구분
    - white box 모델 >> 결과 해석이 가능
    - 과대적합 문제가 발생하기 쉽다
    - Random Forest(앙상블 기반 알고리즘)와 많은 Boosting 기반 앙상블 모델들의 기반 알고리즘으로 사용 된다.

1. 용어
    1) Root Node : 시작 node
    2) Deicision Node : 중간 node (intermediate node)
    3) Leaf Node : 마지막 단계의 노드, 최종 결과를 가짐

2. 과대 적합 문제
    1) 모든 데이터 셋이 모두 잘 분류 되도록 불순도가 0이 될 때까지 분기
    2) root에서부터 하위 노드가 많이 만들어짐 >> 모델이 복잡도 상승 >> 과대적합 가능성 상승
    3) Pruning(가지치기) : 과대 적합 문제를 막기 위해 적당한 시점에 하위 노드가 생성되지 않도록 막는 것

3. Hyper Parameter
    1) Pruning 관련
        (1) max_depth : 최대 깊이
        (2) max_leaf_depth : 생성 될 최대 leaf node의 개수 제한
        (3) min_samples_leaf : 가지를 칠 최소 sample 수 >> sample 수가 지정한 값보다 작으면 불순도와 관계 없이 가지 치기를 하지 않음
    2) Criterion(판단 기준)
        (1) gini(기본값)
        (2) entropy

4. Feature(컬럼) 중요도 조회
    1) Feature_importances_ 속성
        - 모델을 만들 때 각 feature의 중요도를 반환
        - input data 에서 중요한 feature를 찾기 위해 decision tree를 이용하기도 함
'''

## Wine Color 분류하기
# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# feature : 와인의 화학 성분
    # fixed acidity : 고정 산도 / volatile acidity : 휘발성 산도 / citric acid : 시트르산 / residual sugar : 잔류 당분
    # chlorides : 염화물 / free sulfur dioxide : 자유 이산화황 / total sulfur dioxide : 총 이산화황 / density : 밀도
    # pH : 수소 이온 농도 / sulphates : 황산염 /alcohol : 알콜
    # quality : 와인의 등급(A/B/C)
# target : color(0:화이트 / 1:레드)

import pandas as pd
import numpy as np

wine = pd.read_csv('../data/wine.csv')

print(wine.info()) #wine 데이터 프레임의 정보

print('화이트와인과 레드와인의 비율 계산')
print(wine['color'].value_counts() / len(wine))

#feature 데이터
X = wine.drop(columns='color')
#target 데이터
y = wine['color']

#quality의 데이터는 A, B, C로 범주형 데이터 >> 숫자로 변환하는 전처리 과정이 필요
from sklearn.preprocessing import LabelEncoder

#print('변환 전')
#print(X.head(3),'\n')
le = LabelEncoder()
X['quality'] = le.fit_transform(X['quality'])
#print('변환 후')
#print(X.head(3), '\n')
le.classes_ #array(['A', 'B', 'C'], dtype=object)

#데이터 분할 / 학습 / 테스트을 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, plot_roc_curve

#결정 나무 모델은 연속형 변수의 featuring scaling 전처리 하지 않음
#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# 모델 생성
tree = DecisionTreeClassifier(random_state=0)
# 모델 학습
tree.fit(X_train, y_train)
# 모델을 이용한 추론
pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)
proba_train = tree.predict_proba(X_train)[:, 1]
proba_test = tree.predict_proba(X_test)[:, 1]

print("\n정확도 계산(train / test 순)")
print(accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test))

print("\nroc-auc score")
print(roc_auc_score(y_train, proba_train), roc_auc_score(y_test, proba_test))

## 그래프 그리기
import matplotlib.pyplot as plt
_, ax = plt.subplots(1,1,figsize=(8,6))
plot_roc_curve(tree, X_train, y_train, ax=ax, name='Train')
plot_roc_curve(tree, X_test, y_test, ax=ax, name='Test')
plt.show()

## tree 구조를 시각화 하기
from sklearn.tree import export_graphviz
from graphviz import Source

graph = Source(export_graphviz(tree, out_file=None, feature_names=X.columns, class_names=['White', 'Red'], rounded=True, filled=True))
#print(graph)

# Feature 중요도 (학습이 끝난 tree모델.feature_importances_)
featureimportance = tree.feature_importances_
print(featureimportance.shape) #feature별 중요도를 순서대로 배열에 넣어서 반환

fi_s = pd.Series(featureimportance, index = X.columns)
fi_s.sort_values().plot(kind='barh', figsize=(8,6))
plt.show()
    #total sulfur dioxide    0.686318로 가장 중요도가 높은 것을 알 수 있다



#### Grid Search CV를 

# 가지치기(모델 복잡도 관련 규제) 파라미터
# max_depth, max_leaf_nodes, min_samples_leaf 최적의 조합 - GridSearch
# best_estimator_ 를 이용해서 feature 중요도를 조회 + graphviz

from sklearn.model_selection import GridSearchCV

param = {
    'max_depth' : range(3,11),
    'max_leaf_nodes' : range(10, 31, 5),
    'min_samples_leaf' : range(100, 1000, 100)
}

tree = DecisionTreeClassifier(random_state=0)
gs = GridSearchCV(tree, param, scoring='accuracy', cv=5, n_jobs=-1)

gs.fit(X_train, y_train)

print('\nfitting 완료')

df = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score').head()
print(df)

best_model = gs.best_estimator_
print(best_model)

graph = Source(export_graphviz(best_model, out_file=None, feature_names=X.columns, class_names=['White','Red'], rounded=True, filled=True))

#feature importance 조회
feat_impor = pd.Series(best_model.feature_importances_, index=X.columns)
feat_impor.sort_values()

feat_impor.sort_values().plot(kind = 'barh')
plt.show()