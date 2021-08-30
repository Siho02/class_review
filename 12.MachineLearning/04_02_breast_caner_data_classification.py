'''
0. 필요한 것들 import
1.breast_cancer data 모델링
2.breast_cancer data 로딩
3.Train/Test set으로 분리
4.모델링 - DecisionTreeClassifier(max_depth=3), RandomForestClassifier(max_depth=2, n_estimators=200) : 하이퍼파라미터는 위와 동일
5. 평가 (Train/Test set)
6. 평가지표
7. accuracy_score, recall, precision, f1점수, confusion matrix: (정답, 예측분류)
8. PR curve 그리고 AP 점수 확인 : (정답, 양성확률)
9. ROC curve 그리고 AUC 점수 확인
'''
#0. 필요한 모듈 import
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve, average_precision_score, plot_roc_curve, roc_auc_score

#1  breast_cancer data 로딩
X, y = load_breast_cancer(return_X_y=True)  #데이터 셋 로딩

#2  Train/Test set으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)    #train, test 셋 분리

#3  모델링 - DecisionTreeClassifier(max_depth=3), RandomForestClassifier(max_depth=2, n_estimators=200) : 하이퍼파라미터는 위와 동일
tree = DecisionTreeClassifier(max_depth=3, random_state=0)  #tree 모델 생성
rf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)  #randomforest 모델 생성

#트리모델
tree.fit(X_train, y_train)  #모델 학습

pred_tree_train = tree.predict(X_train) #트리 모델 추론 
pred_test_train = tree.predict(X_test)

prob_tree_train = tree.predict_proba(X_train)[:, 1]    #트리 모델 양성 확률 계산
prob_tree_test = tree.predict_proba(X_test)[:, 1]

#랜덤포레스트모델
rf.fit(X_train, y_train)  #모델 학습
pred_rf_train = rf.predict(X_train)
pred_rf_test = rf.predict(X_test)

prob_rf_train = rf.predict_proba(X_train)[:, 1]
prob_rf_test = rf.predict_proba(X_test)[:, 1]

#4  평가 (Train/Test set)
#5  평가지표
#6  accuracy_score, recall, precision, f1점수, confusion matrix: (정답, 예측분류)
    #평가 결과를 출력하는 함수
def print_metrics(y, pred, title=None):
    #y : 정답, pred : 모델이 예측한 값, title : 어떤 모델에 대한 평과 결과인지에 대한 제목
    
    if title:
        print(title)
    acc = accuracy_score(y, pred)
    recall = recall_score(y, pred)
    precision = precision_score(y, pred)
    f1 = f1_score(y, pred)
    cm = confusion_matrix(y, pred)

    print('\t정확도={}, 재현율={}, 정밀도={}, f1점수={}'.format(acc, recall, precision, f1))
    print('\t혼동 행렬', cm)
print_metrics(y_train, pred_tree_train, title = 'DecisionTree train 평가지표')
print_metrics(y_test, pred_test_train , title = 'DecisionTree test 평가지표')
print_metrics(y_train, pred_rf_train , title = 'RandomForest train 평가지표')
print_metrics(y_test, pred_rf_test , title = 'RandomForest test 평가지표')

#7  PR curve 그리고 AP 점수 확인 : (정답, 양성확률)
_, ax = plt.subplots(1,1, figsize=(7,6))
plot_precision_recall_curve(tree, X_train, y_train, ax=ax, name = 'Decision Tree Train')
plot_precision_recall_curve(tree, X_test, y_test, ax=ax, name = 'Decision Tree Test')
plt.grid(True)
plt.show()
#이미지 범례에 ac 점수가 포함되어 있음
#average_precision_score(y_train, prob_train_tree), average_precision_score(y_test, prob_test_tree)
    #(0.9842102478389377, 0.9508563971094506)

_, ax = plt.subplots(1,1, figsize=(7,6))
plot_precision_recall_curve(rf, X_train, y_train, ax=ax, name='Random Forest Train')
plot_precision_recall_curve(rf, X_test, y_test, ax=ax, name='Random Forest Test')
plt.grid(True)
plt.show()
#이미지 범례에 auc 점수가 포함되어 있음(반올림 되어 있긴 함)
#average_precision_score(y_train, proba_train_rf), average_precision_score(y_test, proba_test_rf)
    #(0.9973840556546545, 0.977122126791544)

#8  ROC curve 그리고 AUC 점수 확인
_, ax = plt.subplots(1,1, figsize=(7,6))
plot_roc_curve(rf, X_train, y_train, ax=ax, name='Random Forest Train')
plot_roc_curve(rf, X_test, y_test, ax=ax, name='Random Forest test')
plt.grid(True)
plt.show()
#roc_auc_score(y_train, proba_train_tree), roc_auc_score(y_test, proba_test_tree)
    #(0.9863261093911249, 0.9446097883597883)
#roc_auc_score(y_train, proba_train_rf), roc_auc_score(y_test, proba_test_rf)
    #(0.9957481940144479, 0.9718915343915344)


from graphviz import Source
from sklearn.tree import export_graphviz
graph = Source(export_graphviz(tree, out_file=None))
print(graph)