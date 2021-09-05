'''
앙상블 기법

1. 앙상블 기법의 정의 및 설명
    - 여러 모델을 학습시켜 결합하는 방식으로 문제를 해결하는 방식
    - 개별로 학습한 여러 모델을 조합하여 과적합을 방지하고 일반화 성능을 향상시킬 수 있다
    - 개별 모델의 성능이 확보되지 않았을 때, 성능 향상에 도움을 줄 수 있다

2. 앙상블의 종류
    1) 투표 방식
        (1)설명 : 여러개의 추정기(estimator)가 낸 결과들을 투표를 통해 최종 결과를 내는 방식
        (2)종류 
            - Bagging : 같은 유형의 알고리즘을 조합 / 각각 학습하는 데이터들을 다르게
                - Random Forest : 결정 트리를 기반으로 한다
                    - 다수의 결정 트리를 사용하여 성능을 올린 앙상블 알고리즘의 한 종류
                        - 학습 데이터를 샘플링, 다수의 결정트리 생성 >> 다수결로 결과를 결정
                        - 다수의 결정 트리를 만드는데서 랜덤포레스트라 칭함
                    - 장점 : 처리 속도가 빠름 / 분류 성능 높음
                    - 절차 
                        - 모든 결정 트리를 서로 다르게 만듦
                            - 각 트리는 부트스트랩 샘플링(중복 허용 + 랜덤하게 샘플링)으로 데이터 셋 준비
                            - 총 데이터 수는 원래 데이터셋과 동일 but 일부는 누락, 일부는 중복
                            - 각각의 트리는 전체 피쳐 중 일부의 피쳐만 랜덤하게 가지게 된다
                        - 트리별로 예측 결과를 낸다 
                            - 분류 : 예측을 모아 다수결 투표로 클래스 결과글 냄
                            - 회귀 : 예측 결과의 평균
                    - 주요 하이퍼 파라미터
                        - n_estimators : tree의 갯수(시간과 메모리가 허락하는 한 클수록 좋음)
                        - max_features : 각 트리에서 선택할 feature의 수(값이 클수록 feature 차이가 사라짐)
                        - max_depth, min_samples_leaf 등
                            - DecisionTreeClassifier의 파라미터들
                            - 트리의 최대 깊이, 가지를 치기 위한 최소 샘플 수 등 
                            - 결정 트리에서 과적합을 막기 위한 파라미터들을 랜덤 포레스트에 적용 가능

            - Voting : 서로 다른 종류의 알고리즘을 결합
    2) 부스팅(Boosting)
        (1)설명 : 약한 학습기(weak learner)들을 결합하여 보다 정확하고 강력한 학습기(strong learner)를 만든다
                  약한 학습기들은 순서대로 일을 하며 뒤의 학습기들은 앞의 학습기가 찾지 못한 부분을 추가적으로 찾는다

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
data = pd.read_csv('../data/wine.csv')

y = data['color']
X = data.drop(columns=['color', 'quality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

random_forest = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0, max_features=10, n_jobs=-1)
    #n_esimators(생성할 트리의 갯수(최소 200개 이상이 좋다))
random_forest.fit(X_train, y_train)

pred_train = random_forest.predict(X_train)
pred_test = random_forest.predict(X_test)
prob_train = random_forest.predict_proba(X_train)
prob_test = random_forest.predict_proba(X_test)

print('accuracy score with train / test')
print(accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test))
print('roc_auc_score with train / test')
print(roc_auc_score(y_train, prob_train[:, 1]), roc_auc_score(y_test, prob_test[:, 1]))

print(random_forest.feature_importances_)
    # [0.01052587 0.03745766 0.00194764 0.00769474 0.37191699 0.00093869
    #  0.51464401 0.02732205 0.00939757 0.01625858 0.0018962 ]

fi = pd.Series(random_forest.feature_importances_, index = X.columns)
print(fi.sort_values(ascending=False))


# 유방암 데이터 셋 활용하여 random forest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV

data = load_breast_cancer()
X, y = data['data'], data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

ran_foreset = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1, random_state=0)
ran_foreset.fit(X_train, y_train)

pred_train = ran_foreset.predict(X_train)
pred_test = ran_foreset.predict(X_test)

print('acc_score_train = {}, acc_score_test = {}'.format(accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)))

fi = pd.Series(ran_foreset.feature_importances_, index = data['feature_names'])

fi.sort_values().plot(kind='barh', figsize=(6,7))
plt.show()

##GridSearch 이용해서 최적의 파라미터 찾기
param = {
    'n_estimators' : [100, 200, 300, 400, 500, 600, 700],
    'max_depth' : range(2,7),
    'max_features' : range(4, 30, 2)
}

rf = RandomForestClassifier(random_state=0)
gs = GridSearchCV(rf, param, scoring='accuracy', cv=5, n_jobs=-1)

gs.fit(X_train, y_train)

df = pd.DataFrame(gs.cv_results_)
print(df.head())

print(gs.best_score_)
    # 0.967168262653898

cv_df = pd.DataFrame(gs.cv_results_)
print(cv_df)
'''
     mean_fit_time  std_fit_time  mean_score_time  std_score_time  ... split4_test_score mean_test_score std_test_score rank_test_score
0         0.573641      0.098059         0.031251        0.024207  ...          0.952941        0.941395       0.020680             451
1         1.150848      0.190867         0.073575        0.011716  ...          0.952941        0.943721       0.017006             443
2         2.021078      0.040127         0.156491        0.017123  ...          0.964706        0.946074       0.018834             417
3         2.818092      0.140671         0.178106        0.015988  ...          0.964706        0.946074       0.018834             417
4         3.662164      0.257999         0.246877        0.006250  ...          0.964706        0.946074       0.018834             417
..             ...           ...              ...             ...  ...               ...             ...            ...             ...
450       1.740644      0.028981         0.059376        0.006250  ...          0.976471        0.957756       0.015928             148
451       2.312526      0.055902         0.078126        0.000001  ...          0.976471        0.955404       0.018811             267
452       2.885898      0.061078         0.103126        0.007655  ...          0.976471        0.960109       0.014065              46
453       3.471914      0.073554         0.128126        0.006250  ...          0.976471        0.962462       0.011439              15
454       3.912544      0.158302         0.143752        0.015309  ...          0.976471        0.960109       0.014065              46
'''