'''
Voting

1. Voting의 유형
    1) Hard voting : 다수의 추정기가 결정한 예측값들 중 많은 것을 선택하는 방식

    2)Soft voting : 다수의 추정기에서 각 레이블 별 예측한 확률들의 평균을 내 높은 레이블 값을 결과값으로 선택하는 방식
        - 일반적으로 soft voting의 성능이 더 좋다
    
    3) Voting은 성향이 다르면서 비슷한 성능을 가진 모델들을 묶었을 때, 가장 좋은 성능을 낸다.

2. VotingClassifier 클래스 이용
    1) 매개 변수
        - estimators : 앙상블할 모델들 설정("추정기이름", 추정기)의 튜플을 리스트로 묶어 전달
        - voting : voting 방식(hard(기본값), soft) 지정
'''

from scipy.sparse.construct import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC와 KNN은 scaling된 feature를 사용
# RandomForest는 raw data 사용

# 모델 객체 생성
svc = SVC(random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)

# 모델 학습
svc.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# 모델 평가
pred_train_svc = svc.predict(X_train_scaled)
pred_train_knn = knn.predict(X_train_scaled)
pred_train_rf = rf.predict(X_train)

pred_test_svc = svc.predict(X_test_scaled)
pred_test_knn = knn.predict(X_test_scaled)
pred_test_rf = rf.predict(X_test)


# 평가 함수
def print_metrics(y, pred, title=None):
    acc = accuracy_score(y, pred)
    if title:
        print(title)
    print('정확도 : ', acc)

print('-'*20, 'train data에 대한 accuracy score','-'*20)
print_metrics(y_train, pred_train_svc, 'train SVC')
print_metrics(y_train, pred_train_knn, 'train KNN')
print_metrics(y_train, pred_train_rf, 'train RF')

print('-'*20, 'test data에 대한 accuracy score','-'*20)
print_metrics(y_test, pred_test_svc, 'test SVC')
print_metrics(y_test, pred_test_knn, 'test KNN')
print_metrics(y_test, pred_test_rf, 'test RF')

#Voting
estimators = [
    ('svc', svc),
    ('knn', knn),
    ('rf', rf)
]

#votingclassifier 모델 생성
voting = VotingClassifier(estimators)
#각 모델 학습
voting.fit(X_train_scaled, y_train)

pred_train = voting.predict(X_train_scaled)
pred_test = voting.predict(X_test_scaled)

print('-'*30)
print_metrics(y_train, pred_train)
print_metrics(y_test, pred_test)




### soft voting 해보기
estimators2 = [
    ('svc', SVC(random_state=0, probability=True)),
    ('knn', knn),
    ('rf', rf)
]

voting2 = VotingClassifier(estimators2, voting='soft')
voting2.fit(X_train_scaled, y_train)

pred_train = voting2.predict(X_train_scaled)
pred_test = voting2.predict(X_test_scaled)
print('-'*30)
print_metrics(y_train, pred_train)
print_metrics(y_test, pred_test)


## Pipeline을 이용해 전처리+모델을 묶어서 VotingClassifier에 설정
# SVC, KNN : feature scaling 필요 >> Pipeline
# RF : feature scaling X
from sklearn.pipeline import Pipeline, make_pipeline
order_knn = [
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]
order_svc = [
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=0, probability=True))
]

knn_pipeline = Pipeline(order_knn)
svc_pipeline = Pipeline(order_svc)
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)

estimators3 = [
    ('knn', knn_pipeline),
    ('svc', svc_pipeline),
    ('rf', rf)
]

voting3 = VotingClassifier(estimators3, voting='soft')

voting3.fit(X_train, y_train)

print('-'*30)
print_metrics(y_train, voting3.predict(X_train))
print_metrics(y_test, voting3.predict(X_test))

print(voting3.predict(X_train))

