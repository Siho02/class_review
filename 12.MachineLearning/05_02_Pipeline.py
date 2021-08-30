'''
파이프 라인(Pipeline)
0. 개요
    - 여러 단개의 머신러닝 프로세스(전처리의 각 단계 및 모델 생성, 학습 등) 처리 과정을 설정하여 한번에 처리 되도록 한다
    - 여러개의 변환기와 마지막에 변환기 or 추정기를 넣을 수 있음(추정기는 마지막에만 넣을 수 있다)
    - 전처리 작업 파이프라인 = 변환기들로만 구성
    - 전체 프로세스 파이프라인 = 변환기s + 추정기

1. 파이프라인 생성
    1) list 형태(이름, 변환기)로 묶어 전달
    2) 마지막에는 추정기가 올 수 있다.

2. 파이프라인을 통한 학습
    1) pipeline.fit()
        - 각 순서대로 각 변환기의 fit_transform()이 실행 되고, 그 결과가 다음 단계로 전달
        - 마지막 단계에서는 fit()만 호출 된다
        - 보통 마지막이 추정기일 때 사용

    2) pipeline.fit_transform()
        - fit()과 동일하나 마지막 단계에서도 fit_transform()이 실행
        - 보통 전처리 작업 파이프라인(모든 단계가 변환기)일 때 사용

    3) 파이프라인의 마지막이 추정기(모델)인 경우
        - predict(X), predict_proba(X)
        - 추정기를 이용하여 X에 대한 결과 추론
        - 모델 앞에 있는 변환기를 이용해서 transform() 처리 결과를 다음 단계로 전달
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

#데이터 로드 및 분할
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# 학습 : Feature 전처리(standard scaler) >> 모델학습(SVC)
# 추론 : Feature 전처리(StandardScaler) >> 추론(predict-SVC)
# Pipeline에 넣어줄 변환기와 추정기(모델)들을 순서에 맞춰서 List에 담아준다.

order = [
    ('scaler', StandardScaler()),   #(이름, 객체(변환기))
    ('svc', SVC())
]

pipeline = Pipeline(order, verbose=True)     ##verbose: 학습/추론할 때 로그를 출력

# 학습 (마지막이 추정기일 경우->fit(X, y), 모두 변환기일 경우-> fit_transform(X))
# 1 of 2) X_train을 표준화 (scaler.fit_transform(X_train))
# 2 of 2) 표준화 변환된 X값을 이용해서 SVC를 학습 (svc.fit(X_train_scaled, y_train))
pipeline.fit(X_train, y_train)
'''
[Pipeline] ............ (step 1 of 2) Processing scaler, total=   0.0s
[Pipeline] ............... (step 2 of 2) Processing svc, total=   0.0s
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())], verbose=True)
'''

# 추론
pred_train = pipeline.predict(X_train)
pred_test = pipeline.predict(X_test)
# 1 of 2: X_train을 표준화 처리. (scaler.transform(X_train))
# 2 of 2: 추정기를 이용해서 추론 (svc.predict(X_train_scaled))

print(accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test))
    #0.9929577464788732 0.958041958041958

new_data = X_test[:3]   #new_data.shape = (3, 30)

#transform() >> predict()
pipeline.predict(new_data)  #[1 0 0]



#### Grid Search에서 pipeline 사용

# SVC모델의 최적의 하이퍼파라미터( C, gamma)를 찾기 -> GridSearch
# StandardScaler를 이용해서 전처리
from sklearn.model_selection import GridSearchCV

#파이프라인 생성
order = [
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=0))
]

pipeline = Pipeline(order)

param = {
    'svc__C' : [0.001, 0.01, 0.1, 1, 10],
    'svc__gamma' : [0.001, 0.01, 0.1, 1, 10]
}

gs = GridSearchCV(pipeline, param, scoring='accuracy', cv=4, n_jobs=-1)

gs.fit(X_train, y_train)

print('\ngs.best_praams', gs.best_params_)
print('\ngs_best_score_',gs.best_score_)
#print('\ngs.cv_results_', gs.cv_results_)

best_model = gs.best_estimator_
print(best_model, type(best_model))


### make_pipeline() 함수를 이용한 파이프라인 생성
from sklearn.pipeline import make_pipeline

# make_pipeline(변환기객체, 변환기객체, ....., 추정기): Pipeline을 생성해서 반환
# 프로세스의 이름을 프로세스클래스이름(소문자로변환)으로 해서 Pipeline을 생성.
pipeline2 = make_pipeline(StandardScaler(), SVC())
print('\npipeline2의 데이터 타입 : ', type(pipeline2))

    