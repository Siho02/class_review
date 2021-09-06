'''
    회귀(Regression)

1. 정의 : 지도 학습(supervised learning)으로 예측할 target이 연속형 데이터인 경우

2. 주요 평가지표
    - 예측값과 실제값 간의 차이
    1) MSE(mean squared error)
        - MSE = 1/n * ∑(y_real - y_pred)^2
        - 실제값과 예측값의 차를 제곱해 평균을 냄
        - 함수 : mean_squared_error()
        - 'neg_mean_squared_error'
    2) RMSE(root mean squared error)
        - RMSE = sqrt{1/n * ∑(y_real - y_pred)^2}
        - MSE의 경우 오차의 제곱합이므로 실제 오차의 평균보다 큰 값 >> MSE의 제곱근 = RMSE
        - scikit-learn에서 따로 함수를 지원하지는 않음 >> MSE를 구한 후 np.sqrt()로 구함
    3) R^2(R square, 결정계수)
        - R^2 = (∑(y_pred - y_mean)^2) / (∑(y_i - y_mean)^2)
        - 평균으로 예측했을 때, 오차(총오차)보다 모델을 사용했을 때, 얼만큼 더 좋은 성능을 내는지 비율로 나타낸 값
        - 1에 가까울 수록 성능이 좋은 모델
        - 함수 : r2_score()
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 회귀문제에 사용할 수 있는 가상의 데이터셋을 원하는 조건을 설정해서 생성하는 함수(make_xxxxx())
X, y = make_regression(n_samples = 100, n_features=1, n_informative=1, noise=30, random_state=0)
    # n_samples = 데이터 개수, n_features = feature(컬럼) 개수
    # n_informative = target(label)에 영향을 주는 feature의 개수, noise = 잡음(모델이 찾을 수 없는 값의 범위)

    #X.shape = (100, 1), y.shape(100, )

print(X[:5])
    #[[-0.35955316]
    # [ 0.97663904]
    # [ 0.40234164]
    # [-0.81314628]
    # [-0.88778575]]

print(y[:5])
    #array([-29.38797228, -18.77135911,   0.56377656,  19.90502386, -31.84342122])

plt.scatter(X, y)
plt.title('X, y`s scatter figure')
plt.show()

np.mean(y), np.min(y), np.max(y), np.median(y)
    #(0.1344193000112442, -141.9934839876259, 117.54737632470687, 1.2075194195120451)


#linear regression 모델을 이용해서 추론 및 모델 평가
## linear regression 모델 생성 및 학습
lr = LinearRegression()
lr.fit(X, y)
# lr모델을 이용한 예측
pred = lr.predict(X)

new_X = [[0.56], [1.2], [-0.7]]
print('lr모델을 이용한 예측값 : ', lr.predict(new_X))
    #[ 21.68633197  49.26219191 -32.60364229]

#모델 평가(MSE, RMSE, R2)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, pred)
print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}\n")

#cross validation
lr2 = LinearRegression()
score_lst = cross_val_score(lr2, X, y, cv=3, scoring='r2')
print('fold를 3개로 했을 때 나오는 r2값 : ', score_lst, '평균값 : ', np.mean(score_lst))

score_lst2 = cross_val_score(lr2, X, y, cv=3, scoring="neg_mean_squared_error") #,MSE * -1)
print('fold를 3개로 했을 때 나오는 neg_mean_square_error 값 : ', score_lst2*(-1), '평균값 : ', np.mean(score_lst2)*(-1))

#rmse
print('fold를 3개로 했을 때 나오는 rmse 값 : ', np.sqrt(score_lst2*(-1)), '평균값 : ', np.sqrt(np.mean(score_lst2)*(-1)), '\n')


lr4 = LinearRegression()
lr4.fit(X, y)

print(lr4.coef_, lr4.intercept_) #coef:기울기, intercept:절편

def my_pred(X):
    #print(lr4.coef_[0] * X + lr4.intercept_)
    return lr4.coef_[0] * X + lr4.intercept_

#위에서 생성한 new_X를 array형태로 변환 후 예측함수에 넣어보기
new_X = np.array(new_X) 
#print('my_pred 함수를 통해 예측한 값')
#my_pred(new_X)

print('linear regression모델을 이용해 예측한 값')
print(lr.predict(new_X))

#X, y, 예측결과를 시각화
plt.figure(figsize=(6,7))
plt.scatter(X,y)
y_hat = my_pred(X)  #lr.pred 사용x
plt.plot(X, y_hat, color='red')
plt.show()


### 기존 분류 모델의 회귀 모델
from sklearn.model_selection import train_test_split

# XXXXClassifier (분류), XXXXRegressor(회귀)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #stratify: 회귀에서는 지정하지 않는다.

knn = KNeighborsRegressor()
tree = DecisionTreeRegressor()
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
svr = SVR()

est = [
    ("knn", knn), 
    ("tree", tree),
    ("rf", rf),
    ('GB', gb),
    ("svr",svr)
]

def print_regression_metrics(y, pred, title=None):
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, pred)
    if title:
        print(title)
    print(f'MSE : {mse}, RMSE : {rmse}, R2 : {r2}')

for model_name, model in est:
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print_regression_metrics(y_train, pred_train, title=f'{model_name}-Train')
    print_regression_metrics(y_test, pred_test, title=f'{model_name}-Test')
    print('-'*40)


#Voting을 이용한 회귀
vote = VotingRegressor(est)
vote.fit(X_train, y_train)

print_regression_metrics(y_train, vote.predict(X_train), title="Voting Train")
print_regression_metrics(y_test, vote.predict(X_test), title="Voting Test")
print('-' * 40)


#decision tree 회귀
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz
from graphviz import Source
graph = Source(export_graphviz(tree, out_file=None, rounded=True, filled=True))

print_regression_metrics(y_train, tree.predict(X_train), title='DecsionTree Train')
print_regression_metrics(y_test, tree.predict(X_test), title='DecsionTree Test')
print('-' * 40)