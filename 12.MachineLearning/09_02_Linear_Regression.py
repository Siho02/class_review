'''
회귀 모델(Regression Model)

1. 선형 회귀(Linear Regression)
    - 종속 변수 y와 한개 이상의 독립 변수(또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀 분석 기법
    
    1) 가장 기본적인 선형 회귀 모델
    2) feture 전처리
        (1) 범주형 데이터 : OneHotEncoder
        (2) 연속형 데이터 : Feature scaling(standard scaler를 사용하는 것이 성능이 더 잘 나오는 경향이 있다)

2. 선형 회귀 모델 : y_i_hat = w1*xi1 + w2*xi2 + .... + wp*xip + b
                    (y_i_hat : 예측값 / x : 특성 / w : 가중치 / b : 절편 / p : p번째 특성 / i : i번째 특성)

3. 다항 회귀(Polynomial Regression)
    - 단순한 직선형보다는 복잡한 비선형 형태의 데이터를 추론하기 위한 모델
    - feature들을 거듭제곱한 feature들을 추가하여 모델링
    - PolynomialFeatures 변환기를 이용

4. 손실(loss) 함수
    1) 모델이 출력한 예측값과 실제값 사이의 차이를 계산하는 함수
    2) 평가 지표로 사용되기도 하고 모델을 최적화하는데 사용
    3) 오차함수(error), 비용함수(cost), 목적함수(objective)라고도 함

5. 최적화(optimize)
    1) 정의 : 손실함수의 값이 최소화 되도록 모델을 학습하는 과정
    2) 두가지 방법
        (1) 정규 방정식
        (2) 경사 하강법

'''

#Boston data를 이용하기
# CRIM : 지역별 범죄 발생률 / ZN : 25,000 평방피트를 초과하는 거주지역의 비율 / INDUS: 비상업지역 토지의 비율
# CHAS : 찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0) / NOX : 일산화질소 농도 / RM : 주택 1가구당 평균 방의 개수
# AGE : 1940년 이전에 건축된 소유주택의 비율 / DIS : 5개의 보스턴 고용센터까지의 접근성 지수 / RAD : 고속도로까지의 접근성 지수
# TAX : 10,000 달러 당 재산세율 /PTRATIO : 지역별 교사 한명당 학생 비율 / B : 지역의 흑인 거주 비율 / LSTAT: 하위계층의 비율(%)
# MEDV : Target. 본인 소유의 주택가격(중앙값) (단위: $1,000)

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
data = load_boston()
data.keys()
    # dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

X, y = data['data'], data['target']
df = pd.DataFrame(X, columns=data['feature_names'])
print(df)
df['MEDV'] = y

#print(df['CHAS'].value_counts()) #범주형
    #0.0    471
    #1.0     35
    #Name: CHAS, dtype: int64

#Linear Regression
    # 범주형(CHAS) : onehotencoding
    # 연속형(나머지) : standard scaling

#CHAS데이터 받아 더미데이터프레임으로 변환
chas_df = pd.get_dummies(df['CHAS'])
chas_df.columns = ['CHAS_0', 'CHAS_1']

df2 = df.join(chas_df) #CHAS 원핫인코딩 컬럼을 DF에 추가
df2.drop(columns='CHAS', inplace=True) # 원래 있던 CHAS컬럼 제거
    #      CRIM    ZN  INDUS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV  CHAS_0  CHAS_1
    #0  0.00632  18.0   2.31  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0       1       0
    #1  0.02731   0.0   7.07  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6       1       0
    #2  0.02729   0.0   7.07  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7       1       0
    #3  0.03237   0.0   2.18  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4       1       0
    #4  0.06905   0.0   2.18  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2       1       0

y = df2['MEDV'] #target data
X = df2.drop(columns='MEDV')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_trian_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    # X_train.columns = Index(['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 
    #                           'CHAS_0', 'CHAS_1'], dtype='object')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from util import print_metrics, print_regression_metrics

lr = LinearRegression()
lr.fit(X_trian_scaled, y_train)

print('linear regression의 intercept, coefficient')
print(lr.intercept_, lr.coef_)

# lr.coef_를 연관지어 시리즈로 구성
pd.Series(lr.coef_, index = X_train.columns)

# 모델을 통한 예측
pred_train = lr.predict(X_trian_scaled)
pred_test = lr.predict(X_test_scaled)
print_regression_metrics(y_train, pred_train, title='Train')
print_regression_metrics(y_test, pred_test, title='Test')

# 실제값(MEDV)와 예측값을 pyplot을 이용하여 비교해보기
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='MEDV', marker='o')
plt.plot(range(len(y_test)), pred_test, label='Pred', marker='x')
plt.legend()
plt.show()


###################### 다항 회귀 ##########################
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = X ** 2 + X + 2 + np.random.normal(0,1, size=(m,1))
y = y.flatten()

df = pd.DataFrame({"X" : X.flatten(), "Y" : y})
'''
          X         Y
0  0.292881  1.213510
1  1.291136  5.858995
2  0.616580  3.462414
3  0.269299  0.805577
4 -0.458071  3.240010
'''

#그래프로 표현
plt.figure(figsize=(7,6))
plt.scatter(X, y, alpha=0.5)
plt.show()

lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_, lr.intercept_)
    # (array([0.78189543]), 5.175619278567209)

pred = lr.predict(X)
print_regression_metrics(y, pred)

X_new = np.linspace(-3, 3, 100).reshape(-1,1)
#print(X_new)
y_new = lr.predict(X_new)

plt.figure(figsize=(7,6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X_new, y_new, color='red')
plt.show()
    # X, y의 관계를 정확하게 설명할 수 X

## X의 feature수를 늘려 다항식이 되도록 처리(2차함수) >> polynomial features
from sklearn.preprocessing import PolynomialFeatures

pnf = PolynomialFeatures(degree=2, include_bias=False)
    #degree : 최고차항 지정 / include_bias=False : 상수항을 추가(모든값 1인 feature 추가)
X_poly = pnf.fit_transform(X)
    #X.shape = (100, 1) >> X_poly.shape = (100, 2)
print(pnf.get_feature_names())

lr2 = LinearRegression()
lr2.fit(X_poly, y)
print('lr2의 coef, intercept', lr2.coef_, lr2.intercept_)

##그래프 그리기
X_new_poly = pnf.transform(X_new)
y_new = lr2.predict(X_new_poly)

plt.figure(figsize=(6,6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X_new, y_new, color='red')
plt.show()

print('\n============lr과 lr2의 평가지표============')
print_regression_metrics(y, lr.predict(X))
print_regression_metrics(y, lr2.predict(X_poly))
print()

##input data가 다차원(feature 수 = 3)
data = np.arange(12).reshape(4,3) 
    #array([[ 0,  1,  2],
    #       [ 3,  4,  5],
    #       [ 6,  7,  8],
    #       [ 9, 10, 11]])

pnf2 = PolynomialFeatures(degree=2)
data_poly = pnf2.fit_transform(data)
    #shape = (4, 10)

pnf2.get_feature_names()
    #['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']

pd.DataFrame(data_poly, columns=pnf2.get_feature_names())
'''
    	1	x0	x1	x2	x0^2	x0 x1	x0 x2	x1^2	x1 x2	x2^2
    0	1.0	0.0	1.0	2.0	0.0	    0.0	    0.0	    1.0	    2.0	    4.0
    1	1.0	3.0	4.0	5.0	9.0	    12.0	15.0	16.0	20.0	25.0
    2	1.0	6.0	7.0	8.0	36.0	42.0	48.0	49.0	56.0	64.0
    3	1.0	9.0	10.011.081.0	90.0	99.0	100.0	110.0	121.0
'''

# degree를 5로 늘림
pnf3 = PolynomialFeatures(degree=5)
data_poly2 = pnf3.fit_transform(data)
data_poly2.shape    #(4, 56)
pnf3.get_feature_names()
# ['1','x0','x1','x2','x0^2','x0 x1','x0 x2','x1^2','x1 x2','x2^2','x0^3','x0^2 x1','x0^2 x2','x0 x1^2','x0 x1 x2','x0 x2^2',
#  'x1^3','x1^2 x2','x1 x2^2','x2^3','x0^4','x0^3 x1','x0^3 x2','x0^2 x1^2','x0^2 x1 x2','x0^2 x2^2','x0 x1^3','x0 x1^2 x2',
#  'x0 x1 x2^2','x0 x2^3','x1^4','x1^3 x2','x1^2 x2^2','x1 x2^3','x2^4','x0^5','x0^4 x1','x0^4 x2','x0^3 x1^2','x0^3 x1 x2',
#  'x0^3 x2^2','x0^2 x1^3','x0^2 x1^2 x2','x0^2 x1 x2^2','x0^2 x2^3','x0 x1^4','x0 x1^3 x2','x0 x1^2 x2^2','x0 x1 x2^3',
#  'x0 x2^4','x1^5','x1^4 x2','x1^3 x2^2','x1^2 x2^3','x1 x2^4','x2^5']

pnf = PolynomialFeatures(degree=100, include_bias=False)
X_train_poly_100 = pnf.fit_transform(X)
X_train_poly_100.shape  #(100, 100)
X.shape                 #(100, 1)

lr = LinearRegression()
lr.fit(X_train_poly_100, y)

X_new = np.linspace(-3,3,100).reshape(-1, 1)
X_new_poly_100 = pnf.transform(X_new)
pred_new_100 = lr.predict(X_new_poly_100)

plt.figure(figsize=(7,6))
plt.scatter(X, y)
plt.plot(X_new, pred_new_100, color='r', alpha=0.5)
plt.title('degree = 100')
plt.show()

#############################################################################################
####################### Boston Data와 Polynomial Regression  ################################
#############################################################################################
pnf = PolynomialFeatures(degree=2, include_bias=False)
data = load_boston()

X, y = data['data'], data['target']
df = pd.DataFrame(X, columns=data['feature_names'])
df['MEDV'] = y

chas_df = pd.get_dummies(df['CHAS'])
chas_df.columns = ['CHAS_0', 'CHAS_1']

df2 = df.join(chas_df) #CHAS 원핫인코딩 컬럼을 DF에 추가
df2.drop(columns='CHAS', inplace=True) # 원래 있던 CHAS컬럼 제거

y = df2['MEDV'] #target data
X = df2.drop(columns='MEDV')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_trian_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pnf = PolynomialFeatures(degree=2, include_bias=False)
X_train_scaled_poly = pnf.fit_transform(X_trian_scaled)     #(379, 13)
X_test_scaled_poly = pnf.transform(X_test_scaled)           #(379, 104)

lr3 = LinearRegression()
lr3.fit(X_train_scaled_poly, y_train)
pred_train3 = lr3.predict(X_train_scaled_poly)
pred_test3 = lr3.predict(X_test_scaled_poly)

print_regression_metrics(y_train, pred_train3, title='Train poly')
    #MSE : 4.09231630944325, RMSE : 2.0229474312110165, R2 = 0.952029059282025
print_regression_metrics(y_test, pred_test3, title='Test poly')
    #MSE : 31.957178742416264, RMSE : 5.653068082237845, R2 = 0.6088425475989248

df2.head()
    #      CRIM    ZN  INDUS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV  CHAS_0  CHAS_1
    #0  0.00632  18.0   2.31  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0       1       0
    #1  0.02731   0.0   7.07  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6       1       0
    #2  0.02729   0.0   7.07  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7       1       0
    #3  0.03237   0.0   2.18  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4       1       0
    #4  0.06905   0.0   2.18  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2       1       0

'''
Ridge Regression
    - 손실함수(loss fucntion)에 규제항으로 α * ∑ (w_i)^2 (L2 Norm)을 더해준다
    - α = 0에 가까울수록 규제가 약해진다(0일 경우 선형 회귀와 동일)
    - α가 커질 수록 모든 가중치가 작아져 입력 데이터의 feature들 중 중요하지 않은 feature의 output에 대한 영향력이 작아짐
    - 손실함수(w) = MSE(w) + α * (1/2) * ∑(w_i)^2
'''
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# alpha = 1
ridge1 = Ridge(random_state=0) 
ridge1.fit(X_train_scaled, y_train)
pred_train1 = ridge1.predict(X_train_scaled)
pred_test1 = ridge1.predict(X_test_scaled)
print('alpha=1')
print_regression_metrics(y_train, pred_train1, title='Train')
print_regression_metrics(y_test, pred_test1, title="Test")
ridge1.coef_    #array([-0.96187481,  1.02775462, -0.06861144,  0.59814087, -1.77318401, 2.6205672 , -0.20466821, -2.96504904,  2.00091047, -1.85840697, -2.14955893,  0.75175979, -3.57350065])

#alpah=0.01
ridge1 = Ridge(alpha=0.01, random_state=0) 
ridge1.fit(X_train_scaled, y_train)
pred_train1 = ridge1.predict(X_train_scaled)
pred_test1 = ridge1.predict(X_test_scaled)
print('alpha=0.01')
print_regression_metrics(y_train, pred_train1, title='Train')
print_regression_metrics(y_test, pred_test1, title="Test")
ridge1.coef_    #array([-0.97090686,  1.04648351, -0.04074187,  0.59413006, -1.80840456, 2.61003017, -0.19830017, -3.00178921,  2.07939188, -1.93211252, -2.15735709,  0.75198861, -3.59010071])


#alpah=1000
ridge1 = Ridge(alpha=1000, random_state=0) 
ridge1.fit(X_train_scaled, y_train)
pred_train1 = ridge1.predict(X_train_scaled)
pred_test1 = ridge1.predict(X_test_scaled)
print('alpha=1000')
print_regression_metrics(y_train, pred_train1, title='Train')
print_regression_metrics(y_test, pred_test1, title="Test")
ridge1.coef_    #array([-0.44267768,  0.38220219, -0.51288178,  0.3335525 , -0.37129939, 1.25386598, -0.32729508, -0.06287806, -0.28302417, -0.47738562, -0.87977916,  0.4225767 , -1.16283877])



### GridSearch를 이용한 최적화된 alpha 찾기
from sklearn.model_selection import GridSearchCV
param = {"alpha":[0.01, 0.1, 1, 5, 10, 20, 30, 40, 100]}
ridge = Ridge(random_state=0)
gs = GridSearchCV(ridge, param, cv=4, scoring=['r2', 'neg_mean_squared_error'], refit='r2')

gs.fit(X_train_scaled, y_train)
result_df = pd.DataFrame(gs.cv_results_)
print(result_df.sort_values('rank_test_r2').head())
    #   mean_fit_time  std_fit_time  mean_score_time  ...  mean_test_neg_mean_squared_error std_test_neg_mean_squared_error rank_test_neg_mean_squared_error
    #5       0.001501      0.000501         0.001251  ...                        -23.363210                        5.542457                                2
    #4       0.001501      0.000500         0.001003  ...                        -23.352349                        5.376125                                1
    #6       0.000908      0.000584         0.000751  ...                        -23.435915                        5.597524                                4
    #3       0.001002      0.000708         0.000750  ...                        -23.393786                        5.196172                                3
    #7       0.000000      0.000000         0.003907  ...                        -23.546877                        5.607163                                8


'''
Lasso(Least Absolut Shrinkage and Selection Operator) Regression
    - 손실함수에 규제항으로 𝛼∑|𝑤𝑖|(L1 Norm)더한다.
    - Lasso 회귀의 상대적으로 덜 중요한 특성의 가중치를 0으로 만들어 자동으로 Feature Selection이 된다.
    - 손실함수(𝑤)=MSE(𝑤)+𝛼∑|𝑤𝑖|
'''

from sklearn.linear_model import Lasso

#alpha = 1
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
pred_train = lasso.predict(X_train_scaled)
pred_test = lasso.predict(X_test_scaled)
print_regression_metrics(y_train, pred_train, "alpha=1 Train")
print_regression_metrics(y_test, pred_test, "alpha=1 Test")

#alpha=10
lasso = Lasso(alpha=10, random_state=0) 
lasso.fit(X_train_scaled, y_train)
pred_train = lasso.predict(X_train_scaled)
pred_test = lasso.predict(X_test_scaled)
print_regression_metrics(y_train, pred_train, "alpha=10 Train")
print_regression_metrics(y_test, pred_test, "alpha=10 Test")




###Boston Dataset을 이용해서 Ridge, Lasso(Polynomial Features로 전처리)###
from sklearn.linear_model import Ridge, Lasso, LinearRegression

alpha_lst = [0.01, 0.1, 1, 10, 100]
lr = LinearRegression()
lr.fit(X_train_scaled_poly, y_train)
pred_train_lr = lr.predict(X_train_scaled_poly)
pred_test_lr = lr.predict(X_test_scaled_poly)
print_regression_metrics(y_train, pred_train_lr, title="LinearRegression Train")
print_regression_metrics(y_test, pred_test_lr, title="LinearRegression Test")

#Ridge의 alpha값 변화에 따른 R2값을 저장할 리스트 생성
ridge_train_metrics_list = []
ridge_test_metrics_list = []

for alpha in alpha_lst:
    ridge = Ridge(alpha = alpha, random_state=0)
    ridge.fit(X_train_scaled_poly, y_train)
    pred_train = ridge.predict(X_train_scaled_poly)
    pred_test = ridge.predict(X_test_scaled_poly)
    ridge_train_metrics_list.append(r2_score(y_train, pred_train))
    ridge_test_metrics_list.append(r2_score(y_test, pred_test))

ridge_df = pd.DataFrame({
    'alpha' : alpha_lst,
    'train' : ridge_test_metrics_list,
    'test' : ridge_test_metrics_list
})

print(ridge_df)

#Lasso의 alpha값 변화에 따라서 R2를 저장할 리스트
lasso_train_metrics_list = []
lasso_test_metrics_list = []

for alpha in alpha_lst:
    lasso = Lasso(alpha=alpha, random_state=0)
    lasso.fit(X_train_scaled_poly, y_train)
    pred_train = lasso.predict(X_train_scaled_poly)
    pred_test = lasso.predict(X_test_scaled_poly)
    lasso_train_metrics_list.append(r2_score(y_train, pred_train))
    lasso_test_metrics_list.append(r2_score(y_test, pred_test))

lasso_df = pd.DataFrame({
    "alpha":alpha_lst,
    "train":lasso_train_metrics_list,
    "test":lasso_test_metrics_list
})

print(lasso_df)



'''
엘라스틱넷
    - 릿지와 라쏘를 절충한 모델.
    - 규제항에 릿지, 회귀 규제항을 더해서 추가한다.
    - 혼합뷰율 𝑟을 사용해 혼합정도를 조절
    - 𝑟=0이면 릿지와 같고 𝑟=1이면 라쏘와 같다.
    - 손실함수(𝑤)=MSE(𝑤)+𝑟𝛼∑|𝑤𝑖|+ {(1−𝑟)/2} * 𝛼∑(𝑤_𝑖)^2
'''
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5) #L1규제(Lasso) 비율: 0.4, L2규제(Ridge) 비율: 0.6
elastic.fit(X_train_scaled_poly, y_train)

pred_train = elastic.predict(X_train_scaled_poly)
pred_test = elastic.predict(X_test_scaled_poly)

print_regression_metrics(y_train, pred_train, title='Train')
print_regression_metrics(y_test, pred_test, title='Test')


'''
Summary
    - 일반적으로 선형회귀의 경우 어느정도 규제가 있는 경우가 성능이 좋다.
    - 기본적으로 릿지를 사용한다.
    - Target에 영향을 주는 Feature가 몇 개뿐일 경우 특성의 가중치를 0으로 만들어 주는 라쏘 사용한다.
    - 특성 수가 학습 샘플 수 보다 많거나 feature간에 연관성이 높을 때는 엘라스틱넷을 사용한다.
'''