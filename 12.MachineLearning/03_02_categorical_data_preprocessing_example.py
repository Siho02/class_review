import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# adult.data를 읽어오기 위한 사전 준비(열 이름을 미리 설정)
cols = ['age', 'workclass','fnlwgt','education', 'education-num', 'marital-status', 
        'occupation','relationship', 'race', 'gender','capital-gain','capital-loss', 
        'hours-per-week','native-country', 'income']

# adult.data 읽어오기 + 결측치 제거
data = pd.read_csv("../data/adult.data", header=None, names=cols, na_values='?', skipinitialspace=True)
data.dropna(inplace=True)

# 인코딩할 열과 인코딩이 불필요한 열 설정
encoding_columns = ['workclass','education','marital-status', 'occupation','relationship','race','gender','native-country', 'income']
not_encoding_columns = ['age','fnlwgt', 'education-num','capital-gain','capital-loss','hours-per-week']

# 원본 데이터 훼손 방지를 위한 copy
adult_data = data.copy()

# 반복문을 이용한 encoding
enc_dict = {}
for col in encoding_columns:
    le = LabelEncoder()

    #학습 시킨 후 데이터프레임 내부의 열을 학습 데이터로 변경
    column_value = le.fit_transform(adult_data[col])
    adult_data[col] = column_value

    #enc_dict에 원래 무엇이 있었는지 저장
    enc_dict[col] = le.classes_


# 데이터 분할
X = adult_data.drop(columns = 'income')
y = adult_data['income']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# HoldOut 방식으로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train , stratify=y_train, random_state=0)

# 모델 생성
tree = DecisionTreeClassifier(max_depth=7, random_state=0)

# 모델 학습
tree.fit(X_train, y_train)

# 모델 검증
pred_train = tree.predict(X_train)
pred_val = tree.predict(X_val)

acc_train = accuracy_score(y_train, pred_train)
acc_val = accuracy_score(y_test, pred_val)

print("train 정확도 = {}, validataion 정확도 = {}".format(acc_train, acc_val))

'''
max_depth 지정 x : train 정확도 : 0.999944739168877, val 정확도 : 0.7982761478534726
max_depth 1 : train 정확도 : 0.751105216622458, val 정확도 : 0.7510359688380573
max_depth 2 : train 정확도 : 0.8211759504862953, val 정확도 : 0.8238024200232057
max_depth 3 : train 정확도 : 0.8358200707338639, val 정확도 : 0.8390518813194099
max_depth 5 : train 정확도 : 0.8499115826702034, val 정확도 : 0.854135587601525
max_depth 7 : train 정확도 : 0.8575928381962865, val 정확도 : 0.8556273827283275
'''

# 최종 평가
pred_test = tree.predict(X_test)
acc_test = accuracy_score(y_test, pred_test)
print("test 정확도 : ", acc_test)






## 2. Label Encoder와 OneHot Encoder 사용하기
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#adult.data 읽어 들인 후 결측치 제거
df = pd.read_csv('../data/adult.data', header=None, names=cols, skipinitialspace=True, na_values='?')
df.dropna(inplace=True)

#자료로 쓸 열의 제목만 가져다 새로운 dataframe 생성
data_cols = ['age', 'workclass', 'education', 'occupation', 'gender', 'hours-per-week', 'income']
adult_df = df[data_cols]

#1. get_dummies를 이용한 방법
#label encoder 객체 생성
#훈련 및 변환하여 y에 저장 / X에는 income을 제외한 나머지 열들을 저장
le = LabelEncoder()
y = le.fit_transform(adult_df['income'])
X = pd.get_dummies(adult_df[adult_df.columns[:-1]])

#잘 되었는지 확인
#print(adult_df.head())

#OneHotEncoder 생성 및 훈련, 변환
#workcalss, education, occupation, gender 는 범주형 데이터 이므로 전처리 필요
ohe = OneHotEncoder(sparse=False)
return_val = ohe.fit_transform(adult_data[['workclass', 'education', 'occupation', 'gender']]) #(30162, 39)
#print(return_val)

#age, hours-per-week은 숫자 데이터이므로 전처리 불필요 
v = adult_df[['age', 'hours-per-week']].values  #(30162, 2)
#print(v)

#데이터 프레임 합치기
X = np.concatenate([return_val, v], axis=1) #(30162, 41)


#2. Dataframe으로 변환해서 합치기
v2 = adult_data[['age', 'hours-per-week']]
r_df = pd.DataFrame(return_val, columns=ohe.get_feature_names())
#print(r_df)
X2 = v2.reset_index(drop=True).join(r_df)

#########################모델 학습 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#데이터 분할 train, test, validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)
print(y_train.shape, y_val.shape, y_test.shape)

#tree모델 이용
tree = DecisionTreeClassifier(max_depth=7, random_state=0)
tree.fit(X_train, y_train)

pred_train = tree.predict(X_train)
pred_val = tree.predict(X_val)
acc_train = accuracy_score(y_train, pred_train)
acc_val = accuracy_score(y_val, pred_val)

#Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

pred_train_lr = lr.predict(X_train)
pred_val_lr = lr.predict(X_val)
acc_train_lr = accuracy_score(y_train, pred_train_lr)
acc_val_lr = accuracy_score(y_val, pred_val_lr)

print("Decision Tree 평가")
print("\ttrain 정확도 : {}, Validataion 정확도 : {}".format(acc_train, acc_val))
print('Logistic Regression 평가')
print("\ttrain 정확도 : {}, Validataion 정확도 : {}".format(acc_train_lr, acc_val_lr))