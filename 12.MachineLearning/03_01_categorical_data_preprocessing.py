'''
1. 범주형 데이터의 처리
    1) 범주형 변수
        (1) 정의 : 몇 개의 범주 중 하나에 속하는 값들로 구성된 변수. 분류에 대한 속성을 가지는 변수.
            (예 : 성별(남,여), 혈액형(A,B,AB,O), 성적(A,B,C,D,E,F))

        (2) 종류
            - 비서열 변수 : 범주에 속한 값들이 서열(순위)가 없는 변수
                (예 : 성별, 혈액형 등) 
            - 서열 변수 : 범주세 속한 값들이 서열(순위)가 있는 변수
                (예 : 성적, 직급 등)
        (3) sklearn의 특수성
            - 문자열 값을 처리하지 X >> 입력 또는 출력 데이터가 문자열일 경우 숫자(일반적으로 정수)형으로 변환
            - 범주형 변수 >> 정수형으로 변환
            - 일반 문자열 변수 >> 제거

    2) 범주형 feature의 처리 
        (1) 레이블 인코딩(Label Encoding)
            - 설명 : 문자열(범주형) 값을 오름차순 정렬 후 0부터 1씩 증가하는 값으로 반환
                (ex. [tv, 냉장고, tv, 컴퓨터, 에어컨, tv] >> [0, 1, 0, 2, 3, 0])
            - 적용
                - 적용 가능 : 의사 결정 나무, 랜덤 포레스트(숫자의 차이가 모델에 영향을 주지 x)
                - 적용 불가 : 로지스틱 회귀, SVM, 신경망(숫자의 차이가 모델에 영향을 줌)
            - sklearn.preprocessing.LabelEncoder의 사용
                - fit() : 변환 방법 학습
                - transform() : 문자열을 숫자로 변환
                - fit_transform() : 학습과 변환 동시 수행
                - inverse_transform() : 숫자를 문자열로 역변환
                - classes_ : 인코딩한 클래스 조회

        (2) 원핫 인코딩(OneHot Encoding)
            - N개의 클래스를 N차원의 One-Hot 벡터로 표현되도록 변환
                - 고유값들을 피처로 만들고 정답에 해당하는 열은 1, 나머지는 0으로 표시
            - 숫자의 차이가 모델에 영향을 미치는 선형 계열 모델(로지스틱 회귀, SVM, 신경망 등)에서 범주형 데이터 변환시 사용
            - Scikit learn
                - sklearn.preprocessing.OneHotEncoder
                    - fit() : 변환 학습
                    - transform() : 변환(문자열 >> 숫자)
                    - fit_transform() : 학습과 변환 동시
                    - get_feature_names() : onehot encoding으로 변환된 컬럼의 이름 반환
                - Dataframe을 넣을 경우 모든 변수를 변환
                    - 범주형 컬럼만 처리하도록 미리 처리
            - Pandas
                - pandas.get_dummies(dataframe, [columns = [변환할 컬럼명]]) 함수 이용
                - dataframe에서 범주형 변수(object, category) 변수만 변환

        
'''

## Label Encoding
import numpy as np
import scipy as sp
from sklearn.preprocessing import LabelEncoder

items = ['tv', '냉장고', '컴퓨터', '컴퓨터', '냉장고', '에어컨', 'tv', '에어컨']

#label encoder 객체 생성
le = LabelEncoder()

#변환 학습
le.fit(items)
item_label = le.transform(items)

item_label
    # array([0, 1, 3, 3, 1, 2, 0, 2], dtype=int64)

item_label2 = le.transform(items)
    # array([0, 1, 3, 3, 1, 2, 0, 2], dtype=int64)

print(item_label == item_label2, np.all(item_label == item_label2))


items1 = ['TV', '냉장고', '컴퓨터', '컴퓨터', '냉장고', '에어컨', 'TV', '에어컨', '공기청정기']
items2 = ['TV', '냉장고', '냉장고', '컴퓨터']

item1_labels = le.fit_transform(items1)
    # [0 2 4 4 2 3 0 3 1]
item2_labels = le.transform(items2)
    # item2는 item1보다 자료의 종류가 적다 >> transform만 해도 됨!
    # fit하면 오류 발생
    # [0 1 1 2]

le.classes_
    # [tv, 냉장고, 컴퓨터]

print(le.inverse_transform(item2_labels))
print(le.inverse_transform(item1_labels))


#items3 = ['TV', 'TV', '핸드폰', '건조기', '선풍기'] 
    # 건조기와 선풍기는 item1을 학습 시킬 때 없었으므로 오류 발생! 
    # 만약 건조기와 선풍기도 학습 시키고 싶다면 fit부터 다시!

#-----------------------------------------------------------------------------------------------------------------

'''
2. 데이터 전처리 - 결측치 처리
    1) 결측치와 머신러닝
        (1) 결측치 : 수집하지 못한 값 or 모르는 값
        (2) 머신러닝 알고리즘은 feature에 결측치가 있을 시 처리 불가

    2) 결측치 처리 
        (1) 제거 
            - 특정 열 또는 행에 결측치가 많을 경우 해당 열 또는 행 단위로 제거
            - 데이터 양이 충분한 경우
        (2) 대체 
            - 가장 가능성이 높은 값으로 대체 
                - 수치형 : 평균, 중앙값 등
                - 범주형 : 최빈값
                - 결측치를 추론하는 머신러닝 알고리즘 사용
            - 결측치를 표현하는 값으로 대체
                - ex : 나이(-1, 999 등)
            - 데이터 양이 충분치 않은 경우
'''

'''
    adult_data에 label encoding 적용해보기
        - 1994년 인구조사 데이터 베이스에서 추출한 미국 성인의 소득 데이터셋이다.
        - target 은 income 이며 수입이 $50,000 이하인지 초과인지로 분류되어 있다.
        - https://archive.ics.uci.edu/ml/datasets/adult

'''
import pandas as pd

cols = ['age', 'workclass','fnlwgt','education', 'education-num', 'marital-status', 
        'occupation','relationship', 'race', 'gender','capital-gain','capital-loss', 
        'hours-per-week','native-country', 'income']

data = pd.read_csv("../data/adult.data", header=None, names=cols, na_values='?', skipinitialspace=True)
    # skipinitialspace=True : 필드 앞의 공백 제거
    # na_values='?' : 결측값을 '?'로 대체

print()
print('-'*14,'adult data 읽어오기', '-'*14)
print(data.head())
print()
print('data의 shape', data.shape)
    # 32561, 15
print('income 열의 값', data['income'].value_counts())
print(data.info())
print(data.isnull().sum())
    # workclass의 결측치 : 1836 / occupation의 결측치 : 1843 / native-country의 결측치 : 583

data.dropna(inplace=True)
print(data.isnull().sum())
    # 결측치를 제거하여 모두 0가 나옴

print(data.shape)
    # 30162, 15

encoding_columns = ['workclass','education','marital-status', 'occupation','relationship','race','gender','native-country', 'income']
not_encoding_columns = ['age','fnlwgt', 'education-num','capital-gain','capital-loss','hours-per-week']
    # 이미 수치로 되어 있기 때문에 새로 encoding할 필요 x

print(data[encoding_columns].head(3))
#            workclass  education      marital-status         occupation   relationship   race gender native-country   income
#    0         State-gov  Bachelors       Never-married       Adm-clerical  Not-in-family  White   Male  United-States  <=50K
#    1  Self-emp-not-inc  Bachelors  Married-civ-spouse    Exec-managerial        Husband  White   Male  United-States  <=50K
#    2           Private    HS-grad            Divorced  Handlers-cleaners  Not-in-family  White   Male  United-States  <=50K


print('*' * 20)
adult_data = data.copy()

enc_dict = {}

# 방법1. 반복문을 이용한 encoding
for col in encoding_columns:
    le = LabelEncoder()

    #학습 시킨 후 데이터프레임 내부의 열을 학습 데이터로 변경
    column_value = le.fit_transform(adult_data[col])
    adult_data[col] = column_value

    #enc_dict에 원래 무엇이 있었는지 저장
    enc_dict[col] = le.classes_

print(adult_data.head())
print(enc_dict)

#######################################
# 방법2 . encoding 함수를 만들어 일괄 처리
enc_dict2 = {}

def label_encode(column):
    le = LabelEncoder()
    encoded_values = le.fit_transform(column)
    enc_dict2[column.name] = le
    return encoded_values

print(data[encoding_columns].apply(label_encode))

print(enc_dict2['workclass'], enc_dict2['workclass'].inverse_transform([1,2,2,0]), '\n')

enc_df = data[encoding_columns].apply(label_encode)
print(enc_df.head(3), '\n')

adult_data2 = enc_df.join(data[not_encoding_columns])
print(adult_data2.head(3))


## OneHot Encoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
print()
print("============================OneHot encoding============================")
print()
items = np.array(['TV', '냉장고', '냉장고', '컴퓨터', '에어컨', '컴퓨터', '에어컨'])
    # items의 경우 1 dimension array >> onehot encoding 시 오류 발생 >> 2차원으로 변환
src_items = items[..., np.newaxis]

ohe = OneHotEncoder()
ohe.fit(src_items)
rv = ohe.transform(src_items)

print(type(rv))
print(rv)
rv = rv.toarray()   #무엇이 어떤 제품인지 알기 어렵
'''
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]]
'''

print(ohe.get_feature_names())
# ['x0_TV' 'x0_냉장고' 'x0_에어컨' 'x0_컴퓨터']

ohe_tot_df = pd.DataFrame(rv, columns=ohe.get_feature_names())
print(ohe_tot_df)

# df로 만드는 또 다른 방법
ohe = OneHotEncoder(sparse=False) # 처리 결과를 ndarray로 반환
ohe.fit(src_items)
rv = ohe.transform(src_items)

df = pd.DataFrame(rv, columns=ohe.get_feature_names())

###########################################################################################
# 딕셔너리를 이용한 one hot encoding
# items = ['TV', '냉장고', '냉장고', '컴퓨터', '에어컨', '컴퓨터', '에어컨']
dic = {'item' : items,
        'Count' : [10, 10, 20, 15, 13, 3, 12], 
        'Level' : [1, 1, 1, 2, 3, 3, 1]
        }
df = pd.DataFrame(dic)
print(df)

ohe2 = OneHotEncoder(sparse=False)
ohe2.fit(df)
rv2 = ohe2.transform(df)
print
print('rv2\n', rv2)

############################################################################################
ohe3 = OneHotEncoder(sparse=False)
ohe3.fit(df[['item', 'Level']])
rv3 = ohe3.transform(df[['item', 'Level']])
print('rv3\n', rv3)

#rv3와 df['count']의 values들 열로 합성
result = np.concatenate([rv3, df['Count'].values[..., np.newaxis]], axis=1)
print('rv3 + df["count"]', result)

############################################################################################
ohe4 = OneHotEncoder(sparse=False)
rv4 = ohe4.fit_transform(src_items)
print(rv4)

############################################################################################
ohe5 = OneHotEncoder(sparse=False)
rv5 = ohe5.fit_transform(df[['item', 'Level']])
print(ohe5.get_feature_names())
print(rv5)



###########################################################################################
################# Pandas get_dummies 이용하여 OneHotEncoding 하기 ##########################
dummy_df = pd.get_dummies(df)
print(dummy_df)

dic = {
    'Item' : items,
    'Count' : [10, 10, 20, 15, 13, 3, 12],
    'Level1' : [1,1,1,2,3,3,1],
    'Level2' : ['A', 'B', 'A', 'C', 'A', 'B', 'B']
}
df = pd.DataFrame(dic)
        #   Item  Count  Level1 Level2
        #0   TV     10       1      A
        #1  냉장고     10       1      B
        #2  냉장고     20       1      A
        #3  컴퓨터     15       2      C
        #4  에어컨     13       3      A
        #5  컴퓨터      3       3      B
        #6  에어컨     12       1      B


dummy_df = pd.get_dummies(df)
        #   Count  Level1  Item_TV  Item_냉장고  Item_에어컨  Item_컴퓨터  Level2_A  Level2_B  Level2_C
        #0     10       1        1         0         0         0         1         0         0
        #1     10       1        0         1         0         0         0         1         0
        #2     20       1        0         1         0         0         1         0         0
        #3     15       2        0         0         0         1         0         0         1
        #4     13       3        0         0         1         0         1         0         0
        #5      3       3        0         0         0         1         0         1         0
        #6     12       1        0         0         1         0         0         1         0

dummy_df2 = pd.get_dummies(df, columns=['Item','Level1'])
        #   Count Level2  Item_TV  Item_냉장고  Item_에어컨  Item_컴퓨터  Level1_1  Level1_2  Level1_3
        #0     10      A        1         0         0         0         1         0         0
        #1     10      B        0         1         0         0         1         0         0
        #2     20      A        0         1         0         0         1         0         0
        #3     15      C        0         0         0         1         0         1         0
        #4     13      A        0         0         1         0         0         0         1
        #5      3      B        0         0         0         1         0         0         1
        #6     12      B        0         0         1         0         1         0         0

dummy_df3 = pd.get_dummies(df, columns=['Item', 'Level1', 'Level2'])
        #   Count  Item_TV  Item_냉장고  Item_에어컨  Item_컴퓨터  Level1_1  Level1_2  Level1_3  Level2_A  Level2_B  Level2_C
        #0     10        1         0         0         0         1         0         0         1         0         0
        #1     10        0         1         0         0         1         0         0         0         1         0
        #2     20        0         1         0         0         1         0         0         1         0         0
        #3     15        0         0         0         1         0         1         0         0         0         1
        #4     13        0         0         1         0         0         0         1         1         0         0
        #5      3        0         0         0         1         0         0         1         0         1         0
        #6     12        0         0         1         0         1         0         0         0         1         0

#######################################################################################
test_items = [['tv'], ['선풍기'], ['에어컨'], ['건조기']]
ohe6 = OneHotEncoder(sparse=False)
ohe6.fit(test_items)
np.unique(src_items)    #['TV' '냉장고' '에어컨' '컴퓨터']

#ohe6.transform(src_items) #에러 발생 >> 없는 클래스 값을 가지고 있음!
