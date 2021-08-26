'''
1. 분류와 회귀의 평가 방법
    1) 이진 분류 평가 지표
        (0)용어
            - 이진분류에서의 양성과 음성
                -양성 : 예측하려는 대상
                -음성 : 예측하려는 대상이 아닌 것
                -ex : 암환자 분류(양 : 암환자 / 음 : 비환자), 스팸메일 분류(양 : 스팸메일 / 음 : 정상메일) ...
            - 함수 : classification_report(y실제값, y예측값) : 클래스별로 recall, precision, f1 score, accuracy 통합하여 반환

        (1)정확도(accuracy) 
            - 정확도 : (맞계 예측한 건수) / (전체 예측 건수)
            - accuracy_score(정답, 예측한값)
            - 정확도 평가 지표의 문제점
                - 불균형 데이터의 경우 정확한 평가지표가 될 수 없다
                - 양성 : 음성 = 1 : 9인 경우 모두 음성이라고 해도 정확도는 90%로 높으나 관측하고자 하는 대상을 찾는 것은 어렵다

        (2)재현율, 민감도(Recall / Sensitivity)
            - 실제 positive인 것 중 positive로 예측한 것의 비율
            - TPR(true positive rate)f라고도 함
            - ex: 실제 스팸메일 중 스팸메일로 예측한 것의 비율 / 금융 사기 데이터 중 사기로 예측한 비율
            - 함수 : recall_score(y실제값, y예측값)

        (3)정밀도(Precision)
            - positive로 예측한 것 중 실제로 positive인 것의 비율
            - PPV(positive predictive value)라고도 함
            - ex: 스팸메일로 예측한 것 중 실제 스팸메일의 비율 / 금융 사기로 예측한 것 중 실제 금융 사기 사건의 비율
            - 함수 : precision_scroe(y실제값, y예측값)
            
        (4)F1 score
            - 정밀도와 재현율의 조화평균
            - recall과 precision의 비슷할 수록 높은 값을 가짐 
            - f1 score가 높다는 것은 정밀도와 재현율 모두 좋다고 판단
            - 함수 : f1_score(y실제값, y예측값)

        (5) 기타 
            - 특이도(Specificity)
                - 실제 negative인 것들 중 negative로 맞게 예측한 것의 비율
                - TNR(true negative rate)라고도 함
            - 위양성률(fall Out)
                - 실제로는 negative이지만 positive로 잘못 예측한 것의 비율 = 1 - 특이도
                - FPR(false positive rate)라고도 함
                - FPR = FP / (TN + FP)
        
        (6)PR curve와 AP score
            - PR curve(precision recall curve)
                - 0과 1 사이의 모든 임계값에 대하여 재현율(recall)과 정밀도(precision)의 변화를 이용한 평가 지표
                - X축에 재현율, Y축에 정밀도를 놓고 임계값을 1 to 0로 변화할 때, 두값의 변화를 선그래프로 그림
            - AP score(average precision score)
                - PR curve의 성능평가 지표를 하나의 점수로 표현한 것
                - PR curve의 선아래 면적을 계산한 값으로 높을 수록 성능이 우수
            
            - recall 관점에서 성능 지표 확인
              (threshold를 변경하여 recall 점수를 상승시켰을 때, precision의 하강률을 지표화 한 것)
                - precision_recall_curve() : threshold 별 recall / precision 점수를 반환하는 함수
                - plot_precision_recall_curve() : PR curve를 그려주는 함수
                - average_precision_score() : AP score 계산 함수

        (6)ROC curve와 AUC score
            - ROC(receiver operating characteristic) curve
                - FPR(false positive rate) 
                    - 위양성율(fall out)
                    - 1-TNR(특이도)
                    - 실제로는 음성이나 잘못하여 양성으로 예측한 비율 FP/(TN+FP)
                - TPR(true positive rate)
                    - 재현율(recall)
                    - 실제 양성 중 맞게 예측한 비율 TP/(FN+TP)
                - ROC 곡선
                    - 이진 분류의 모델 성능 평가 지표
                    - 양성 클래스와 음성 클래스의 탐지의 중요도가 비슷할 때 사용(개 vs 고양이)
                    - 불균형 데이터 셋 평가시 사용
                    - FPR을 X축, TPR을 Y축으로 함
                    - 임계값을 변경하여 FPR의 변화에 따라 TPR의 변화를 나타내는 곡선
                    - PR curve
                        - 양성 클래스 탐지가 음성 클래스 탐지보다 중요한 경우 사용(암환자 vs 비환자)
            - AUC
                - ROC곡선의 아래 면적
                - 0~1 사이의 실수 >> 클수록 좋음
                - 0.5~0.6 : 성능 나쁨 / 0.6~0.7 : 의미는 있으나 좋은 모델X / 0.7~0.8 : 괜찮은 모델 / 0.8~0.9 : 좋은 모델 / 0.9~1.0 : 매우 좋은 모델
                - 가장 완벽한 것은 FPR이 0이고 TPR이 1인 것이다. 일반 적으로 FPR이 작을 때 (0에 가까울때) TPR이 높은 경우가 좋은 상황이다. 
                  선 아래의 면적이 넓은 곡선이 나올 수록 좋은 모델이다.

            - 함수
                - roc_curve(y값, 예측확률) : FPR, TPR, Thresholds(임계치)
                - roc_auc_score(y값, 예측확률) : AUC점수 반환


    2) 회귀 평가 방법
        (1)MSE(mean square error)
        (2)RMSE(root mean square error)
        (3)R^2(결정 계수)
    3) sklearn 평가 함수

'''
#MNIST data set을 이용한 정확도 평가
#1 1) (1)
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.arrayprint import repr_format
from scipy.sparse.construct import random
from sklearn.datasets import load_digits
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix

mnist = load_digits()
#print(mnist.keys())
    #dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
#print(mnist.feature_names)
    #['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', ... , 'pixel_7_6', 'pixel_7_7']
X = mnist['data']
y = mnist['target']

#print(X.shape, y.shape)        #(1797, 64) (1797,)
#print(np.unique(y, return_counts=True))        #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

print('y[500] = ', y[500])
img = X[500].reshape(8,8)
plt.imshow(img, cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.show() #그림이 나옴


#print(np.unique(y, return_counts=True))
    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
    # 균형 데이터임 
# 불균형 데이터로 변경하기

y = np.where(y == 9, 1, 0) #True : 1, False : 0
#print(np.unique(y, return_counts=True)) 
    #(array([0, 1]), array([1617,  180], dtype=int64))
#np.unique(y, return_counts=True)[1]/y.size
    #array([0.89983306, 0.10016694])

#데이터 셋 분할(훈련, 테스트)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
#    X_train.shape, X_test.shape, y_train.shape, y_test.shape
#     ((1347, 64), (450, 64), (1347,), (450,))

np.unique(y_train, return_counts=True)[1]/y_train.size
#   array([0.89977728, 0.10022272])



'''
모델 생성 및 학습
1. Dummy Model 정의
    - target label 중 무조건 최빈값으로 예측하는 모델 정의
'''
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

#최빈값으로 예측하는 dummy 모델 생성
dummy_model = DummyClassifier(strategy='most_frequent')

#학습
dummy_model.fit(X_train, y_train)

#추론 + 평가
pred_train = dummy_model.predict(X_train)
pred_test = dummy_model.predict(X_test)

print('train 정확도 : {}, test 정확도 : {}'.format(accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)))

np.unique(y_train, return_counts=True)
#   (array([0, 1]), array([1212,  135], dtype=int64))
np.unique(pred_train), np.unique(pred_test)
#   (array([0]), array([0]))

'''
혼동행렬(confusion matrix)
- 분류의 평가지표의 기본 지표으로 사용된다.
- 혼동행렬을 이용해 다양한 평가지표(정확도, 재현률, 정밀도, F1 점수, AUC 점수)를 계산할 수 있다.
- 함수: confusion_matrix(정답, 모델예측값)
- 결과의 0번축: 실제 class, 1번 축: 예측 class
                      predict
                    Neg(0)    Pos(1)
    actual Neg(0)     TN        FP
           Pos(1)     FN        TP

    -TP(True Positive) - 양성으로 예측했는데 맞은 개수
    -TN(True Negative) - 음성으로 예측했는데 맞은 개수
    -FP(False Positive) - 양성으로 예측했는데 틀린 개수 (음성을 양성으로 예측)
    -FN(False Negative) - 음성으로 예측했는데 틀린 개수 (양성을 음성으로 예측)
'''

from sklearn.metrics import confusion_matrix
print("train 혼동 행렬")
confusion_matrix(y_train, pred_train)
#   array([[1212,    0],
#           [ 135,    0]], dtype=int64)

print('test 혼동 행렬')
confusion_matrix(y_test, pred_test)
#   array([[405,   0],
#          [ 45,   0]], dtype=int64)

import matplotlib.pyplot as plt

#train data를 가지고 표현
fig, ax = plt.subplots(1, 1, figsize=(5,5))
plot_confusion_matrix(dummy_model, X_train, y_train, display_labels=['Neg', 'Pos'], values_format='d', cmap='Blues', ax=ax)
                    # 학습한 모델 ,예측할 X, 정답 y,                                 값들은 정수로     색깔은 파란색 계열, axes(그래프를 그릴 axes)
plt.title('train confusion matrix')
plt.show()

#test data를 가지고 표현
fig, ax = plt.subplots(1,1, figsize=(5,5))
plot_confusion_matrix(dummy_model, X_test, y_test, display_labels=['Neg', 'Pos'], values_format='d')
plt.title('test confusion matrix')
plt.show()


#
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix
acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(y_test, pred_test)
print('각각의 함수를 이용하여 acc, recall, precision, f1 점수 구하기')
print('accuracy_score_train = {}, accuracy_scroe_test = {}'.format(acc_train, acc_test))
recall_train = recall_score(y_train, pred_train)
recall_test = recall_score(y_test, pred_test)
print('recall_score_train = {}, recall_scroe_test = {}'.format(recall_train, recall_test))
precision_train = precision_score(y_train, pred_train)
precision_test = precision_score(y_test, pred_test)
print('precision_score_train = {}, precision_scroe_test = {}'.format(precision_train, precision_test))
f1_train = f1_score(y_train, pred_train)
f1_test = f1_score(y_test, pred_test)
print('f1_score_train = {}, f1_scroe_test = {}'.format(f1_train, f1_test))
print('\n\n')

#calssification report를 이용하여 한번에 구하기!
from sklearn.metrics import classification_report
report = classification_report(y_train, pred_train)
print(report)


##머신러닝 모델을 이용한 학습 실습
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

#모델 생성 및 학습
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

#평가 및 예측
pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)

#혼동 행렬
confusion_matrix(y_train, pred_train)
#   array([[1167,   45],
#         [  27,  108]], dtype=int64)

plot_confusion_matrix(tree, X_train, y_train, display_labels=['Not 9(Neg)','9(Pos)'], cmap='Blues')
plt.show()

confusion_matrix(y_test, pred_test)
#   array([[394,  11],
#          [ 11,  34]], dtype=int64)
plot_confusion_matrix(tree, X_test, y_test, display_labels=['Not 9(Neg)','9(Pos)'], cmap='Blues')
plt.show()

accuracy_score(y_train, pred_train), recall_score(y_train, pred_train), precision_score(y_train, pred_train), f1_score(y_train, pred_train)
    # (0.9465478841870824, 0.8, 0.7058823529411765, 0.7500000000000001)
accuracy_score(y_test, pred_test), recall_score(y_test, pred_test), precision_score(y_test, pred_test), f1_score(y_test, pred_test)
    # (0.9511111111111111, 0.7555555555555555, 0.7555555555555555, 0.7555555555555555)

train_report = classification_report(y_train, pred_train)
'''
              precision    recall  f1-score   support

           0       0.98      0.96      0.97      1212
           1       0.71      0.80      0.75       135

    accuracy                           0.95      1347
   macro avg       0.84      0.88      0.86      1347
weighted avg       0.95      0.95      0.95      1347
'''

test_report = classification_report(y_test, pred_test)
'''
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       405
           1       0.76      0.76      0.76        45

    accuracy                           0.95       450
   macro avg       0.86      0.86      0.86       450
weighted avg       0.95      0.95      0.95       450
'''


'''
재현율과 정밀도의 관계
1. 재현율 : 실제 positive인 것 중 positive로 예측한 것의 비율
        - 실제 양성을 음성으로 잘못 판단하면 업무상 영향이 큰 경우
        - False Negative를 낮추는데 초점을 맞춤
        - 암환자 판정 모델, 보험사기 적발 모델 등
2. 정밀도 : positive로 예측한 것 중 실제로 positive인 것의 비율
        - 실제 음성을 양성으로 잘못 판단하면 업무상 영향이 큰 경우
        - False Positive를 낮추는데 초점을 맞춤
        - 스팸메일 판정
3. Threshold(임계값) 변경을 이용한 재현율, 정밀도 변환
    1) Threshoold 
        (1) 정의 : 모델이 분류의 답을 결정할 때의 기준이 되는 값
        (2) 사용 : 정밀도나 재현율을 특히 강조해야 하는 상황일 경우 임계값 변경을 통해 평가 수치를 올릴 수 있다.
        (3) 주의사항 : 극단적으로 임계점을 올리나가 낮춰서 한쪽의 점수를 높이면 안된다. 
                      (ex: 암환자 예측시 재현율을 너무 높이면 정밀도가 낮아져 걸핏하면 정상인을 암환자로 예측하게 된다.)
    2) 임계값 변경에 따른 정밀도와 재현율
        (1) 임계값 ↑ : 양성 예측 기준이 높아져 음성으로 예측되는 샘플 수가 증가 >> 정밀도↑ & 재현율↓
        (2) 임계값 ↓ : 양성 예측 기준이 낮아져 양성으로 예측되는 샘플 수가 증가 >> 재현율↑ & 정밀도↓
        (3) 재현율과 정밀도는 음의 상관관계
        (4) 재현율과 위양성율은 양의 상관관계
'''

## 임계값 변화에 따른 recall, precision 변화
# model.predict(X) : 분류 >> 최종 class에 대한 추론 결과
# model.predict_prob(X) : 추론 확률(0일 확률, 1일 확률)
pred_test = tree.predict(X_test)
print(pred_test[-10:], '예측 결과')

prob_test = tree.predict_proba(X_test)
print(prob_test[-10:], '예측 확률(0, 1)')

import pandas as pd
import matplotlib.pyplot as plt
#임계값을 변화 시켰을 때, recall과 precision의 변화율을 확인하는 지표
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, prob_test[:, 1])
    # (y정답, 1(positive)일 확률)
    # precision.shape = (8, ) / recalls.shape = (8, ) /  thresholds.shape = (7, )
    # threshold의 경우 하나 적게 나옴

print(thresholds)
print(precisions)
print(recalls)

#threshold에 1 추가
thresholds = np.append(thresholds,1)

df = pd.DataFrame({ 
    'Threshold' : thresholds,
    'Recall' : recalls,
    'Precision' : precisions}
)
'''
   Threshold    Recall  Precision
0   0.008264  1.000000   0.100000
1   0.013043  0.977778   0.107579
2   0.033898  0.844444   0.431818
3   0.040000  0.822222   0.513889
4   0.149254  0.822222   0.560606
5   0.545455  0.755556   0.755556
6   0.750000  0.622222   0.823529
7   1.000000  0.000000   1.000000 

threshold가 0.545455일 때, 가장 recall과 precision의 값이 유사
'''

# precision_recall_curve의 결과를 선그래프로 확인
# X축 thresholds, y: recall/precision
plt.figure(figsize=(7,7))
plt.plot(thresholds, precisions, marker='o', label='Precision')
plt.plot(thresholds, recalls, marker='o', label='Recall')

plt.xlabel("Threshold")
plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
plt.grid(True)
plt.title('threshold & precision & recall')
plt.show()

#Binarizer - 임계값 변경하기
#   - Transformer로 양성 여부를 선택하는 임계값을 변경할 수 있다

prob_test[:,1]
#array([0.00826446, 0.01304348, 0.01304348, 0.01304348, 0.01304348,
#       0.01304348, 0.04      , 0.01304348, 0.01304348, 0.01304348,
#       ...
#       0.01304348, 0.54545455, 0.01304348, 0.01304348, 0.01304348,
#       0.54545455, 0.01304348, 0.01304348, 0.01304348, 0.01304348])

threshold = 0.7

# prob_test의 값이 0.7보다 크면 양성으로, 작으면 음성으로 바꾼다
r = np.where(prob_test[:, 1] > threshold, 1, 0)

np.unique(r, return_counts=True)
    #(array([0, 1]), array([416,  34], dtype=int64))

recall_score(y_test, r), precision_score(y_test, r)
    #(0.6222222222222222, 0.8235294117647058)

#Binarizer 이용하기!
from sklearn.preprocessing import Binarizer

exam = [[0.3, 0.8, 0.7, 0.6, 0.5]]

threshold1 = 0.5 
b = Binarizer(threshold = threshold1)
b.fit_transform(exam)
    # [[0. 1. 1. 1. 0.]]

threshold2 = 0.6
b = Binarizer(threshold = threshold2)
b.fit_transform(exam)
    # [[0. 1. 1. 0. 0.]]

binarizer1 = Binarizer(threshold=0.5)
pred_test_0_5 = binarizer1.fit_transform(prob_test)[:, 1]
print('threshold = 0.5일 때 recall_score = {}, precision_score = {}'.format(recall_score(y_test, pred_test_0_5), precision_score(y_test, pred_test_0_5)))

binarizer2 = Binarizer(threshold=0.1)
pred_test_0_1 = binarizer2.fit_transform(prob_test)[:, 1]
print('threshold = 0.1일 때 recall_score = {}, precision_score = {}'.format(recall_score(y_test, pred_test_0_1), precision_score(y_test, pred_test_0_1)))
#recall값 상승 / precision값 하강 >> 실제로 positive인 것을 더 많이 맞췄다.

binarizer3 = Binarizer(threshold=0.7)
pred_test_0_7 = binarizer3.fit_transform(prob_test)[:, 1]
print('threshold = 0.7일 때 recall_score = {}, precision_score = {}'.format(recall_score(y_test, pred_test_0_7), precision_score(y_test, pred_test_0_7)))
#precision값 상승 >> positive로 예측한 것들이 더 많이 맞췄다.




#PR curve와 AP score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#tree모델 생성 및 학습
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
rf = RandomForestClassifier(max_depth=2, n_estimators=200, random_state=0)

tree.fit(X_train, y_train)
rf.fit(X_train, y_train)

#추론
prob_test_tree = tree.predict_proba(X_test)[:, 1] #양성인 것 찾기
prob_test_rf = rf.predict_proba(X_test)[:, 1]

#평가 
# 평가지표 - precision_recall_curve (p, r, th)
precision1, recall1, threshold1 = precision_recall_curve(y_test, prob_test_tree) # (y정답, 양성확률)
precision2, recall2, threshold2 = precision_recall_curve(y_test, prob_test_rf)
precision1.shape, recall1.shape, threshold1.shape, precision2.shape, recall2.shape, threshold2.shape
#   ((8,), (8,), (7,), (198,), (198,), (197,))


#표로 확인
pd.DataFrame({
    'Threshold' : np.append(threshold1, 1),
    'Precision' : precision1,
    'Recall' : recall1
})
'''
	Threshold	Precision	Recall
0	0.008264	0.100000	1.000000
1	0.013043	0.107579	0.977778
2	0.033898	0.431818	0.844444
3	0.040000	0.513889	0.822222
4	0.149254	0.560606	0.822222
5	0.545455	0.755556	0.755556
6	0.750000	0.823529	0.622222
7	1.000000	1.000000	0.000000
'''

df_rf= pd.DataFrame({
        "Threshold": np.append(threshold2, 1), 
        "Precision": precision2,
        "Recall":recall2
})

'''
Threshold	Precision	Recall
0	0.080666	0.228426	1.000000
1	0.081213	0.224490	0.977778
2	0.083313	0.225641	0.977778
3	0.083402	0.226804	0.977778
4	0.083429	0.227979	0.977778
..	  ....      	...	        ...
193	0.418662	1.000000	0.088889
194	0.429407	1.000000	0.066667
195	0.430382	1.000000	0.044444
196	0.456759	1.000000	0.022222
197	1.000000	1.000000	0.000000
'''


plt.figure(figsize=(9,7))
plt.plot(recall2, precision2, marker = 'o', label='RF')
plt.plot(recall1, precision1, marker = 'x', label='Tree')

plt.legend()    #범례
plt.title('randomforest, tree test PR curve')
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid(True)
plt.show()


#plot_precision_recall_curve() 이용해서 그래프 그리기
fig, ax = plt.subplots(1,2, figsize=(15,7))

plot_precision_recall_curve(tree, X_test, y_test, ax=ax[0])
ax[0].set_title('Tree PR Curve')
ax[0].grid(True)

plot_precision_recall_curve(rf, X_test, y_test, ax=ax[1])
ax[1].set_title('RandomForest PR Curve')
ax[1].grid(True)

plt.show()

# AP Score 조회
print("Decision Tree AP Score", average_precision_score(y_test, prob_test_tree)) #(y 정답, 양성일 확률)
    # Decision Tree AP Score 0.6766948888666132 
print("RandomForest AP Score", average_precision_score(y_test, prob_test_rf))
    # RandomForest AP Score 0.8486345312919419

# ROC 커브
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve

# roc_curve(y값, 양성확률): FPRs(fall out-위양성율), TPRs(recall), Thresholds

fprs1, tprs1, thresholds1 = roc_curve(y_test, prob_test_tree) #tree 예측결과
fprs2, tprs2, thresholds2 = roc_curve(y_test, prob_test_rf) #random forest 예측결과

fprs1.shape, tprs1.shape, thresholds1.shape, fprs2.shape, tprs2.shape, thresholds2.shape
    # ((8,), (8,), (8,), (30,), (30,), (30,))

pd.DataFrame({
    "TH" : threshold1,
    "FPR" : fprs1,
    "TPR" : tprs1
})
'''
	TH	FPR	TPR
0	1.750000	0.000000	0.000000
1	0.750000	0.014815	0.622222
2	0.545455	0.027160	0.755556
3	0.149254	0.071605	0.822222
4	0.040000	0.086420	0.822222
5	0.033898	0.123457	0.844444
6	0.013043	0.901235	0.977778
7	0.008264	1.000000	1.000000
'''

# ROC curve 직접 작성
plt.figure(figsize=(8,6))
plt.plot(fprs1, tprs1, marker='o', label='Tree Model') #X: FPR, Y: TPR
plt.plot(fprs2, tprs2, marker='x', label='RF model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# roc auc score
print("DecisionTree roc auc 점수:", roc_auc_score(y_test, prob_test_tree)) # (y정답, 양성확률)
print("RandomForest roc auc 점수:", roc_auc_score(y_test, prob_test_rf))

# plot_roc_curve
_, ax = plt.subplots(1,1, figsize=(8,6)) # 변수명 `_`: 사용하지 않겠다.
plot_roc_curve(tree, X_test, y_test, ax=ax)
plot_roc_curve(rf, X_test, y_test, ax=ax)
plt.grid(True)
plt.show()