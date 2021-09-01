# KNN을 이용한 iris 데이터 분류

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#데이터 load 및 split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=0)

order = [
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
]

pipeline = Pipeline(order, verbose=True)

param = {
    'knn__n_neighbors' : range(1, 11),
    'knn__p' : [1, 2]                   #1:맨해튼거리 2:유클리드거리
}

gs = GridSearchCV(pipeline, param, scoring='accuracy', cv=4, n_jobs=-1)

gs.fit(X_train, y_train)

df = pd.DataFrame(gs.cv_results_)
#print(df)
print(df[df.columns[6:]].sort_values('rank_test_score').head())

print('best score : ', gs.best_score_)

print('best params : ', gs.best_params_)

best_model = gs.best_estimator_
print(type(best_model))

pred_test = best_model.predict(X_test)

print('accuracy score : ', accuracy_score(y_test, pred_test) )