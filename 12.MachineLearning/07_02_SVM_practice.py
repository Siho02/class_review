############################################################# iris dataset으로 다중 클래스 분류하기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,  random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

param = {
    'kernal' : ['linear', 'rbf'],
    'C' : [0.001, 0.01, 0.1, 1, 10],
    'gamma' : [0.001, 0.01, 0.1, 1, 10]
}

svc = SVC(random_state=0)
gs = GridSearchCV(svc, param, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train_scaled, y_train)

print(X_test_scaled.shape, y_test.shape)

#best_model = gs.best_estimator_
#print(accuracy_score(y_test, best_model.predict(X_test_scaled)))