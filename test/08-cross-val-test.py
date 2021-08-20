from hypothesis.strategies import *
from strategies import Xy
from hypothesis import given

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = datasets.load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4)
clf = svm.SVC(kernel='linear', C=1).fit(Xtr, ytr)
print(clf.score(Xte, yte))
print(cross_val_score(clf, X, y, cv=5))

@given(Xy())
def test_cv(arr):
    X = arr[0]
    y = arr[1].ravel()
    clf = BaggingClassifier()
    assert np.mean(cross_val_score(clf, X, y, cv=2)) <= 1

@given(Xy())
def test_cv_dummy(arr):
    X = arr[0]
    y = arr[1].ravel()
    clf = DummyClassifier()
    assert np.mean(cross_val_score(clf, X, y, cv=2)) <= 1
