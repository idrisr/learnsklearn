from hypothesis.strategies import lists
from hypothesis import given
from strategies import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np


@given(cats(10))
def test(s):
    enc = OneHotEncoder()
    enc.fit(s)
    assert enc.transform(s).toarray().shape[0] == len(s)
    assert np.all(enc.inverse_transform(enc.transform(s)) == s)


@given(cats(1000))
def test_shape(s):
    enc = OneHotEncoder(categories=[gender_cats, browser_cats, from_cats])
    assert enc.fit_transform(s).shape == (len(s), 8, )
    assert np.max(enc.fit_transform(s)) == 1.0
    assert np.min(enc.fit_transform(s)) == 0.0
