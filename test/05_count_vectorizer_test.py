from sklearn.feature_extraction.text import CountVectorizer
from hypothesis.strategies import *
from hypothesis import given
from strategies import *
import numpy as np


@given(corpus_strategy())
def test_vectorizer(c):
    v = CountVectorizer(analyzer='word')
    x = v.fit_transform(c)
    assert x.toarray().shape[0] == len(c)
    assert x.toarray().shape[1] <= len(set(sentence.split()))
