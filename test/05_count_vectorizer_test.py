from sklearn.feature_extraction.text import CountVectorizer
from hypothesis.strategies import *
from hypothesis import given
import numpy as np


# Convert a collection of text documents to a matrix of token counts


def corpus_strategy():
    return lists(min_size=2, 
            elements=lists(elements=sampled_from('welcome to the jungle'.split()),
                min_size=4).map(lambda x: " ".join(x))
            )

@given(corpus_strategy())
def test_vectorizer(c):
    v = CountVectorizer(analyzer='word')
    x = v.fit_transform(c)
    assert x.toarray().shape[0] == len(c)
