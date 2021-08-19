from hypothesis.strategies import (composite, sampled_from, tuples, lists,
floats, integers)
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np

__all__ = ['gender_cats', 'browser_cats', 'from_cats', 'cats', 'sentence',
        'corpus_strategy', 'sentence_strategy', 'Xy']

gender_cats = ['male', 'female']
browser_cats = ['Safari', 'IE', 'Firefox']
from_cats = ["Europe", "US", "Japan"]


@composite
def Xy(draw):
    i = draw(integers(min_value=10, max_value=1000))
    j = draw(integers(min_value=10, max_value=1000))
    size = (i,j,)
    X = draw(arrays(dtype=float, 
            shape=(i, j),
            elements = floats(allow_nan=False, allow_infinity=False)
            ))
    y = draw(
            arrays(dtype=int, 
                shape=(i, 1),
                elements=sampled_from(range(10)),
                ).filter(lambda x: np.unique(x).shape[0] > 5)
            )
    return (X, y)


@composite
def cats(draw, max_size=1):
    gender = sampled_from(gender_cats)
    browser = sampled_from(browser_cats)
    from_ = sampled_from(from_cats)
    sample = tuples(gender, browser, from_)

    return draw(lists(sample, max_size=max_size, min_size=1))


sentence = 'it was the best of times it was the blurst of times'
def corpus_strategy():
    return lists(min_size=2, elements=sentence_strategy(),)

def sentence_strategy():
    return lists(elements=sampled_from(sentence.split()), min_size=4).map(lambda x: " ".join(x))
