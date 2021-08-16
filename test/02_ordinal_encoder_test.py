from hypothesis.strategies import composite, sampled_from, tuples, lists, one_of
from hypothesis import given

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pytest

#  Encode categorical features as a one-hot numeric array.

gender_cats = ['male', 'female']
browser_cats = ['Safari', 'IE', 'Firefox']
from_cats = ["Europe", "US", "Japan"]

@composite
def cats(draw, max_size=1):
    gender = sampled_from(gender_cats)
    browser = sampled_from(browser_cats)
    from_ = sampled_from(from_cats)
    sample = tuples(gender, browser, from_)

    return draw(lists(sample, max_size=max_size, min_size=1))


@given(cats(20))
def test_ordencoder(s):
    enc = OrdinalEncoder(categories='auto')
    assert np.min(enc.fit_transform(s)) == 0.0


@given(cats(20))
def test_ordencoder_missing(s):
    enc = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=42,
            dtype=float)
    # with auto for categories, you can handle missing values
    enc.fit(s)
    enc.transform([['female', 'IE', 'US']])


@given(cats(20))
def test_ordencoder_missing(s):
    enc = OrdinalEncoder(categories=[[], browser_cats, from_cats],
            handle_unknown='use_encoded_value', unknown_value=42,
            dtype=float)

    # you get a valueerror for a missing when you're explicit about categories
    with pytest.raises(ValueError):
        enc.fit(s)


@given(cats(20))
def test_encoder_size(s):
    enc = OrdinalEncoder()
    m = enc.fit_transform(s)
    assert m.shape == (len(s), 3, )
