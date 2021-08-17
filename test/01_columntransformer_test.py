from hypothesis.strategies import *
from hypothesis import given
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import numpy as np


#  applies transformers to columns of an array or pandas DataFrame
def make_df():
    return data_frames([
        column('city', elements=sampled_from(['chicago', 'london', 'new york'])),
        column('B', dtype=float)], index=range_indexes(min_size=40))


@given(make_df())
def test_column_transform(df):
    n_uniq = len(df.city.unique())
    ct = ColumnTransformer(
        [
            ('city_category', OneHotEncoder(dtype=np.int32), ['city'])
            ],
        remainder='passthrough')
    t = ct.fit_transform(df)
    assert np.sum(t[:, :n_uniq]) == df.shape[0]


@given(make_df())
def test_column_transform2(df):
    n_uniq = len(df.city.unique())
    ct = ColumnTransformer(
        [
            ('city_category', OrdinalEncoder(dtype=np.int32), ['city'])
            ],
        remainder='passthrough')
    t = ct.fit_transform(df)
    assert np.max(t[:, 0]) == n_uniq -1
