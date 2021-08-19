from hypothesis.strategies import *
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames, range_indexes
from sklearn.base import BaseEstimator, TransformerMixin

def make_df():
    return data_frames([
        column('rooms', elements=integers(min_value=100, max_value=800)),
        column('households', elements=integers(min_value=10, max_value=80)),
        column('population', elements=integers(min_value=1000,
            max_value=1000000))],
        index=range_indexes(min_size=5))


class CombinedAttributesAddr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        X['rooms_per_household'] = X['rooms'] / X['households']
        X['population_per_household'] = X['population'] / X['households']
        return X

@given(make_df())
def test_transform(df):
    comb = CombinedAttributesAddr()
    t = comb.fit_transform(df)
    assert 'rooms_per_household' in t
    assert 'population_per_household' in t
