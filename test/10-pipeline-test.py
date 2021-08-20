from hypothesis.strategies import *
from hypothesis import given
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.utils import all_estimators
from sklearn.base import MetaEstimatorMixin


def estimator_st(): 
    return sampled_from([x[1] for x in all_estimators() if MetaEstimatorMixin
        not in x[1].mro()])


@given(estimator_st())
def test(estimator):
    try: 
        Pipeline([
            ('scaler', StandardScaler()),
            ('ohc', OneHotEncoder()),
            ('ord', OrdinalEncoder()),
            #  ('est', estimator())
            ])
    except TypeError as e:
        print(estimator.mro(), e)
