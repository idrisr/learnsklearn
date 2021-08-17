from sklearn.preprocessing import StandardScaler
from hypothesis import given
from hypothesis.strategies import *
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

# create some numpy arrays
# scale them
# make sure the mean and std is 0 and 1

@given(arrays(float, shape=tuples(integers(2, 10), integers(2, 10)),
    elements=floats(1, 100), unique=True, fill=nothing()))
def test_scaler(m):
    s = StandardScaler()
    t = s.fit_transform(m)
    assert np.mean(t[:, -1]) == pytest.approx(0, abs=0.1)
    assert np.std(t[:, -1]) == pytest.approx(1, abs=0.1)
    i = s.inverse_transform(t)
    np.testing.assert_allclose(i, m)
