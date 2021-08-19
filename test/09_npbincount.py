from hypothesis.extra.numpy import arrays, integer_dtypes
from hypothesis import given, settings
from hypothesis.strategies import tuples, integers, composite
import numpy as np


@composite
def array1d(draw):
    dtype = draw(integer_dtypes())
    max_value = 17504615792640-0 # found thru experimentation and np throwing
    #  malloc errors

    max_ = min(np.iinfo(dtype).max, 
            draw(integers(min_value=0, max_value=max_value)))
    shape = tuples(integers(min_value=1, max_value=100000),)
    return draw(arrays(dtype=dtype, shape=shape, elements=integers(min_value=0,
        max_value=max_)))

@given(array1d())
@settings(max_examples=200)
def test_bincount(arr):
    res = np.bincount(arr)
    assert res.shape[0] == np.amax(arr) + 1
