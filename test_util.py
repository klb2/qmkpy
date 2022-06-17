import numpy as np
import pytest

import util

@pytest.mark.parametrize("array,expected",
                         (([0, 1, 0], True), ([1, 1, 1, 1], True),
                          ([0, 0], True), ([1], True),
                          ([0, -1], False), ([1, 2, 3], False),
                          (np.array([[0, 1], [1, 0]]), True),
                          (np.array([[-1, 1], [1, 0]]), False),
                          (np.array([[-1, 1], [1, 1]]), False),
                          (np.array([[0, 1, 1, 1, 0]]), True),
                          (np.zeros((5, 5)), True), (np.ones((5, 5)), True),
                          (np.random.rand(10, 10), False),
                         ))
def test_is_binary(array, expected):
    _isbinary = util.is_binary(array)
    assert _isbinary == expected
