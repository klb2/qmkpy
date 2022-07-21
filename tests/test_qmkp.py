import numpy as np
import pytest

from qmkpy import value_density, total_profit_qmkp
from qmkpy import checks


def test_value_density():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    sel_objects = [1, 3]
    expected = np.array([5, 6/2, 12/3, 8/4])
    vd = value_density(profits, weights, sel_objects)
    assert np.all(expected == vd)

def test_value_density_reduced_output():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    sel_objects = [1, 3]
    vd = value_density(profits, weights, sel_objects, reduced_output=True)
    expected = np.array([6/2, 8/4])
    assert np.all(expected == vd)

def test_profit():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    assignments = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
    expected = 14 # KS1: 1+2+4, KS2: 0, KS3: 1+3+3
    _objective = total_profit_qmkp(profits, assignments)
    print(_objective)
    assert expected == _objective

def test_profit2():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    assignments = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
    expected = 11 # KS1: 1 + 2 + 4, KS2: 1, KS3: 3
    _objective = total_profit_qmkp(profits, assignments)
    print(_objective)
    assert expected == _objective
