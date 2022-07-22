import numpy as np
import pytest

from qmkpy import value_density, total_profit_qmkp
from qmkpy.util import chromosome_from_assignment, assignment_from_chromosome
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

@pytest.mark.parametrize("assignments,expected",
                         ((np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]), [1, 0, 0, 2]),
                          (np.array([[0, 0], [0, 1], [1, 0], [0, 0]]), [-1, 1, 0, -1]),
                          (np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0]]), [2, 1, -1, 0]),
                         ))
def test_assignment_to_chromosome(assignments, expected):
    chromosome = chromosome_from_assignment(assignments)
    assert np.all(chromosome == expected)

@pytest.mark.parametrize("expected,chromosome",
                         ((np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]), [1, 0, 0, 2]),
                          (np.array([[0, 0], [0, 1], [1, 0], [0, 0]]), [-1, 1, 0, -1]),
                          (np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0]]), [2, 1, -1, 0]),
                         ))
def test_chromosome_to_assignment(expected, chromosome):
    num_ks = np.shape(expected)[1]
    assignments = assignment_from_chromosome(chromosome, num_ks)
    assert np.all(assignments == expected)

@pytest.mark.parametrize("assignments,chromosome",
                         ((np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]), [1, 0, 0, 2]),
                          (np.array([[0, 0], [0, 1], [1, 0], [0, 0]]), [-1, 1, 0, -1]),
                          (np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0]]), [2, 1, -1, 0]),
                         ))
def test_assignment_chromosome_swap(assignments, chromosome):
    num_ks = np.shape(assignments)[1]
    _chromosome = chromosome_from_assignment(assignments)
    _assign = assignment_from_chromosome(_chromosome, num_ks)
    assert np.all(_assign == assignments) and np.all(_chromosome == chromosome)
