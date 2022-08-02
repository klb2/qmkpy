import numpy as np
import pytest

from qmkpy import value_density, total_profit_qmkp
from qmkpy.util import chromosome_from_assignment, assignment_from_chromosome
from qmkpy import checks


@pytest.mark.parametrize("assignments,expected",
                         (([1, 3], [5, 6/2, 12/3, 8/4]),
                          ([], [1, 1/2, 2/3, 3/4]),
                          ([2], [3, 5/2, 2/3, 9/4]),
                          ([0, 1, 2, 3], [7, 11/2, 14/3, 17/4]),
                         ))
def test_value_density_list_assign(assignments, expected):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    vd = value_density(profits, weights, assignments)
    assert np.all(expected == vd)

@pytest.mark.parametrize("assignments,expected",
                         (([[0, 0], [1, 0], [0, 0], [1, 0]], [[5, 1], [6/2, 1/2], [12/3, 2/3], [8/4, 3/4]]),
                          ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1/2, 1/2, 1/2], [2/3, 2/3, 2/3], [3/4, 3/4, 3/4]]),
                          ([[1, 0], [1, 0], [1, 1], [1, 0]], [[7, 3], [11/2, 5/2], [14/3, 2/3], [17/4, 9/4]]),
                         ))
def test_value_density_matrix(assignments, expected):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    assignments = np.array(assignments)
    num_ks = np.shape(assignments)[1]
    vd = value_density(profits, weights, assignments)
    assert np.all(expected == vd) and np.all(np.shape(vd) == (len(weights), num_ks))

@pytest.mark.parametrize("assignments,expected",
                         (([1, 3], [5, 12/3]),
                          ([], [1, 1/2, 2/3, 3/4]),
                          ([2], [3, 5/2, 9/4]),
                          ([0, 1, 2, 3], []),
                         ))
def test_value_density_reduced_output_list_assignment(assignments, expected):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    vd, unassigned = value_density(profits, weights, assignments, reduced_output=True)
    expected_unassigned = set(range(len(weights))).difference(assignments)
    assert np.all(expected == vd) and set(unassigned) == expected_unassigned

@pytest.mark.parametrize("assignments,expected",
                         (([[0, 0], [0, 1], [0, 0], [0, 1]], [[1, 5], [2/3, 12/3]]),
                          ([[0, 0], [0, 0], [0, 0], [0, 0]], [[1, 1], [1/2, 1/2], [2/3, 2/3], [3/4, 3/4]]),
                          ([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 3, 1], [1/2, 5/2, 1/2], [3/4, 9/4, 3/4]]),
                          ([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], np.empty((0, 3))),
                         ))
def test_value_density_reduced_output_matrix_assignment(assignments, expected):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    vd, unassigned = value_density(profits, weights, assignments, reduced_output=True)
    _assigned = np.where(np.any(assignments, axis=1))[0]
    expected_unassigned = set(range(len(weights))).difference(_assigned)
    assert np.all(expected == vd) and set(unassigned) == expected_unassigned


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

def test_profit_fail():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    assignments = np.array([[0, 0, 1],
                            [2, 0, 0],
                            [-1, 0, 0],
                            [0, 0, 1]])
    with pytest.raises(ValueError):
        _objective = total_profit_qmkp(profits, assignments)

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
