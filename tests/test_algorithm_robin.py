import numpy as np
import pytest

from qmkpy import total_profit_qmkp
from qmkpy.algorithms import round_robin


def test_rr_with_starting():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [1, 0, 0]])
    solution = round_robin(profits, weights, capacities,
                           starting_assignment=starting_assignment)
    total_profit = total_profit_qmkp(profits, solution)
    assert np.all(np.shape(solution) == (len(weights), len(capacities))) and total_profit > 0

def test_rr_change_with_starting():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [1, 0, 0]])
    solution = round_robin(profits, weights, capacities,
                           starting_assignment=starting_assignment)
    _new_assignments = solution - starting_assignment
    assert np.all(_new_assignments >= 0)

@pytest.mark.parametrize("starting_assignment", (
                         [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
                         [[2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
                         [[1, 0], [0, 0], [1, 0], [0, 0]],
                         [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                         [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
                         ))
def test_rr_feasibility_starting_assignment(starting_assignment):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array(starting_assignment)
    with pytest.raises(ValueError):
        solution = round_robin(profits, weights, capacities,
                               starting_assignment=starting_assignment)
