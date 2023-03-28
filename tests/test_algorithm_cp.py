import numpy as np
import pytest

from qmkpy import total_profit_qmkp
from qmkpy.algorithms import constructive_procedure


@pytest.mark.parametrize("profits",
    (np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]]),
     np.array([[[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]],
               [[6, 0, 2, 3], [0, 3, 7, 1], [2, 7, 3, 6], [3, 1, 6, 9]],
               [[5, 2, 3, 3], [2, 6, 3, 8], [3, 3, 4, 1], [3, 8, 1, 8]]]),
    )
)
def test_cp_with_starting(profits):
    #profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    solution = constructive_procedure(
        profits, weights, capacities, starting_assignment=starting_assignment
    )
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert (
        np.all(np.shape(solution) == (len(weights), len(capacities)))
        and total_profit > 0
    )


@pytest.mark.parametrize("profits",
    (np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]]),
     np.array([[[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]],
               [[6, 0, 2, 3], [0, 3, 7, 1], [2, 7, 3, 6], [3, 1, 6, 9]],
               [[5, 2, 3, 3], [2, 6, 3, 8], [3, 3, 4, 1], [3, 8, 1, 8]]]),
    )
)
def test_cp_change_with_starting(profits):
    #profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    solution = constructive_procedure(
        profits, weights, capacities, starting_assignment=starting_assignment
    )
    _new_assignments = solution - starting_assignment
    assert np.all(_new_assignments >= 0)


@pytest.mark.parametrize("profits",
    (np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]]),
     np.array([[[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]],
               [[6, 0, 2, 3], [0, 3, 7, 1], [2, 7, 3, 6], [3, 1, 6, 9]],
               [[5, 2, 3, 3], [2, 6, 3, 8], [3, 3, 4, 1], [3, 8, 1, 8]]]),
    )
)
@pytest.mark.parametrize(
    "starting_assignment",
    (
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 0], [0, 0], [1, 0], [0, 0]],
        [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
    ),
)
def test_cp_feasibility_starting_assignment(profits, starting_assignment):
    #profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    starting_assignment = np.array(starting_assignment)
    with pytest.raises(ValueError):
        constructive_procedure(
            profits, weights, capacities, starting_assignment=starting_assignment
        )
