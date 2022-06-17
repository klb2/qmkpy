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

@pytest.mark.parametrize("assignments",
                         (np.zeros((4, 3)),
                          [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
                          [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
                         ))
def test_is_feasible_solution_pass(assignments):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    is_feasible = util.is_feasible_solution(assignments,
                                            capacities=capacities,
                                            weights=weights,
                                            profits=profits)
    assert is_feasible == True

@pytest.mark.parametrize("assignments",
                         (np.ones((4, 3)),
                          [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
                          [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
                          [[1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                          [[0, 0], [1, 0], [0, 1], [1, 0]],
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         ))
def test_is_feasible_solution_fail(assignments):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    is_feasible = util.is_feasible_solution(assignments,
                                            capacities=capacities,
                                            weights=weights,
                                            profits=profits)
    assert is_feasible == False

@pytest.mark.parametrize("assignments",
                         (np.ones((4, 3)),
                          [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
                          [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
                          [[1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                          [[0, 0], [1, 0], [0, 1], [1, 0]],
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         ))
def test_is_feasible_solution_raise(assignments):
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    with pytest.raises(ValueError) as e_info:
        is_feasible = util.is_feasible_solution(assignments,
                                                capacities=capacities,
                                                weights=weights,
                                                profits=profits,
                                                raise_error=True)
