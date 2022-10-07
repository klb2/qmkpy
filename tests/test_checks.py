import numpy as np
import pytest

from qmkpy import checks


@pytest.mark.parametrize(
    "array,expected",
    (
        ([0, 1, 0], True),
        ([1, 1, 1, 1], True),
        ([0, 0], True),
        ([1], True),
        ([0, -1], False),
        ([1, 2, 3], False),
        (np.array([[0, 1], [1, 0]]), True),
        (np.array([[-1, 1], [1, 0]]), False),
        (np.array([[-1, 1], [1, 1]]), False),
        (np.array([[0, 1, 1, 1, 0]]), True),
        (np.zeros((5, 5)), True),
        (np.ones((5, 5)), True),
        (np.random.rand(10, 10), False),
    ),
)
def test_is_binary(array, expected):
    _isbinary = checks.is_binary(array)
    assert _isbinary == expected


@pytest.mark.parametrize(
    "assignments",
    (
        np.zeros((4, 3)),
        [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
    ),
)
def test_is_feasible_solution_pass(assignments):
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    is_feasible = checks.is_feasible_solution(
        capacities=capacities, weights=weights, profits=profits, assignments=assignments
    )
    assert is_feasible is True


@pytest.mark.parametrize(
    "assignments",
    (
        np.ones((4, 3)),
        [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0], [1, 0], [0, 1], [1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ),
)
def test_is_feasible_solution_fail(assignments):
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    is_feasible = checks.is_feasible_solution(
        capacities=capacities, weights=weights, profits=profits, assignments=assignments
    )
    assert is_feasible is False


@pytest.mark.parametrize(
    "assignments",
    (
        np.ones((4, 3)),
        [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0], [1, 0], [0, 1], [1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ),
)
def test_is_feasible_solution_raise(assignments):
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 3, 2, 2]
    capacities = [5, 5, 3]
    with pytest.raises(ValueError):
        checks.is_feasible_solution(
            capacities=capacities,
            weights=weights,
            profits=profits,
            assignments=assignments,
            raise_error=True,
        )


@pytest.mark.parametrize(
    "profits,weights",
    (
        ([[1, 0, 1], [0, 2, 3], [1, 3, 5]], [1, 2, 3]),
        ([[1, 1], [3, 2]], [4, 1]),
        ([[-2, -2], [-5, 2]], [-2, 0]),
        ([[2, -2], [5, 2]], None),
    ),
)
def test_dimension_check(profits, weights):
    profits = np.array(profits)
    checks.check_dimensions(profits, weights)


@pytest.mark.parametrize(
    "profits,weights",
    (
        ([[1, 0, 1], [0, 2, 3]], [1, 2, 3]),
        ([[1, 1], [3, 2]], [4, 1, 2]),
        ([[2, 0], [-5, 2], [5, 2]], None),
        ([[-2, -2], [-5, 2], [5, 2]], [-2, 0]),
        ([[-2, -2], [-5, 2], [5, 2]], [-2, 0, 3]),
    ),
)
def test_dimension_check_fail(profits, weights):
    profits = np.array(profits)
    with pytest.raises(ValueError):
        checks.check_dimensions(profits, weights)


@pytest.mark.parametrize(
    "profits,expected",
    (
        ([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]], True),
        ([[1, 1], [1, 1]], True),
        ([[1, 0, 2], [0, 3, 4], [2, 4, 5]], True),
        ([[1, 5, 2], [0, 1, 2], [6, 5, 2]], False),
        ([[1, 1, 2, 3], [1, 9, 6, 4], [2, 6, 8, 5], [3, 4, 5, 7]], True),
        ([[1, 2], [3, 4]], False),
    ),
)
def test_symmetric_check(profits, expected):
    symmetric = checks.is_symmetric_profits(profits)
    assert symmetric == expected


@pytest.mark.parametrize(
    "profits,expected",
    (
        ([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]], True),
        ([[1, 1], [1, 1]], True),
        ([[1, 0, 2], [0, 3, 4], [2, 4, 5]], True),
        ([[1, 1, 2, 3], [1, 9, 6, 4], [2, 6, 8, 5], [3, 4, 5, 7]], True),
    ),
)
def test_symmetric_check_no_raise(profits, expected):
    symmetric = checks.is_symmetric_profits(profits, raise_error=True)
    assert symmetric == expected


@pytest.mark.parametrize(
    "profits",
    (
        [[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 3, 6], [3, 5, 1, 0]],
        [[1, 5, 2], [0, 1, 2], [6, 5, 2]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [1, 2, 3, 4, 5],
        [[1, 1], [0, 0], [2, 2]],
    ),
)
def test_symmetric_check_raise(profits):
    with pytest.raises(ValueError):
        checks.is_symmetric_profits(profits, raise_error=True)
