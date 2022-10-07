import numpy as np
import pytest

from qmkpy import total_profit_qmkp
from qmkpy.algorithms import (
    constructive_procedure,
    fcs_procedure,
    random_assignment,
    round_robin,
)
from qmkpy import checks


SOLVERS = (constructive_procedure, fcs_procedure, random_assignment, round_robin)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_feasibility(solver):
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    solution = solver(profits, weights, capacities)
    print(solution)
    assert checks.is_feasible_solution(solution, profits, weights, capacities)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver(solver):
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert total_profit >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_large(solver):
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert total_profit >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_no_assignments(solver):
    weights = [10, 5, 14, 52]
    capacities = [1, 4, 2, 1]
    num_elements = len(weights)
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert (total_profit == 0) and np.all(solution == 0)
