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


SOLVERS = (random_assignment, constructive_procedure, fcs_procedure, round_robin)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_feasibility_hp(solver):
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 2, 4, 6], [2, 2, 8, 10], [4, 8, 4, 12], [6, 10, 12, 6]])
    profit3 = np.array([[2, 2, 3, 4], [2, 2, 5, 6], [3, 5, 3, 7], [4, 6, 7, 4]])
    profits = np.array([profit1, profit2, profit3])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    solution = solver(profits, weights, capacities)
    print(solution)
    assert checks.is_feasible_solution(solution, profits, weights, capacities)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_hp(solver):
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 2, 4, 6], [2, 2, 8, 10], [4, 8, 4, 12], [6, 10, 12, 6]])
    profit3 = np.array([[2, 2, 3, 4], [2, 2, 5, 6], [3, 5, 3, 7], [4, 6, 7, 4]])
    profits = np.array([profit1, profit2, profit3])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert total_profit >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_large_hp(solver):
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
    profits = profits @ profits.transpose([0, 2, 1])
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert total_profit >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_no_assignments_hp(solver):
    weights = [10, 5, 14, 52]
    capacities = [1, 4, 2, 1]
    num_elements = len(weights)
    num_knapsacks = len(capacities)
    profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
    profits = profits @ profits.transpose([0, 2, 1])
    solution = solver(profits, weights, capacities)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert (total_profit == 0) and np.all(solution == 0)

@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_hp_vs_homog(solver):
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
    profits = profits @ profits.transpose([0, 2, 1])
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution_hp = solver(profits, weights, capacities)
    solution_homo = solver(profits[0], weights, capacities)
    assert np.any(solution_hp != solution_homo)
