import numpy as np
import pytest

from qmkpy import total_profit_qmkp, QMKProblem
from qmkpy.algorithms import constructive_procedure, fcs_procedure
from qmkpy import checks


def test_solver_consistency():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    cp_solution = constructive_procedure(profits, weights, capacities)
    problem = QMKProblem(profits, weights, capacities,
                         algorithm=constructive_procedure)
    pr_solution, profit = problem.solve()
    assert np.all(pr_solution == cp_solution) and profit > 0

def test_parameter_consistency():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    cp_solution = constructive_procedure(profits, weights, capacities)
    problem = QMKProblem(profits, weights, capacities,
                         algorithm=constructive_procedure)
    pr_solution, profit = problem.solve()
    assert np.all(problem.capacities == [5, 5, 3])

def test_parameter_writeable():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    problem = QMKProblem(profits, weights, capacities)
    with pytest.raises(ValueError):
        problem.capacities[0] = 2
    assert np.all(problem.capacities == [5, 5, 3])

def test_solver_set_later():
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    cp_solution = constructive_procedure(profits, weights, capacities)
    problem = QMKProblem(profits, weights, capacities)
    problem.algorithm = constructive_procedure
    pr_solution, profit = problem.solve()
    assert np.all(pr_solution == cp_solution) and profit > 0

def test_solver_with_args():
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    problem = QMKProblem(profits, weights, capacities,
                         algorithm=fcs_procedure,
                         args=(.5, 10))
    pr_solution, profit = problem.solve()
    assert checks.is_feasible_solution(pr_solution, profits, weights, capacities)

def test_solver_with_args_set_later():
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    problem = QMKProblem(profits, weights, capacities)
    problem.algorithm = fcs_procedure
    problem.args = (.5,)
    pr_solution, profit = problem.solve()
    assert checks.is_feasible_solution(pr_solution, profits, weights, capacities)
