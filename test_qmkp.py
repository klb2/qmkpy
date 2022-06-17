import numpy as np
import pytest

from qmkp import value_density, total_profit_qmkp, constructive_procedure


def test_value_density():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    sel_objects = [1, 3]
    expected = np.array([5, 6/2, 12/3, 8/4])
    vd = value_density(profits, sel_objects, weights)
    assert np.all(expected == vd)

def test_value_density_reduced_output():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 4]
    sel_objects = [1, 3]
    vd = value_density(profits, sel_objects, weights, reduced_output=True)
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

def test_cp():
    profits = np.array([[1, 1, 2, 3],
                        [1, 1, 4, 5],
                        [2, 4, 2, 6],
                        [3, 5, 6, 3]])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    solution = constructive_procedure(capacities, weights, profits)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert np.all(np.shape(solution) == (len(weights), len(capacities))) and total_profit > 0

def test_cp_large():
    num_elements = 20
    num_knapsacks = 5
    profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution = constructive_procedure(capacities, weights, profits)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert total_profit > 0

def test_cp_with_starting():
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
    solution = constructive_procedure(capacities, weights, profits,
                                      starting_assignment=starting_assignment)
    print(solution)
    total_profit = total_profit_qmkp(profits, solution)
    print(total_profit)
    assert np.all(np.shape(solution) == (len(weights), len(capacities))) and total_profit > 0
