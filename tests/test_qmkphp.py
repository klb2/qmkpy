# import tempfile
import os.path
import pathlib

import numpy as np
import pytest

from qmkpy import QMKProblem, QMKProblemHP, total_profit_qmkp
from qmkpy.algorithms import constructive_procedure, fcs_procedure
from qmkpy import checks


SAVE_LOAD_STRATEGIES = ("numpy", "pickle", "txt", "json")

def test_qmkphp_creation():
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 0, 4, 1], [2, 3, 5, 6], [0, 2, 3, 4], [1, 2, 3, 4]])
    profits = np.array([profit1, profit2])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    problem = QMKProblemHP(profits, weights, capacities)
    assert (isinstance(problem, QMKProblemHP) and
            np.all(problem.profits == profits) and
            np.all(problem.weights == weights) and
            np.all(problem.capacities == capacities))

def test_qmkphp_creation_from_qmkp():
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 0, 4, 1], [2, 3, 5, 6], [0, 2, 3, 4], [1, 2, 3, 4]])
    profits = np.array([profit1, profit2])
    weights = [1, 2, 3, 3]
    capacities = [5, 5, 3]
    problem = QMKProblem(profits, weights, capacities)
    assert (isinstance(problem, QMKProblemHP) and
            np.all(problem.profits == profits) and
            np.all(problem.weights == weights) and
            np.all(problem.capacities == capacities))


def test_profit_homog_from_heterog():
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit3 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profits = np.array([profit1, profit2, profit3])
    assignments = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    expected = 14  # KS1: 1+2+4, KS2: 0, KS3: 1+3+3
    _objective = total_profit_qmkp(profits, assignments)
    print(_objective)
    assert expected == _objective

def test_profit_hetero():
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 2, 4, 6], [2, 2, 8, 10], [4, 8, 4, 12], [6, 10, 12, 6]])
    profit3 = np.array([[2, 2, 3, 4], [2, 2, 5, 6], [3, 5, 3, 7], [4, 6, 7, 4]])
    profits = np.array([profit1, profit2, profit3])
    assignments = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    expected = 17  # KS1: 1+2+4, KS2: 0, KS3: 2+4+4
    _objective = total_profit_qmkp(profits, assignments)
    print(_objective)
    assert expected == _objective

def test_profit_hetero2():
    profit1 = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    profit2 = np.array([[2, 2, 4, 6], [2, 2, 8, 10], [4, 8, 4, 12], [6, 10, 12, 6]])
    profit3 = np.array([[2, 2, 3, 4], [2, 2, 5, 6], [3, 5, 3, 7], [4, 6, 7, 4]])
    profits = np.array([profit1, profit2, profit3])
    assignments = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    expected = 13  # KS1: 1+2+4, KS2: 2, KS3: 4
    _objective = total_profit_qmkp(profits, assignments)
    print(_objective)
    assert expected == _objective


def test_profit_fail():
    profits = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [2, 4, 2, 6], [3, 5, 6, 3]])
    assignments = np.array([[0, 0, 1], [2, 0, 0], [-1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        total_profit_qmkp(profits, assignments)
