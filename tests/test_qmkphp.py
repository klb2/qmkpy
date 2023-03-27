# import tempfile
import os.path
import pathlib

import numpy as np
import pytest

from qmkpy import QMKProblem, QMKProblemHP
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
