import os.path
import filecmp

import numpy as np

from qmkpy import io
from qmkpy import qmkp


PWD = os.path.dirname(__file__)
EX_TXT = os.path.join(PWD, "ex_qmkp.txt")
EX_JSON = os.path.join(PWD, "ex_qmkp.json")


def test_load_txt():
    loaded_problem = io.load_problem_txt(EX_TXT)

    profits = np.array([[1, 5, 6, 7], [5, 2, 8, 9], [6, 8, 3, 10], [7, 9, 10, 4]])
    weights = [10, 20, 30, 40]
    capacities = [5, 8, 1, 9, 2]
    name = "Reference Problem"

    assert (
        np.all(loaded_problem.profits == profits)
        and np.all(loaded_problem.weights == weights)
        and np.all(loaded_problem.capacities == capacities)
        and loaded_problem.name == name
    )


def test_save_txt(tmp_path):
    profits = np.array([[1, 5, 6, 7], [5, 2, 8, 9], [6, 8, 3, 10], [7, 9, 10, 4]])
    weights = [10, 20, 30, 40]
    capacities = [5, 8, 1, 9, 2]

    problem = qmkp.QMKProblem(profits, weights, capacities)
    outfile = os.path.join(tmp_path, "save.txt")
    io.save_problem_txt(outfile, problem, name="Reference Problem")
    assert filecmp.cmp(outfile, EX_TXT, shallow=False)


def test_load_json():
    loaded_problem = io.load_problem_json(EX_JSON)

    profits = np.array([[1, 5, 6, 7], [5, 2, 8, 9], [6, 8, 3, 10], [7, 9, 10, 4]])
    weights = [10, 20, 30, 40]
    capacities = [5, 8, 1, 9, 2]
    name = "qmkp_10_3_679"

    assert (
        np.all(loaded_problem.profits == profits)
        and np.all(loaded_problem.weights == weights)
        and np.all(loaded_problem.capacities == capacities)
        and loaded_problem.name == name
    )


def test_save_json(tmp_path):
    profits = np.array([[1, 5, 6, 7], [5, 2, 8, 9], [6, 8, 3, 10], [7, 9, 10, 4]])
    weights = [10, 20, 30, 40]
    capacities = [5, 8, 1, 9, 2]
    name = "qmkp_10_3_679"

    problem = qmkp.QMKProblem(profits, weights, capacities, name=name)
    outfile = os.path.join(tmp_path, "save.json")
    io.save_problem_json(outfile, problem)
    assert filecmp.cmp(outfile, EX_JSON, shallow=False)
