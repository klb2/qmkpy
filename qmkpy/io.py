"""
Input/Output functions.

This module contains functions to save and load QMKP instances.
"""

import os
from typing import Union, Optional
import pickle
import json

import numpy as np

from . import qmkp


def save_problem_numpy(fname: Union[str, bytes, os.PathLike], problem):
    """Save a QMKProblem using Numpys npz format

    Save a QMKProblem instance using the compressed npz format. This only saves
    the :attr:`problem.profits`, :attr:`problem.weights`, and
    :attr:`problem.capacities` arrays.


    See Also
    --------
    :meth:`.load_problem_numpy()`
        For loading a saved model.

    :meth:`numpy.savez_compressed()`
        For details on the ``.npz`` format.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : qmkpy.QMKProblem
        Problem instance to be saved


    Returns
    -------
    None
    """

    np.savez_compressed(
        fname,
        profits=problem.profits,
        weights=problem.weights,
        capacities=problem.capacities,
    )


def load_problem_numpy(fname: str):
    """Load a previously stored QMKProblem instance from the Numpy format

    This function allows loading a QMKProblem from a compressed ``.npz`` file,
    which was created by the :meth:`qmkpy.io.save_problem_numpy()` method.

    See Also
    --------
    :meth:`qmkpy.io.save_problem_numpy()`
        For saving a model in the Numpy format.

    :meth:`numpy.load()`
        For details on loading the ``.npz`` format.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the saved model


    Returns
    -------
    problem : qmkpy.QMKProblem
        Loaded problem instance
    """

    _ext = os.path.splitext(fname)[1]
    if not _ext == ".npz":
        fname = fname + ".npz"
    _arrays = np.load(fname)
    problem = qmkp.QMKProblem(**_arrays)
    return problem


def save_problem_pickle(fname: Union[str, bytes, os.PathLike], problem):
    """Save a QMKProblem using the Python Pickle format

    Save a QMKProblem object using the Python pickle library. By this, the
    whole object is stored in a binary format.

    See Also
    --------
    :meth:`qmkpy.io.load_problem_pickle()`
        For loading a saved model.

    :func:`pickle.dump()`
        For details on the underlying pickling function.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : qmkpy.QMKProblem
        Problem instance to be saved


    Returns
    -------
    None
    """

    with open(fname, "wb") as out_file:
        pickle.dump(problem, out_file)


def load_problem_pickle(fname: Union[str, bytes, os.PathLike]):
    """Load a previously stored QMKProblem instance from the Pickle format

    This function allows loading a QMKProblem object from a Python pickled
    object file.

    **Caution:** All warnings as for the regular :func:`pickle.load` apply!

    See Also
    --------
    :meth:`qmkpy.io.save_problem_pickle()`
        For saving a model in the Pickle format.

    :meth:`pickle.load()`
        For details on loading a pickled object.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the saved model


    Returns
    -------
    problem : qmkpy.QMKProblem
        Loaded problem instance
    """

    with open(fname, "rb") as obj_file:
        problem = pickle.load(obj_file)
    return problem


def save_problem_txt(
    fname: Union[str, bytes, os.PathLike],
    qmkp,
    sep: str = "\t",
    name: Optional[str] = None,
):
    """Save a QMKProblem instance in text form

    Save a QMKProblem instance in text form inspired by the format established
    by Alain Billionnet and Eric Soutif for the regular QKP.
    The original description can be found at
    https://cedric.cnam.fr/~soutif/QKP/format.html.

    The file format is as follows:

      1. The first line provides a name/reference of the problem
      2. The second line specifies the number of items
      3. The third line specifies the number of knapsacks
      4. The fourth line is blank to separate the meta information from the
         rest
      5. The fifth line contains the linear profits (main diagonal elements of
         the profit matrix) separated by ``sep``.
      6. The next lines contain the upper triangular part of the profit matrix
         (i.e., :math:`p_{ij}`).
      7. Blank line separating profits from the rest
      8. Weights :math:`w_{i}` of the items, separated by ``sep``.
      9. Blank line separating weights and capacities
      10. Capacities :math:`c_{u}` of the knapsacks, separated by ``sep``.

    For the example with parameters

    .. math::

        P = \\begin{pmatrix}1 & 2 & 3\\\\ 2 & 4 & 5\\\\ 3 & 5 & 6\\end{pmatrix},
        \\quad
        w = \\begin{pmatrix}10\\\\ 20\\\\ 30\\end{pmatrix},
        \\quad
        c = \\begin{pmatrix}5 \\\\ 8\\\\ 1\\\\ 9\\\\ 2\\end{pmatrix}

    the output-file looks as follows

    .. code-block::

        Name of the Problem
        3
        5

        1   4   6
        2   3
        5

        10  20  30

        5   8   1   9   2


    See Also
    --------
    :meth:`qmkpy.io.load_problem_txt()`
        For loading a saved model.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : qmkpy.QMKProblem
        Problem instance to be saved

    sep : str
        Separator string that is used to separate the numbers in the file.

    name : str ,optional
        Optional name of the problem that is used as the first line of the
        output file. If it is ``None``, it will first be checked whether the
        attribute :attr:`problem.name` is set. If this is also ``None``, the
        name defaults to
        ``qmkp_{num_items:d}_{num_ks:d}_{np.random.randint(0, 1000):03d}``.


    Returns
    -------
    None
    """
    profits = qmkp.profits
    weights = qmkp.weights
    capacities = qmkp.capacities

    num_items = len(weights)
    num_ks = len(capacities)
    if name is None:
        name = f"qmkp_{num_items:d}_{num_ks:d}_{np.random.randint(0, 1000):03d}"

    _blank = ""
    content = [name, f"{num_items:d}", f"{num_ks:d}", _blank]

    _lin_prof = np.diag(profits)
    lin_prof = sep.join(str(x) for x in _lin_prof)
    content.append(lin_prof)

    idx_prof_triu = np.triu_indices(num_items, 1)
    _triu_prof = profits[idx_prof_triu]
    for k in range(num_items - 1):
        _start_idx = int(k * num_items - k * (k + 1) / 2)
        _row = _triu_prof[_start_idx:_start_idx + (num_items - 1 - k)]
        row = sep.join(str(x) for x in _row)
        content.append(row)

    content.append(_blank)
    _weights = sep.join(str(x) for x in weights)
    content.append(_weights)

    content.append(_blank)
    _capac = sep.join(str(x) for x in capacities)
    content.append(_capac)

    content = [x + "\n" for x in content]
    with open(fname, "w") as out_file:
        out_file.writelines(content)


def load_problem_txt(fname: Union[str, bytes, os.PathLike], sep: str = "\t"):
    """Load a previously stored QMKProblem instance from the text format

    This function loads a QMKProblem instance from a text file according to the
    format specified in :meth:`qmkpy.io.save_problem_txt()`.


    See Also
    --------
    :meth:`qmkpy.io.save_problem_txt()`
        For saving a model in the text format.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the saved model

    sep : str
        Separator string that is used to separate the numbers in the file.


    Returns
    -------
    problem : qmkpy.QMKProblem
        Loaded problem instance
    """

    with open(fname, "r") as _pf:
        content = _pf.readlines()
    content = [_c.strip() for _c in content]
    reference = content[0]
    num_items = int(content[1])
    num_ks = int(content[2])

    # Blank Line to separate header and profits
    _blank = content[3]
    assert _blank == ""

    # Reconstruct profit matrix
    start_line_prof = 4
    profits = np.zeros((num_items, num_items))
    lin_prof = np.fromstring(content[start_line_prof], sep=sep)
    assert len(lin_prof) == num_items
    prof_triu = []
    for _row in content[start_line_prof + 1:start_line_prof + num_items]:
        _prof_row = np.fromstring(_row, sep=sep)
        prof_triu = np.concatenate((prof_triu, _prof_row))
    idx_triu = np.triu_indices(num_items, 1)
    profits[idx_triu] = prof_triu
    profits = profits + profits.T
    np.fill_diagonal(profits, lin_prof)

    # Blank Line to separate profits and weights
    _blank = content[start_line_prof + num_items]
    assert _blank == ""

    # Reconstruct weights
    weights = np.fromstring(content[start_line_prof + num_items + 1], sep=sep)
    assert len(weights) == num_items

    # Blank Line to separate weights and capacities
    _blank = content[start_line_prof + num_items + 2]
    assert _blank == ""

    # Reconstruct capacities
    capacities = np.fromstring(content[start_line_prof + num_items + 3], sep=sep)
    assert len(capacities) == num_ks

    problem = qmkp.QMKProblem(profits, weights, capacities, name=reference)
    return problem


def save_problem_json(
    fname: Union[str, bytes, os.PathLike], problem, name: Optional[str] = None
):
    """Save a QMKProblem as a JSON file

    Save a QMKProblem instance using the JavaScript Object Notation (JSON)
    format. This only saves the :attr:`problem.profits`,
    :attr:`problem.weights` and :attr:`problem.capacities` arrays, and the
    :attr:`problem.name` attribute if it is set.


    See Also
    --------
    :meth:`.load_problem_json()`
        For loading a saved model.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : qmkpy.QMKProblem
        Problem instance to be saved

    name : str ,optional
        Optional name of the problem that is used as the first line of the
        output file. If it is ``None``, it will first be checked whether the
        attribute :attr:`problem.name` is set. If this is also ``None``, the
        name defaults to
        ``qmkp_{num_items:d}_{num_ks:d}_{np.random.randint(0, 1000):03d}``.


    Returns
    -------
    None
    """

    num_items = len(problem.weights)
    num_ks = len(problem.capacities)

    if name is None:
        if problem.name is not None:
            name = problem.name
        else:
            name = f"qmkp_{num_items:d}_{num_ks:d}_{np.random.randint(0, 1000):03d}"
    _problem = {
        "name": name,
        "profits": problem.profits.tolist(),
        "weights": problem.weights.tolist(),
        "capacities": problem.capacities.tolist(),
    }

    with open(fname, "w") as out_file:
        json.dump(_problem, out_file, indent=2)


def load_problem_json(fname: str):
    """Load a previously stored QMKProblem instance from the JSON format

    This function allows loading a QMKProblem from a ``.json`` file, which was
    created by the :meth:`qmkpy.io.save_problem_json()` method.

    See Also
    --------
    :meth:`qmkpy.io.save_problem_json()`
        For saving a model in the JSON format.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the saved model


    Returns
    -------
    problem : qmkpy.QMKProblem
        Loaded problem instance
    """

    with open(fname, "r") as json_file:
        _problem = json.load(json_file)
    problem = qmkp.QMKProblem(**_problem)
    return problem
