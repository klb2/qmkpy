import os
from typing import Iterable, Any, Union, Callable, Optional, Tuple, NoReturn
import pickle

import numpy as np

from . import qmkp


def save_problem_numpy(fname: Union[str, bytes, os.PathLike],
                       problem):
    """

    See Also
    --------
    :meth:`.load_problem_numpy()`
        For loading a saved model.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : QMKProblem
        Problem instance to be saved


    Returns
    -------
    None
    """
    np.savez_compressed(fname,
                        profits=problem.profits,
                        weights=problem.weights,
                        capacities=problem.capacities)

def load_problem_numpy(fname: str):
    """
    """
    _ext = os.path.splitext(fname)[1]
    if not _ext == ".npz":
        fname = fname + ".npz"
    _arrays = np.load(fname)
    problem = qmkp.QMKProblem(**_arrays)
    return problem

def save_problem_pickle(fname: Union[str, bytes, os.PathLike],
                        problem):
    """

    See Also
    --------
    :meth:`.load_problem_pickle()`
        For loading a saved model.


    Parameters
    ----------
    fname : str or PathLike
        Filepath of the model to be saved at

    problem : QMKProblem
        Problem instance to be saved


    Returns
    -------
    None
    """
    with open(fname, 'wb') as out_file:
        pickle.dump(problem, out_file)

def load_problem_pickle(fname: Union[str, bytes, os.PathLike]):
    """
    """
    with open(fname, 'rb') as obj_file:
        problem = pickle.load(obj_file)
    return problem

def save_problem_txt(fname: Union[str, bytes, os.PathLike],
                     qmkp, sep: str = "\t", name: Optional[str] = None):
    """Save a QMKProblem instance in text form

    Save a QMKProblem instance in text form inspired by the format established
    by Alain Billionnet and Eric Soutif for the regular QKP.
    The original description can be found at
    https://cedric.cnam.fr/~soutif/QKP/format.html.

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
    for k in range(num_items-1):
        _start_idx = int(k*num_items - k*(k+1)/2)
        _row = _triu_prof[_start_idx:_start_idx+(num_items-1-k)]
        row = sep.join(str(x) for x in _row)
        content.append(row)

    content.append(_blank)
    _weights = sep.join(str(x) for x in weights)
    content.append(_weights)

    content.append(_blank)
    _capac = sep.join(str(x) for x in capacities)
    content.append(_capac)

    content = [x + "\n" for x in content]
    with open(fname, 'w') as out_file:
        out_file.writelines(content)
    

def load_problem_txt(fname: Union[str, bytes, os.PathLike], sep:str = "\t"):
    """Load a QMKProblem instance from text form

    ...
    """
    with open(fname, 'r') as _pf:
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
    for _row in content[start_line_prof+1:start_line_prof+num_items]:
        _prof_row = np.fromstring(_row, sep=sep)
        prof_triu = np.concatenate((prof_triu, _prof_row))
    idx_triu = np.triu_indices(num_items, 1)
    profits[idx_triu] = prof_triu
    profits = profits + profits.T
    np.fill_diagonal(profits, lin_prof)

    # Blank Line to separate profits and weights
    _blank = content[start_line_prof+num_items]
    assert _blank == ""

    # Reconstruct weights
    weights = np.fromstring(content[start_line_prof+num_items+1], sep=sep)
    assert len(weights) == num_items

    # Blank Line to separate weights and capacities
    _blank = content[start_line_prof+num_items+2]
    assert _blank == ""

    # Reconstruct capacities
    capacities = np.fromstring(content[start_line_prof+num_items+3], sep=sep)
    assert len(capacities) == num_ks

    problem = qmkp.QMKProblem(profits, weights, capacities)
    return problem
