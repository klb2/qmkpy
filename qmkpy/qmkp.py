import os
from typing import Iterable, Any, Union, Callable, Optional, Tuple, NoReturn
import pickle

import numpy as np

from . import checks
from .util import value_density

class QMKProblem:
    """Base class to represent a quadratic multiple knapsack problem.

    This class defines a standard QMKP with :math:`N` items and :math:`K`
    knapsacks.

    Attributes
    ----------
    profits : array of size N x N
        Symmetric matrix that contains the (joint) profit values :math:`p_{ij}`

    weights : list of length N
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of length K
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as `K=len(capacities)`.

    algorithm : Callable, optional
        Function that is used to solve the QMKP. It needs to follow the
        argument order `algorithm(profits, weights, capacities, ...)`.

    args : tuple, optional
        Optional tuple of additional arguments that are passed to `algorithm`.
    """

    def __init__(self,
                 profits: Union[np.array, Iterable[Iterable]],
                 weights: Iterable[float],
                 capacities: Iterable[float], 
                 algorithm: Optional[Callable] = None,
                 args: Optional[tuple] = None):
        profits = np.array(profits)
        checks.check_dimensions(profits, weights)
        self.profits = profits
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)

        self.profits.setflags(write=False)
        self.weights.setflags(write=False)
        self.capacities.setflags(write=False)

        self.assignments = np.zeros((len(self.weights), len(self.capacities)))

        self.algorithm = algorithm
        self.args = args
    
    def solve(self, algorithm: Optional[Callable] = None,
              args: Optional[tuple] = None) -> Tuple[np.array, float]:
        """Solve the QMKP

        Solve the QMKP using `algorithm`. This function both returns the
        optimal assignment and the total resulting profit.

        Parameters
        ----------
        algorithm : Callable, optional
            Function that is used to solve the QMKP. It needs to follow the
            argument order `algorithm(profits, weights, capacities, ...)`.
            If it is `None`, the object attribute `self.algorithm` is used.

        args : tuple, optional
            Optional tuple of additional arguments that are passed to
            `algorithm`. If it is `None`, the object attribute `self.args` is
            used.


        Returns
        -------
        assignments : np.array (N x K)
            Found assignments which are represented by a :math:`N\\times K`
            binary matrix where :math:`a_{ij}=1` means that item :math:`i` is
            assigned to knapsack :math:`j`.

        total_profit : float
            Final total profit for the found solution.
        """

        if algorithm is None:
            algorithm = self.algorithm
        if args is None:
            args = self.args
        if args is None:
            args = ()
        assignments = algorithm(self.profits, self.weights, self.capacities,
                                *args)
        profit = total_profit_qmkp(self.profits, assignments)
        return assignments, profit

    def save(self, fname: Union[str, bytes, os.PathLike],
             strategy: str = "numpy") -> NoReturn:
        """Save the QMKP instance

        Save the profits, weights, and capacities of the problem.

        Parameters
        ----------
        fname : str or PathLike
            Filepath of the model to be saved at

        strategy : str
            Strategy that is used to store the model. Valid choices are
            (case-insensitive):

            - ``numpy``: Save the individual arrays of the model using the
              :meth:`np.savez_compressed` function.
            - ``pickle``: Save the whole object using Pythons :mod:`pickle`
              module

        Returns
        -------
        None
        """

        strategy = strategy.lower()
        if strategy == "numpy":
            np.savez_compressed(fname, profits=self.profits,
                                weights=self.weights,
                                capacities=self.capacities)
        elif strategy == "pickle":
            with open(fname, 'wb') as out_file:
                pickle.dump(self, out_file)
        else:
            raise NotImplementedError("The strategy '%s' is not implemented.",
                                      strategy)

    @classmethod
    def load(cls, fname: str, strategy: str = "numpy"):
        """Load a QMKProblem instance

        TODO

        Parameters
        ----------

        Returns
        -------
        problem : QMKProblem
            Loaded problem instance
        """

        strategy = strategy.lower()
        if strategy == "numpy":
            _ext = os.path.splitext(fname)[1]
            if not _ext == ".npz":
                fname = fname + ".npz"
            _arrays = np.load(fname)
            problem = QMKProblem(**_arrays)
        elif strategy == "pickle":
            with open(fname, 'rb') as obj_file:
                problem = pickle.load(obj_file)
        else:
            raise NotImplementedError("The strategy '%s' is not implemented.",
                                      strategy)
        return problem


def total_profit_qmkp(profits: np.array, assignments: np.array) -> float:
    """
    Parameters
    ----------
    profits : array (N x N)
        Symmetric matrix containing the profits :math:`p_{ij}`

    assignments : array (N x M)
        Matrix with binary elements where column :math:`j` corresponds to the
        assignments of the :math:`N` items to knapsack :math:`j`.
    """

    if not checks.is_binary(assignments):
        raise ValueError("The assignments matrix needs to be binary.")
    _profit_matrix = assignments.T @ profits @ assignments
    _double_main_diag = assignments.T @ np.diag(profits)
    ks_profits = _double_main_diag + np.diag(_profit_matrix)
    return np.sum(ks_profits)/2
