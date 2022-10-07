"""
General definitions of the quadratic multiple knapsack problem.

This module contains the basic implementation of the quadratic multiple
knapsack problem (QMKP).
In particular, this includes the base class :class:`QMKProblem`.
"""

import os
from typing import Iterable, Union, Callable, Optional, Tuple, NoReturn

import numpy as np

from . import checks
from . import io


class QMKProblem:
    """Base class to represent a quadratic multiple knapsack problem.

    This class defines a standard QMKP with :math:`N` items and :math:`K`
    knapsacks.

    Attributes
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`. The profit of the single items
        :math:`p_i` corresponds to the main diagonal elements, i.e.,
        :math:`p_i = p_{ii}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of float
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    algorithm : Callable, optional
        Function that is used to solve the QMKP. It needs to follow the
        argument order ``algorithm(profits, weights, capacities, ...)``.

    args : tuple, optional
        Optional tuple of additional arguments that are passed to
        :attr:`algorithm`.

    assignments : np.array, optional
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
        This attribute is overwritten when calling :meth:`.solve()`.

    name : str, optional
        Optional name of the problem instance
    """

    def __init__(
        self,
        profits: Union[np.array, Iterable[Iterable]],
        weights: Iterable[float],
        capacities: Iterable[float],
        algorithm: Optional[Callable] = None,
        args: Optional[tuple] = None,
        assignments: Optional[np.array] = None,
        name: Optional[str] = None,
    ):
        profits = np.array(profits)
        checks.check_dimensions(profits, weights)
        self.profits = profits
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)

        self.profits.setflags(write=False)
        self.weights.setflags(write=False)
        self.capacities.setflags(write=False)

        self.algorithm = algorithm
        self.args = args

        if assignments is None:
            self.assignments = np.zeros((len(self.weights), len(self.capacities)))

        self.name = name

    def __eq__(self, other):
        if not isinstance(other, QMKProblem):
            return NotImplemented
        _eq_profits = np.array_equal(self.profits, other.profits)
        _eq_weights = np.array_equal(self.weights, other.weights)
        _eq_capacities = np.array_equal(self.capacities, other.capacities)
        return _eq_profits and _eq_weights and _eq_capacities

    def __str__(self):
        if self.name is not None:
            _print = str(self.name)
        else:
            num_ks = len(self.capacities)
            num_items = len(self.weights)
            _print = f"QMKProblem({num_items:d}, {num_ks:d})"
        return _print

    def solve(
        self,
        algorithm: Optional[Callable] = None,
        args: Optional[tuple] = None,
    ) -> Tuple[np.array, float]:
        """Solve the QMKP

        Solve the QMKP using ``algorithm``. This function both returns the
        optimal assignment and the total resulting profit.
        This method also automatically sets the solution to the object's
        attribute :attr:`.assignments`.


        Parameters
        ----------
        algorithm : Callable, optional
            Function that is used to solve the QMKP. It needs to follow the
            argument order ``algorithm(profits, weights, capacities, ...)``.
            If it is ``None``, the object attribute :attr:`.algorithm` is used.

        args : tuple, optional
            Optional tuple of additional arguments that are passed to
            ``algorithm``. If it is ``None``, the object attribute
            :attr:`.args` is used.


        Returns
        -------
        assignments : np.array
            Binary matrix of size :math:`N\\times K` which represents the final
            assignments of items to knapsacks. If :math:`a_{ij}=1`, element
            :math:`i` is assigned to knapsack :math:`j`.

        total_profit : float
            Final total profit for the found solution.
        """

        if algorithm is None:
            algorithm = self.algorithm
        if args is None:
            args = self.args
        if args is None:
            args = ()
        assignments = algorithm(self.profits, self.weights, self.capacities, *args)
        profit = total_profit_qmkp(self.profits, assignments)

        self.assignments = assignments
        return assignments, profit

    def save(
        self, fname: Union[str, bytes, os.PathLike], strategy: str = "numpy"
    ) -> NoReturn:
        """Save the QMKP instance

        Save the profits, weights, and capacities of the problem. There exist
        different strategies that are explained in the :attr:`strategy`
        parameter.


        See Also
        --------
        :meth:`.load()`
            For loading a saved model.


        Parameters
        ----------
        fname : str or PathLike
            Filepath of the model to be saved at

        strategy : str
            Strategy that is used to store the model. Valid choices are
            (case-insensitive):

            - ``numpy``: Save the individual arrays of the model using the
              :func:`np.savez_compressed` function. See also
              :meth:`qmkpy.io.save_problem_numpy()`.
            - ``pickle``: Save the whole object using Pythons :mod:`pickle`
              module. See also :meth:`qmkpy.io.save_problem_pickle()`.
            - ``txt``: Save the arrays of the model using the text-based format
              established by Billionnet and Soutif. See also
              :meth:`qmkpy.io.save_problem_txt()`.
            - ``json``: Save the arrays of the model using the JSON format.

        Returns
        -------
        None
        """

        strategy = strategy.lower()
        if strategy == "numpy":
            io.save_problem_numpy(fname, self)
        elif strategy == "pickle":
            io.save_problem_pickle(fname, self)
        elif strategy == "txt":
            io.save_problem_txt(fname, self)
        elif strategy == "json":
            io.save_problem_json(fname, self)
        else:
            raise NotImplementedError("The strategy '%s' is not implemented.", strategy)

    @classmethod
    def load(cls, fname: str, strategy: str = "numpy"):
        """Load a QMKProblem instance

        This functions allows loading a previously saved QMKProblem instance.
        The :meth:`.save()` method provides a way of saving a problem.


        See Also
        --------
        :meth:`.save()`
            Method to save a QMKProblem instance which can then be loaded.


        Parameters
        ----------
        fname : str
            Filepath of the saved model

        strategy : str
            Strategy that is used to store the model. Valid choices are
            (case-insensitive):

            - ``numpy``: Save the individual arrays of the model using the
              :meth:`np.savez_compressed` function.
            - ``pickle``: Save the whole object using Pythons :mod:`pickle`
              module
            - ``txt``: Save the arrays of the model using the text-based format
              established by Billionnet and Soutif.
            - ``json``: Save the arrays of the model using the JSON format.

        Returns
        -------
        problem : QMKProblem
            Loaded problem instance
        """

        strategy = strategy.lower()
        if strategy == "numpy":
            problem = io.load_problem_numpy(fname)
        elif strategy == "pickle":
            problem = io.load_problem_pickle(fname)
        elif strategy == "txt":
            problem = io.load_problem_txt(fname)
        elif strategy == "json":
            problem = io.load_problem_json(fname)
        else:
            raise NotImplementedError("The strategy '%s' is not implemented.", strategy)
        return problem


def total_profit_qmkp(profits: np.array, assignments: np.array) -> float:
    """Calculate the total profit for given assignments.

    This function calculates the total profit of a QMKP for a given profit
    matrix :math:`P` and assignments :math:`\\mathcal{A}` as

    .. math:: \\sum_{u=1}^{K}\\left(\\sum_{i\\in\\mathcal{A}_u} p_{i} + \\sum_{\\substack{j\\in\\mathcal{A}_u\\\\j\\neq i}} p_{ij}\\right)

    where :math:`\\mathcal{A}_{u}` is the set of items that are assigned to
    knapsack :math:`u`.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`. The profit of the single items
        :math:`p_i` corresponds to the main diagonal elements, i.e.,
        :math:`p_i = p_{ii}`.

    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.

    Returns
    -------
    float
        Value of the total profit
    """

    if not checks.is_binary(assignments):
        raise ValueError("The assignments matrix needs to be binary.")
    _profit_matrix = assignments.T @ profits @ assignments
    _double_main_diag = assignments.T @ np.diag(profits)
    ks_profits = _double_main_diag + np.diag(_profit_matrix)
    return np.sum(ks_profits) / 2
