from typing import Iterable, Any, Union, Callable, Optional, Tuple

import numpy as np

from . import checks

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
        self.weights = weights
        self.capacities = capacities

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

def value_density(profits: np.array,
                  weights: Iterable[float],
                  sel_objects: Iterable[float],
                  reduced_output: bool = False) -> Iterable[float]:
    """
    Calculate the value density given a set of selected objects.

    Note that this function will always add object :math:`i` to the selected
    objects for the value of object :math:`i`.

    Parameters
    ----------
    profits : array (N x N)
        Symmetric matrix containing the profits :math:`p_{ij}`

    weights : list
        Weights of the objects

    sel_objects : list
        Set of selected objects

    reduced_output : bool, optional
        If set to `True` only the value density values of the selected objects
        are returned.

    Returns
    -------
    densities : list
        List that contains the value densities of the objects. The length is
        equal to :math:`N`, if `reduced_output` is `False`.
        If `reduced_output` is `True`, the return has length
        `len(densities)==len(sel_objects)`.
    """

    num_objects = len(weights)
    _sel_objects = np.zeros(num_objects)
    _sel_objects[sel_objects] = 1
    _sel_objects = np.reshape(_sel_objects, (num_objects, 1))
    sel_objects_matrix = np.tile(_sel_objects, (1, num_objects))
    np.fill_diagonal(sel_objects_matrix, 1)
    contributions = np.diag(profits @ sel_objects_matrix)
    densities = contributions/weights
    if reduced_output:
        densities = densities[sel_objects]
    return densities
