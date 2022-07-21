from typing import Iterable, Any, Union, Callable, Optional, Tuple

import numpy as np

from . import checks

class QMKProblem:
    def __init__(self, capacities: Iterable[float], weights: Iterable[float],
                 profits: Union[np.array, Iterable[Iterable]],
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
        if algorithm is None:
            algorithm = self.algorithm
        if args is None:
            args = self.args
        assignments = algorithm(self.profits, self.weights, self.capacities,
                                *args)

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
    if not is_binary(assignments):
        raise ValueError("The assignments matrix needs to be binary.")
    _profit_matrix = assignments.T @ profits @ assignments
    _double_main_diag = assignments.T @ np.diag(profits)
    ks_profits = _double_main_diag + np.diag(_profit_matrix)
    return np.sum(ks_profits)/2

def value_density(profits: np.array, sel_objects: Iterable[float],
                  weights: Iterable[float], reduced_output: bool = False):
    """
    This function will always add object :math:`i` to the selected objects for
    the value of object :math:`i`.

    Parameters
    ----------
    profits : array (N x N)
        Symmetric matrix containing the profits :math:`p_{ij}`

    sel_objects : list
        Set of selected objects

    weights : list
        Weights of the objects

    reduced_output: bool, optional
        If set to `True` only the value density values of the selected objects
        are returned.
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
