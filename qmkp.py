from typing import Iterable, Any

import numpy as np

from util import is_binary


def total_profit_qmkp(profits: np.array, assignments: np.array):
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
                  weights: Iterable[float]):
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
    """
    num_objects = len(weights)
    _sel_objects = np.zeros(num_objects)
    _sel_objects[sel_objects] = 1
    _sel_objects = np.reshape(_sel_objects, (num_objects, 1))
    sel_objects_matrix = np.tile(_sel_objects, (1, num_objects))
    np.fill_diagonal(sel_objects_matrix, 1)
    contributions = np.diag(profits @ sel_objects_matrix)
    densities = contributions/weights
    return densities

def constructive_procedure(capacities: Iterable[float],
                           weights: Iterable[float],
                           profits: np.array):
    """Algorithm 1

    """
    capacities = np.array(capacities)

    # 1. Initialization
    num_items = len(weights)
    num_ks = len(capacities)
    j_prime = list(range(num_items))
    idx_c_bar = np.argsort(capacities)[::-1]
    c_bar = capacities[idx_c_bar]
    densities = value_density(profits, j_prime, weights)
    idx_sort_objects = np.argsort(densities)[::-1]

    # 2. Iterative Step
    solution = np.zeros((num_items, num_ks))
    #idx_curr_object = 0
    #while len(j_prime) > 0:
    for idx_curr_object in idx_sort_objects:
        _weight_l = weights[idx_curr_object]
        _ks_w_space = np.where(c_bar >= _weight_l)[0]
        if len(_ks_w_space) >= 1:
            _ks_bar = _ks_w_space[0]
            _real_ks = idx_c_bar[_ks_bar]
            solution[idx_curr_object, _real_ks] = 1
            c_bar[_ks_bar] = c_bar[_ks_bar] - _weight_l
        else:
            solution[idx_curr_object, :] = 0
        j_prime.remove(idx_curr_object)
    assert len(j_prime) == 0
    return solution

def fcs_procedure():
    return

def efcs_procedure():
    """
    """
    # 1. Initialization
    s_opt = fcs_procedure()

    #2. Iterative Step
    while True:
        pass

    return


def solve_qmkp(capacities: Iterable[float], weights: Iterable[float],
               profits: np.array):
    """
    (Based on Algorithm 4 of (Aider, Gacem, Hifi, 2022).)

    Parameters
    ----------
    capacities : list of int
        Capacities of the knapsacks. The number of knapsacks :math:`M` is
        determined as `M=len(c)`.

    weights : list
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    profits : array of size :math:`N \\times N`
        Symmetric matrix that contains the (joint) profit values :math:`p_{ij}`

    Returns
    -------
    x_opt : 
        Found solution to the QMKP
    """
    # 1. Initialization Step (EFCS)
    s_opt = efcs_procedure() #TODO

    # 2. Iterative Steps
    while True:
        s_prime = efcs_procedure()
    pass

def fcs_procedure():
    return

def efcs_procedure():
    """
    """
    # 1. Initialization
    s_opt = fcs_procedure()

    #2. Iterative Step
    while True:
        pass

    return

def solve_qmkp(capacities: Iterable[float], weights: Iterable[float],
               profits: np.array):
    """

    (Based on Algorithm 4 of (Aider, Gacem, Hifi, 2022).)

    Parameters
    ----------
    capacities : list of int
        Capacities of the knapsacks. The number of knapsacks :math:`M` is
        determined as `M=len(c)`.

    weights : list
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    profits : array of size :math:`N \\times N`
        Symmetric matrix that contains the (joint) profit values :math:`p_{ij}`

    Returns
    -------
    x_opt : 
        Found solution to the QMKP
    """
    # 1. Initialization Step (EFCS)
    s_opt = efcs_procedure() #TODO

    # 2. Iterative Steps
    while True:
        s_prime = efcs_procedure()
    return


if __name__ == "__main__":
    capacities = [1, .3, 2]
    weights = [1, 4, 2.2]
    profits = np.array([[0, 2, 1], [2, 0, .2], [1, .2, 0]])
    solve_qmkp(capacities, weights, profits)
