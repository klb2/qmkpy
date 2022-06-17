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

def constructive_procedure(capacities: Iterable[float],
                           weights: Iterable[float],
                           profits: np.array,
                           starting_assignment: np.array = None):
    """Algorithm 1

    """
    capacities = np.array(capacities)
    num_items = len(weights)
    num_ks = len(capacities)

    # 1. Initialization
    if starting_assignment is None:
        starting_assignment = np.zeros((num_items, num_ks))
    if not is_binary(starting_assignment):
        raise ValueError("The starting assignment needs to be a binary matrix")
    if not np.all(np.shape(starting_assignment) == (num_items, num_ks)):
        raise ValueError("The shape of the starting assignment needs to be num_items x num_knapsacks")

    _unassigned_items = ~np.any(starting_assignment, axis=1)
    j_prime = np.where(_unassigned_items)[0]
    #j_prime = list(range(num_items))

    start_load = weights @ starting_assignment
    capacities = capacities - start_load
    idx_c_bar = np.argsort(capacities)[::-1]
    c_bar = capacities[idx_c_bar]

    densities = value_density(profits, j_prime, weights, reduced_output=True)
    idx_sort_objects = np.argsort(densities)[::-1]

    # 2. Iterative Step
    #solution = np.zeros((num_items, num_ks))
    solution = np.copy(starting_assignment)
    for _idx_curr_object in idx_sort_objects:
        idx_curr_object = j_prime[_idx_curr_object]
        _weight_l = weights[idx_curr_object]
        _ks_w_space = np.where(c_bar >= _weight_l)[0]
        if len(_ks_w_space) >= 1:
            _ks_bar = _ks_w_space[0]
            _real_ks = idx_c_bar[_ks_bar]
            solution[idx_curr_object, _real_ks] = 1
            c_bar[_ks_bar] = c_bar[_ks_bar] - _weight_l
        else:
            solution[idx_curr_object, :] = 0
        #j_prime.remove(idx_curr_object)
    #assert len(j_prime) == 0
    return solution

def fcs_procedure(capacities: Iterable[float],
                  weights: Iterable[float],
                  profits: np.array):
    capacities = np.array(capacities)

    # 1. Initialization
    num_items = len(weights)
    num_ks = len(capacities)
    solution_best = constructive_procedure(capacities, weights, profits)
    alpha = np.random.rand()
    while True:
        pass
    return solution_best

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
    solution = constructive_procedure(capacities, weights, profits)
    #solve_qmkp(capacities, weights, profits)
