from typing import Iterable, Any, Optional

import numpy as np

from .qmkp import value_density, total_profit_qmkp
from .checks import is_binary


def constructive_procedure(profits: np.array,
                           weights: Iterable[float],
                           capacities: Iterable[float],
                           starting_assignment: np.array = None) -> np.array:
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

    densities = value_density(profits, weights, j_prime, reduced_output=True)
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

def fcs_procedure(profits: np.array,
                  weights: Iterable[float],
                  capacities: Iterable[float],
                  alpha: Optional[float] = None,
                  len_history: int = 50) -> np.array:
    """
    (Based on Algorithm 2 of (Aider, Gacem, Hifi, 2022).)

    Parameters
    ----------
    profits : array of size N x N
        Symmetric matrix that contains the (joint) profit values :math:`p_{ij}`

    weights : list
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of int
        Capacities of the knapsacks. The number of knapsacks :math:`M` is
        determined as `M=len(c)`.

    alpha : float, optional
        Float between 0 and 1 that indicates the ratio of assignments that
        should be dropped in an iteration. If not provided, a uniformly random
        value is chosen.

    len_history : int, optional
        Number of consecutive iterations without any improvement before the
        algorithm terminates.


    Returns
    -------
    x_opt : array of size N x K
        Found solution to the QMKP
    """
    capacities = np.array(capacities)

    # 1. Initialization
    num_items = len(weights)
    num_ks = len(capacities)
    current_solution = constructive_procedure(profits, weights, capacities)
    solution_best = np.copy(current_solution)
    if alpha is None:
        alpha = np.random.rand()
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha needs to be in the interval [0, 1]")
    #profit_history = []
    no_improvement = 0
    while no_improvement < len_history:
        s1 = np.where(np.any(current_solution, axis=1))[0]
        _dropped_items = np.random.choice(s1, size=int(len(s1)*alpha),
                                          replace=False)
        start_assign = np.copy(current_solution)
        start_assign[_dropped_items, :] = 0
        s_prime = constructive_procedure(profits, weights, capacities,
                                         starting_assignment=start_assign)
        _profit_best = total_profit_qmkp(profits, solution_best)
        _profit_prime = total_profit_qmkp(profits, s_prime)
        no_improvement = no_improvement + 1
        if _profit_prime > _profit_best:
            solution_best = s_prime
            #print(f"Improvement after {no_improvement} steps")
            no_improvement = 0
        if np.random.rand() > 0.5:
            current_solution = solution_best
    return solution_best

def efcs_procedure(capacities: Iterable[float],
                   weights: Iterable[float],
                   profits: np.array):
    """
    """
    capacities = np.array(capacities)

    # 1. Initialization
    num_items = len(weights)
    num_ks = len(capacities)
    current_solution = fcs_procedure(capacities, weights, profits)
    solution_best = np.copy(current_solution)

    tabu_list = set()

    #2. Iterative Step
    no_improvement = 0
    while no_improvement < 50:
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
    solution = constructive_procedure(profits, weights, capacities)
    #solve_qmkp(capacities, weights, profits)
