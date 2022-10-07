"""
Solution algorithms for the QMKP.

This module contains all the algorithms that can be used to solve the quadratic
multiple knapsack problem (QMKP).
"""

from typing import Iterable, Optional

import numpy as np

from .qmkp import total_profit_qmkp
from .util import value_density
from .checks import is_binary


def constructive_procedure(
    profits: np.array,
    weights: Iterable[float],
    capacities: Iterable[float],
    starting_assignment: np.array = None,
) -> np.array:
    """Constructive procedure that completes a starting assignment

    This constructive procedure is based on Algorithm 1 from [AGH22]_ and the
    greedy heuristic in [HJ06]_. It is a greedy algorithm that completes a
    partial solution of the QMKP.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of float
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    starting_assignments : np.array, optional
        Binary matrix of size :math:`N\\times K` which represents existing
        starting assignments of items to knapsacks. If :math:`a_{ij}=1`,
        element :math:`i` is assigned to knapsack :math:`j`. These assignments
        are not modified and will only be completed.
        If it is `None`, no existing assignment is assumed.


    Returns
    -------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.

    Raises
    ------
    ValueError
        Raises a :class:`ValueError` if the starting assignment is infeasible.
    """

    capacities = np.array(capacities)
    weights = np.array(weights)
    num_items = len(weights)
    num_ks = len(capacities)

    # 1. Initialization
    if starting_assignment is None:
        starting_assignment = np.zeros((num_items, num_ks))
    if not is_binary(starting_assignment):
        raise ValueError("The starting assignment needs to be a binary matrix")
    if not np.all(np.shape(starting_assignment) == (num_items, num_ks)):
        raise ValueError(
            "The shape of the starting assignment needs to be num_items x num_knapsacks"
        )

    start_load = weights @ starting_assignment
    capacities = capacities - start_load

    if np.any(capacities < 0):
        raise ValueError(
            "The starting assignment already violates the weight capacity limit"
        )

    # densities = value_density(profits, weights, j_prime, reduced_output=True)
    dens_v, unassigned = value_density(
        profits, weights, starting_assignment, reduced_output=True
    )
    # idx_sort_objects = np.argsort(densities)[::-1]

    # 2. Iterative Step
    solution = np.copy(starting_assignment)
    while len(unassigned) > 0 and np.min(weights[unassigned]) < np.max(capacities):
        idx_sort_v_flat = np.argsort(dens_v, axis=None)[::-1]
        idx_sort_v = np.unravel_index(idx_sort_v_flat, np.shape(dens_v))
        for idx_el_v, idx_user in zip(*idx_sort_v):
            idx_element = unassigned[idx_el_v]
            if weights[idx_element] <= capacities[idx_user]:
                solution[idx_element, idx_user] = 1
                capacities[idx_user] = capacities[idx_user] - weights[idx_element]
                break
        dens_v, unassigned = value_density(
            profits, weights, solution, reduced_output=True
        )
    # idx_c_bar = np.argsort(capacities)[::-1]
    # c_bar = capacities[idx_c_bar]
    # for _idx_curr_object in idx_sort_objects:
    #    idx_curr_object = j_prime[_idx_curr_object]
    #    _weight_l = weights[idx_curr_object]
    #    _ks_w_space = np.where(c_bar >= _weight_l)[0]
    #    if len(_ks_w_space) >= 1:
    #        _ks_bar = _ks_w_space[0]
    #        _real_ks = idx_c_bar[_ks_bar]
    #        solution[idx_curr_object, _real_ks] = 1
    #        c_bar[_ks_bar] = c_bar[_ks_bar] - _weight_l
    #    else:
    #        solution[idx_curr_object, :] = 0
    # j_prime.remove(idx_curr_object)
    # assert len(j_prime) == 0
    return solution


def fcs_procedure(
    profits: np.array,
    weights: Iterable[float],
    capacities: Iterable[float],
    alpha: Optional[float] = None,
    len_history: int = 50,
) -> np.array:
    """Implementation of the fix and complete solution (FCS) procedure

    This fix and complete solution (FCS) procedure is based on Algorithm 2 from
    [AGH22]_. It is basically a stochastic hill-climber wrapper around the
    constructive procedure :meth:`constructive_procedure` (also see [HJ06]_).


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of float
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    alpha : float, optional
        Float between 0 and 1 that indicates the ratio of assignments that
        should be dropped in an iteration. If not provided, a uniformly random
        value is chosen.

    len_history : int, optional
        Number of consecutive iterations without any improvement before the
        algorithm terminates.

    Returns
    -------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
    """

    capacities = np.array(capacities)

    # 1. Initialization
    current_solution = constructive_procedure(profits, weights, capacities)
    solution_best = np.copy(current_solution)
    if alpha is None:
        alpha = np.random.rand()
    if not 0 < alpha < 1:
        raise ValueError("Alpha needs to be in the interval (0, 1)")
    if len_history < 1:
        raise ValueError("The history length needs to be larger or equal to 1.")
    # profit_history = []
    no_improvement = 0
    while no_improvement < len_history:
        s1 = np.where(np.any(current_solution, axis=1))[0]
        _dropped_items = np.random.choice(s1, size=int(len(s1) * alpha), replace=False)
        start_assign = np.copy(current_solution)
        start_assign[_dropped_items, :] = 0
        s_prime = constructive_procedure(
            profits, weights, capacities, starting_assignment=start_assign
        )
        _profit_best = total_profit_qmkp(profits, solution_best)
        _profit_prime = total_profit_qmkp(profits, s_prime)
        no_improvement = no_improvement + 1
        if _profit_prime > _profit_best:
            solution_best = s_prime
            # print(f"Improvement after {no_improvement} steps")
            no_improvement = 0
        if np.random.rand() > 0.5:
            current_solution = solution_best
    return solution_best


def random_assignment(
    profits: np.array, weights: Iterable[float], capacities: Iterable[float]
) -> np.array:
    """Generate a random (feasible) assignment

    This function generates a random feasible solution to the specified QMKP.
    The algorithm works as follows

        1. Generate a random permutation of the items
        2. For each item :math:`i` do

            1. Determine the possible knapsacks :math:`\\mathcal{K}_i` that
               could support the item
            2. Random and uniformly select a choice from :math:`\\mathcal{K}_i
               \\cup \\{\\text{skip}\\}`.

    This way, a feasible solution is generated without the guarantee that every
    item is assigned (even if it could still be assigned).


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`. The profit of the single items
        :math:`p_i` corresponds to the main diagonal elements, i.e.,
        :math:`p_i = p_{ii}`.

    weights : list
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    Returns
    -------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
    """

    capacities = np.array(capacities)
    num_items = len(weights)
    num_ks = len(capacities)
    assignments = np.zeros((num_items, num_ks))
    for _item in np.random.permutation(range(num_items)):
        avail_ks = np.argwhere(capacities >= weights[_item])
        avail_ks = np.ravel(avail_ks)
        if len(avail_ks) == 0:
            continue
        if np.random.rand() < 1.0 / len(avail_ks):
            continue
        _ks = np.random.choice(avail_ks)
        assignments[_item, _ks] = 1
        capacities[_ks] = capacities[_ks] - weights[_item]
    return assignments


def round_robin(
    profits: np.array,
    weights: Iterable[float],
    capacities: Iterable[float],
    starting_assignment: np.array = None,
    order_ks: Optional[Iterable[int]] = None,
) -> np.array:
    """Simple round-robin algorithm

    This algorithm follows a simple round-robin scheme to assign items to
    knapsacks.
    The knapsacks are iterated in the order provided by ``order_ks``.
    In each round, the current knapsack selects the item with the highest value
    density that still fits in the knapsack.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of float
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    starting_assignments : np.array, optional
        Binary matrix of size :math:`N\\times K` which represents existing
        starting assignments of items to knapsacks. If :math:`a_{ij}=1`,
        element :math:`i` is assigned to knapsack :math:`j`. These assignments
        are not modified and will only be completed.
        If it is `None`, no existing assignment is assumed.

    order_ks : list of int, optional
        Order in which the knapsacks select the items. If none is given, they
        are iterated by index, i.e., ``order_ks = range(num_ks)``.


    Returns
    -------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
    """
    num_ks = len(capacities)
    num_items = len(weights)
    weights = np.array(weights)

    if starting_assignment is None:
        starting_assignment = np.zeros((num_items, num_ks))

    if not is_binary(starting_assignment):
        raise ValueError("The starting assignment needs to be a binary matrix")
    if not np.all(np.shape(starting_assignment) == (num_items, num_ks)):
        raise ValueError(
            "The shape of the starting assignment needs to be num_items x num_knapsacks"
        )

    if order_ks is None or len(order_ks) == 0:
        order_ks = np.arange(num_ks)
    if not set(order_ks).issubset(set(range(num_ks))):
        raise ValueError(
            "The order of the knapsacks must only contain indices of the knapsacks, i.e., every element needs to be an integer from {0, 1, ..., K-1}."
        )

    start_load = weights @ starting_assignment
    remain_capac = capacities - start_load

    if np.any(remain_capac < 0):
        raise ValueError(
            "The starting assignment already violates the weight capacity limit"
        )

    assignments = np.copy(starting_assignment)
    densities, unassigned = value_density(
        profits, weights, starting_assignment, reduced_output=True
    )

    while len(unassigned) > 0 and np.min(weights[unassigned]) < np.max(
        remain_capac[order_ks]
    ):
        for idx_ks in order_ks:
            if not np.any(weights[unassigned] <= remain_capac[idx_ks]):
                continue
            _idx_best_unass_items = np.argsort(densities[:, idx_ks])[::-1]
            _idx_best_items = unassigned[_idx_best_unass_items]
            _best_poss_unass_item = np.argmax(
                weights[_idx_best_items] <= remain_capac[idx_ks]
            )
            idx_selected_item = _idx_best_items[_best_poss_unass_item]
            assignments[idx_selected_item, idx_ks] = 1
            remain_capac[idx_ks] -= weights[idx_selected_item]
            densities, unassigned = value_density(
                profits, weights, assignments, reduced_output=True
            )
    return assignments
