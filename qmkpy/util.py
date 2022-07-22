from typing import Iterable, Any, Union, Callable, Optional, Tuple

import numpy as np

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

def chromosome_from_assignment(assignments: np.array) -> Iterable[int]:
    """Return the chromosome from an assignment matrix

    TODO

    Parameters
    ----------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
    
    Returns
    -------
    chromosome : np.array
        Chromosome version of ``assignments``, which is a list of length
        :math:`N` where :math:`c_{i}=k` means that item :math:`i` is assigned
        to knapsack :math:`k`. If the item is not assigned, we set
        :math:`c_{i}=-1`.
    """

    chromosome = -np.ones(len(assignments))
    _assigned_items = np.argwhere(assignments == 1)
    chromosome[_assigned_items[:, 0]] = _assigned_items[:, 1]
    return chromosome
