from typing import Iterable, Any, Union, Callable, Optional, Tuple

import numpy as np

def value_density(profits: np.array,
                  weights: Iterable[float],
                  sel_objects: Iterable[float],
                  reduced_output: bool = False) -> Iterable[float]:
    """Calculate the value density given a set of selected objects.

    Note that this function will always add object :math:`i` to the selected
    objects for the value of object :math:`i`.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    sel_objects : list
        Set of selected objects

    reduced_output : bool, optional
        If set to ``True`` only the value density values of the selected
        objects are returned.

    Returns
    -------
    densities : list
        List that contains the value densities of the objects. The length is
        equal to :math:`N`, if ``reduced_output`` is ``False``.
        If ``reduced_output`` is ``True``, the return has length
        ``len(densities)==len(sel_objects)``.
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

    The chromosome version of ``assignments`` is a list of length :math:`N`
    where :math:`c_{i}=k` means that item :math:`i` is assigned
    to knapsack :math:`k`. If the item is not assigned, we set
    :math:`c_{i}=-1`.

    Example
    -------
    Assume that we have 4 items and 3 knapsacks. Let Items 1 and 4 be assigned
    to Knapsack 1, Item 2 is assigned to Knapsack 3 and Item 3 is not assigned.
    In the binary representation, this is
        
    .. math::
        
        A = 
        \\begin{pmatrix} 1 & 0 & 0\\\\
                         0 & 0 & 1\\\\
                         0 & 0 & 0\\\\
                         1 & 0 & 0
        \\end{pmatrix}

    The corresponding chromosome is

    .. math::

        C(A) = \\begin{pmatrix}1 & 3 & 0 & 1\\end{pmatrix}

    However, in the 0-index based representation in Python, this function will
    return 

    .. code-block:: python

        chromosome_from_assignment(A) = [0, 2, -1, 0]
    
    as the chromosome.


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
    return chromosome.astype(int)

def assignment_from_chromosome(chromosome: Iterable[int], num_ks: int) -> np.array:
    """Return the assignment matrix from a chromosome

    Return the binary assignment matrix that corresponds to the chromosome.
    For more details about the connection between assignment matrix and
    chromosome check :func:`chromosome_from_assignment`.

    See Also
    --------
    :func:`chromosome_from_assignment`
        For more details on the connection between assignment matrix and
        chromosome.

    Parameters
    ----------
    chromosome : np.array
        Chromosome version of ``assignments``, which is a list of length
        :math:`N` where :math:`c_{i}=k` means that item :math:`i` is assigned
        to knapsack :math:`k`. If the item is not assigned, we set
        :math:`c_{i}=-1`.

    num_ks : int
        Number of knapsacks :math:`K`.

    
    Returns
    -------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
    """
    chromosome = np.array(chromosome, dtype=int)
    num_items = len(chromosome)
    assignments = np.zeros((num_items, num_ks))
    _assigned_items = np.argwhere(chromosome >= 0)
    assignments[_assigned_items, chromosome[_assigned_items]] = 1
    return assignments
