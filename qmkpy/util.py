"""
Utility functions.

This module contains various utility functions, e.g., the conversion from the
binary assignment matrix to the chromosome form.
"""

from typing import Iterable, Union, Optional

import numpy as np


def value_density(
    profits: np.array,
    weights: Iterable[float],
    assignments: Union[np.array, Iterable[int]],
    reduced_output: bool = False,
) -> Iterable[float]:
    """Calculate the value density given a set of selected objects.

    This function calculates the value density of item :math:`i` for knapsack
    :math:`k` and given assignments :math:`\\mathcal{A}_k` according to

    .. math::

        \\mathrm{vd}_i(\\mathcal{A}_k) =
            \\frac{1}{w_i} \\left(p_i +
                \\sum_{\\substack{j\\in\\mathcal{A}_k\\\\j\\neq i}} p_{ij}
            \\right)

    This value indicates the profit (per item weight) that is gained by adding
    the item :math:`i` to knapsack :math:`k` when the items
    :math:`\\mathcal{A}_k` are already assigned.
    It is adapted from the notion of the value density from (Hiley, Julstrom,
    2006).

    When a full array of assignements is used as input, a full array of value
    densities is returned as

    .. math::
        \\mathrm{vd}(\\mathcal{A}) =
        \\begin{pmatrix}
            \\mathrm{vd}_1(\\mathcal{A}_1) & \\mathrm{vd}_1(\\mathcal{A}_2) & \\cdots & \\mathrm{vd}_1(\\mathcal{A}_K)\\\\
            \\mathrm{vd}_2(\\mathcal{A}_1) & \\mathrm{vd}_2(\\mathcal{A}_2) & \\cdots & \\mathrm{vd}_2(\\mathcal{A}_K)\\\\
            \\vdots & \\cdots & \\ddots & \\vdots \\\\
            \\mathrm{vd}_N(\\mathcal{A}_1) & \\mathrm{vd}_N(\\mathcal{A}_2) & \\cdots & \\mathrm{vd}_N(\\mathcal{A}_K)
        \\end{pmatrix}

    When the parameter ``reduced_output`` is set to ``True`` only a subset with
    the values of the unassigned items is returned.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`.

    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    assignments : np.array or list of int
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
        Alternatively, one can provide a list of the indices of selected items.
        In this case, it is assumed that these items are assigned to all
        knapsacks.

    reduced_output : bool, optional
        If set to ``True`` only the value density values of the unassigned
        objects are returned. Additionally, the indices of the unassigned items
        are returned as a second output.

    Returns
    -------
    densities : np.array
        Array that contains the value densities of the objects. The length is
        equal to :math:`N`, if ``reduced_output`` is ``False``.
        If ``reduced_output`` is ``True``, the return has length
        ``len(densities)==len(unassigned_items)``.
        The number of dimensions is equal to the number of dimensions of
        ``assignments``. Each column corresponds to a knapsack. If only a flat
        array is used as input, a flat array is returned.

    unassigned_items : list
        List of the indices of the unassigned items. This is only returned when
        ``reduced_output`` is set to ``True``.
    """

    num_objects = len(weights)
    _flat = False
    if np.ndim(assignments) == 1:
        _flat = True
        idx_assignments = assignments
        assignments = np.zeros((num_objects, 1))
        assignments[idx_assignments] = 1
    unassigned_items = ~np.any(assignments, axis=1)
    unassigned_items = np.where(unassigned_items)[0]
    contributions = profits @ assignments
    _main_diag = np.diag(profits)
    _main_diag_contrib = np.diag(_main_diag) @ (np.ones_like(assignments) - assignments)
    contributions = contributions + _main_diag_contrib
    densities = contributions / np.reshape(weights, (-1, 1))
    if _flat:
        densities = np.ravel(densities)
    if reduced_output:
        densities = densities[unassigned_items], unassigned_items
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
    chromosome : np.array or list of int
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


def get_unassigned_items(assignments: Union[np.array, Iterable[int]]) -> Iterable[int]:
    """Return the list of unassigned items

    Return the list of unassigned items (as their indices) from a given
    assignment. It can be either in the binary matrix form or in the
    chromosome form.

    Parameters
    ----------
    assignments : np.array or list of int
        Either a binary matrix of size :math:`N\\times K` which represents the
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
        Or assignments in the chromosome form, which is a list of length
        :math:`N` where :math:`c_{i}=k` means that item :math:`i` is assigned
        to knapsack :math:`k`. If the item is not assigned, we set
        :math:`c_{i}=-1`.

    Returns
    -------
    unassigned_items : list of int
        List of the indices of the unassigned items.
    """

    assignments = np.array(assignments)
    if np.ndim(assignments) == 1:
        unassigned_items = np.where(assignments == -1)[0]
    elif np.ndim(assignments) == 2:
        unassigned_items = ~np.any(assignments, axis=1)
        unassigned_items = np.where(unassigned_items)[0]
    else:
        raise NotImplementedError("Only the binary and chromosome form are accepted.")
    return unassigned_items


def get_empty_knapsacks(
    assignments: Union[np.array, Iterable[int]], num_ks: Optional[int] = None
) -> Iterable[int]:
    """Return the list of empty knapsacks

    Return the list of empty knapsacks (as their indices) from a given
    assignment. An empty knapsack is one without any assigned item. The
    assignments can be either in the binary matrix form or in the chromosome
    form. If the chromosome form is used, the total number of knapsacks needs
    to be additionally provided.

    Parameters
    ----------
    assignments : np.array or list of int
        Either a binary matrix of size :math:`N\\times K` which represents the
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
        Or assignments in the chromosome form, which is a list of length
        :math:`N` where :math:`c_{i}=k` means that item :math:`i` is assigned
        to knapsack :math:`k`. If the item is not assigned, we set
        :math:`c_{i}=-1`.

    num_ks : int (optional)
        Total number :math:`K` of knapsacks. This only needs to provided, if
        the assignments are given in the chromosome form.

    Returns
    -------
    empty_ks : list of int
        List of the indices of the empty knapsacks.
    """

    assignments = np.array(assignments)
    if np.ndim(assignments) == 1:
        if not isinstance(num_ks, int):
            raise TypeError(
                "If the assignments are given in the chromosome form, the total number of knapsacks is required."
            )
        if np.max(assignments) >= num_ks:
            raise ValueError(
                "The number of total knapsacks is too small for the given assignment chromosome."
            )
        _all_ks = set(range(num_ks))
        empty_ks = _all_ks.difference(assignments)
        empty_ks = list(empty_ks)
    elif np.ndim(assignments) == 2:
        empty_ks = ~np.any(assignments, axis=0)
        empty_ks = np.where(empty_ks)[0]
    else:
        raise NotImplementedError("Only the binary and chromosome form are accepted.")
    return empty_ks


def get_remaining_capacities(
    weights: Iterable[float],
    capacities: Iterable[float],
    assignments: Union[np.array, Iterable[int]],
):
    """Return the remaining weight capacities of the knapsacks

    Returns the remaining weight capacities of the knapsacks for given
    assignments. The function does *not* raise an error when a weight
    constraint is violated but will return a negative remaining capacity in
    this case.


    Parameters
    ----------
    weights : list of float
        List of weights :math:`w_i` of the :math:`N` items that can be
        assigned.

    capacities : list of float
        Capacities of the knapsacks. The number of knapsacks :math:`K` is
        determined as ``K=len(capacities)``.

    assignments : np.array or list of int
        Either a binary matrix of size :math:`N\\times K` which represents the
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.
        Or assignments in the chromosome form, which is a list of length
        :math:`N` where :math:`c_{i}=k` means that item :math:`i` is assigned
        to knapsack :math:`k`. If the item is not assigned, we set
        :math:`c_{i}=-1`.

    Returns
    -------
    remaining_capacities : list of float
        List of the remaining capacities. Can be negative, if a knapsack is
        overloaded.
    """

    assignments = np.array(assignments)
    if np.ndim(assignments) == 1:
        num_ks = len(capacities)
        assignments = assignment_from_chromosome(assignments, num_ks)

    load = weights @ assignments
    remain_capac = capacities - load
    return remain_capac
