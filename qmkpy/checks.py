"""
Various checks/verification functions.

This module contains various functions to perform check/verify provided
parameters in the context of the QMKP.
For example, this includes a check whether a provided assignment complies with
the weight/capacity constraints.
"""

from typing import Iterable, NoReturn, Optional

import numpy as np


def check_dimensions(
    profits: np.array, weights: Optional[Iterable[float]] = None
) -> NoReturn:
    """Simple check whether the dimensions of the parameters match.

    This function checks that

    1. The profit matrix is quadratic of size :math:`N`,
    2. The number of items is equal to :math:`N`, i.e., ``len(weights)==N``.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` containing the profits
        :math:`p_{ij}`.

    weights : list of float, optional
        List which contains the weights of the :math:`N` items.

    Raises
    ------
    ValueError
        This function raises a :class:`ValueError`, if there is a mismatch.
    """

    _row_p, _cols_p = np.shape(profits)
    if not _row_p == _cols_p:
        raise ValueError("The profit matrix is not square")
    if weights is not None:
        num_items = len(weights)
        if not num_items == _row_p:
            raise ValueError(
                "The number of items does not match the number of profits."
            )


def is_binary(x: Iterable[float]) -> bool:
    """Check whether a provided array is binary

    This function checks that all elements of the input are either 0 or 1.

    Parameters
    ----------
    x : Iterable
        Array of numbers

    Returns
    -------
    binary : bool
        Returns ``True`` when the array ``x`` is binary and ``False``
        otherwise.
    """

    x = np.array(x)
    return ((x == 0) | (x == 1)).all()


def is_feasible_solution(
    assignments: np.array,
    profits: np.array,
    weights: Iterable[float],
    capacities: Iterable[float],
    raise_error: bool = False,
) -> bool:
    """Check whether a provided assignment is a feasible solution.

    This function performs a formal check whether the provided assignments is a
    feasible solution of the specified QMKProblem.
    This means that the shapes of the arrays match and that no weight capacity
    constraint is violated.


    Parameters
    ----------
    assignments : np.array
        Binary matrix of size :math:`N\\times K` which represents the final
        assignments of items to knapsacks. If :math:`a_{ij}=1`, element
        :math:`i` is assigned to knapsack :math:`j`.

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

    raise_error : bool, optional
        If ``raise_error`` is ``False``, the function returns a ``bool``,
        that states whether the solution is feasible.
        If ``raise_error`` is ``True``, the function raises a ``ValueError``
        instead.

    Returns
    -------
    bool
        Indication if the solution is feasible (``True``) or not (``False``)

    Raises
    ------
    ValueError
        This is only raised when ``raise_error`` is ``True``.
    """

    assignments = np.array(assignments)
    num_items = len(weights)
    num_ks = len(capacities)
    error_msg = None

    if not np.all(np.shape(assignments) == (num_items, num_ks)):
        error_msg = "There is a mismatch of dimensions of the assigment matrix. It needs to be (num_items x num_knapsacks)."
    if not is_binary(assignments):
        error_msg = "The assignment matrix needs to be binary."
    if np.any(np.sum(assignments, axis=1) > 1):
        error_msg = "Each element can only by assigned at most once."

    if error_msg is not None:
        if raise_error:
            raise ValueError(error_msg)
        else:
            return False

    loads = weights @ assignments
    if np.any(loads > capacities):
        error_msg = "The capacity constraint is violated"

    if error_msg is None:
        return True
    else:
        if raise_error:
            raise ValueError(error_msg)
        else:
            return False


def is_symmetric_profits(profits: np.array, raise_error: bool = False) -> bool:
    """Check whether the profit matrix is symmetric.

    This function performs a check whether the profit matrix :math:`P` is
    symmetric. This is expected for the QMKP.

    By default, the function returns ``True`` if the matrix is symmetric and
    ``False`` otherwise.
    When ``raise_error`` is set to ``True``, a :class:`ValueError` is raised
    instead.


    Parameters
    ----------
    profits : np.array
        Symmetric matrix of size :math:`N\\times N` that contains the (joint)
        profit values :math:`p_{ij}`. The profit of the single items
        :math:`p_i` corresponds to the main diagonal elements, i.e.,
        :math:`p_i = p_{ii}`.

    raise_error : bool, optional
        If ``raise_error`` is ``False``, the function returns a ``bool``,
        that states whether the solution is feasible.
        If ``raise_error`` is ``True``, the function raises a ``ValueError``
        instead.

    Returns
    -------
    bool
        Indication if the solution is feasible (``True``) or not (``False``)

    Raises
    ------
    ValueError
        This is raised when ``raise_error`` is ``True`` and the matrix is not
        symmetric. It can also be raised when the provided ``profits`` is not a
        square matrix.
    """

    profits = np.array(profits)
    if np.ndim(profits) != 2:
        raise ValueError("The profits argument needs to be a 2D matrix.")
    _row_p, _cols_p = np.shape(profits)
    if not _row_p == _cols_p:
        raise ValueError("The profit matrix is not square")

    _symmetric = np.allclose(profits, profits.T)
    if raise_error and not _symmetric:
        raise ValueError("The profit matrix is not symmetric.")
    return _symmetric
