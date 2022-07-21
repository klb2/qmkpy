from typing import Iterable, Any, Union, NoReturn, Optional

import numpy as np

def check_dimensions(profits: np.array,
                     weights: Optional[Iterable[float]] = None) -> NoReturn:
    """
    Simple check whether the dimensions of the parameters match.

    Parameters
    ----------
    profits : array (N x N)
        Symmetric matrix containing the profits :math:`p_{ij}`

    weights : list of length N, optional
        List which contains the weights of the :math:`N` items.

    Raises
    ------
    This function raises a ValueError, if there is a mismatch
    """
    _row_p, _cols_p = np.shape(profits)
    if not _row_p == _cols_p:
        raise ValueError("The profit matrix is not square")
    if weights is not None:
        num_items = len(weights)
        if not num_items == _row_p:
            raise ValueError("The number of items does not match the number of profits.")

def is_binary(x):
    x = np.array(x)
    return ((x == 0) | (x == 1)).all()

def is_feasible_solution(assignments: np.array, capacities: Iterable[float],
                         weights: Iterable[float], profits: np.array,
                         raise_error: bool = False):
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
