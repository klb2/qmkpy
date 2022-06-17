from typing import Iterable, Any

import numpy as np

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
