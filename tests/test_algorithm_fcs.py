import numpy as np
import pytest

from qmkpy import total_profit_qmkp
from qmkpy.algorithms import fcs_procedure, constructive_procedure


@pytest.mark.parametrize("heterogeneous", (False, True))
@pytest.mark.parametrize("alpha", (None, 0.5, 0.1, 0.999, 0.2))
def test_fcs_with_alpha(heterogeneous, alpha):
    num_elements = 8
    num_knapsacks = 3
    if heterogeneous:
        profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
        profits = profits @ profits.transpose([0, 2, 1])
    else:
        profits = np.random.randint(0, 8, size=(num_elements, num_elements))
        profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution = fcs_procedure(profits, weights, capacities, alpha=alpha)
    total_profit = total_profit_qmkp(profits, solution)
    assert (
        np.all(np.shape(solution) == (num_elements, num_knapsacks))
        and total_profit >= 0
    )


@pytest.mark.parametrize("heterogeneous", (False, True))
@pytest.mark.parametrize("len_history", (1, 10, 15, 20, 50))
def test_fcs_with_history(heterogeneous, len_history):
    num_elements = 8
    num_knapsacks = 3
    if heterogeneous:
        profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
        profits = profits @ profits.transpose([0, 2, 1])
    else:
        profits = np.random.randint(0, 8, size=(num_elements, num_elements))
        profits = profits @ profits.T
    #profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    #profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    solution = fcs_procedure(profits, weights, capacities, len_history=len_history)
    total_profit = total_profit_qmkp(profits, solution)
    assert (
        np.all(np.shape(solution) == (num_elements, num_knapsacks))
        and total_profit >= 0
    )


@pytest.mark.parametrize("heterogeneous", (False, True))
@pytest.mark.parametrize("alpha", (0, 1, -0.2, 1.2))
def test_fcs_alpha_feasible(heterogeneous, alpha):
    num_elements = 8
    num_knapsacks = 3
    if heterogeneous:
        profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
        profits = profits @ profits.transpose([0, 2, 1])
    else:
        profits = np.random.randint(0, 8, size=(num_elements, num_elements))
        profits = profits @ profits.T
    #profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    #profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    with pytest.raises(ValueError):
        fcs_procedure(profits, weights, capacities, alpha=alpha)


@pytest.mark.parametrize("heterogeneous", (False, True))
@pytest.mark.parametrize("len_history", (0, 0.1, -20))
def test_fcs_history_feasible(heterogeneous, len_history):
    num_elements = 8
    num_knapsacks = 3
    if heterogeneous:
        profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
        profits = profits @ profits.transpose([0, 2, 1])
    else:
        profits = np.random.randint(0, 8, size=(num_elements, num_elements))
        profits = profits @ profits.T
    #profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    #profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    with pytest.raises(ValueError):
        fcs_procedure(profits, weights, capacities, len_history=len_history)


@pytest.mark.parametrize("heterogeneous", (False, True))
@pytest.mark.parametrize("alpha", (None, 0.5, 0.1, 0.999, 0.2))
def test_fcs_compare_cp(heterogeneous, alpha):
    num_elements = 8
    num_knapsacks = 3
    if heterogeneous:
        profits = np.random.randint(0, 8, size=(num_knapsacks, num_elements, num_elements))
        profits = profits @ profits.transpose([0, 2, 1])
    else:
        profits = np.random.randint(0, 8, size=(num_elements, num_elements))
        profits = profits @ profits.T
    #profits = np.random.randint(0, 8, size=(num_elements, num_elements))
    #profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    sol_fcs = fcs_procedure(profits, weights, capacities, alpha=alpha)
    profit_fcs = total_profit_qmkp(profits, sol_fcs)
    sol_cp = constructive_procedure(profits, weights, capacities)
    profit_cp = total_profit_qmkp(profits, sol_cp)
    assert profit_fcs >= profit_cp
