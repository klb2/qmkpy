import numpy as np
from qmkpy import total_profit_qmkp, QMKProblem
from qmkpy import algorithms

def example_algorithm(profits, weights, capacities):
    assignments = np.zeros((len(weights), len(capacities)))
    remaining_capacities = np.copy(capacities)
    items_by_weight = np.argsort(weights)
    for _item in items_by_weight:
        _weight = weights[_item]
        _first_ks = np.argmax(remaining_capacities >= _weight)
        assignments[_item, _first_ks] = 1
        remaining_capacities[_first_ks] -= _weight
    return assignments

weights = [5, 2, 3, 4]  # four items
capacities = [1, 5, 5, 6, 2]  # five knapsacks
profits = np.array([[3, 1, 0, 2],
                    [1, 1, 1, 4],
                    [0, 1, 2, 2],
                    [2, 4, 2, 3]])  # symmetric profit matrix

qmkp = QMKProblem(profits, weights, capacities)
qmkp.algorithm = example_algorithm
assignments, total_profit = qmkp.solve()

print(assignments)
print(total_profit)
