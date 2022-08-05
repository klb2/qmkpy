import numpy as np
from qmkpy import total_profit_qmkp, QMKProblem
from qmkpy import algorithms
from qmkpy import util

weights = [5, 2, 3, 4]  # four items
capacities = [10, 5, 12, 4, 2]  # five knapsacks
profits = np.array([[3, 1, 0, 2],
                    [1, 1, 1, 4],
                    [0, 1, 2, 2],
                    [2, 4, 2, 3]])  # symmetric profit matrix

qmkp = QMKProblem(profits, weights, capacities)
qmkp.algorithm = algorithms.constructive_procedure
assignments, total_profit = qmkp.solve()
chromosome = util.chromosome_from_assignment(assignments)

print(f"Item weights:\t{weights}")
print(f"KS capacities:\t{capacities}")
print(f"Profit matrix:\n{profits}")
print("-"*10)
print(f"Solution (assignments):\n{assignments}")
print(f"Solution as chromosome:\n{chromosome}")
print(f"Total profit with this solution: {total_profit:.2f}")
