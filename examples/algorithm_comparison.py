import numpy as np
import matplotlib.pyplot as plt

from qmkpy import total_profit_qmkp, QMKProblem
from qmkpy import algorithms, checks

SOLVERS = {"CP": algorithms.constructive_procedure,
           "FCS": algorithms.fcs_procedure,
           "Random": algorithms.random_assignment,
           "Round Robin": algorithms.round_robin,
          }

def run_simulation():
    num_elements = 20
    num_knapsacks = 5
    num_runs = 500
    results = {k: [] for k in SOLVERS}

    for run in range(num_runs):
        print(f"Run {run+1:d}/{num_runs:d}")
        profits = 5*np.random.rand(num_elements, num_elements)
        profits = profits @ profits.T
        weights = np.random.randint(1, 5, size=(num_elements,))
        capacities = np.random.randint(3, 12, size=(num_knapsacks,))
        qmkp = QMKProblem(profits, weights, capacities)
        for _name, solver in SOLVERS.items():
            qmkp.algorithm = solver
            _assignments, total_profit = qmkp.solve()
            assert checks.is_feasible_solution(_assignments, profits, weights,
                                               capacities)
            results[_name].append(total_profit)
    
    print("Finished all runs.")
    fig, axs = plt.subplots()
    for _name, _profits in results.items():
        print(f"Algorithm: {_name}\tAverage: {np.mean(_profits):.3f}")
        #axs.hist(_profits, bins=100, label=_name)
        axs.hist(_profits, bins=100, label=_name, density=True, histtype="step",
                 cumulative=True)
    fig.legend()

    return results

if __name__ == "__main__":
    run_simulation()
    plt.show()
