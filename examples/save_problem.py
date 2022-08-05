import numpy as np
from qmkpy import QMKProblem


def main():
    FILENAME = "problem.txt"

    num_elements = 200
    num_knapsacks = 20

    profits = 5*np.random.rand(num_elements, num_elements)
    profits = profits @ profits.T
    weights = np.random.randint(1, 5, size=(num_elements,))
    capacities = np.random.randint(3, 12, size=(num_knapsacks,))
    qmkp = QMKProblem(profits, weights, capacities)
    print(f"Saving the problem into file: {FILENAME}")
    qmkp.save("problem.txt", strategy='txt')
    print("Successfully saved the problem.")

if __name__ == "__main__":
    main()
