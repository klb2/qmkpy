Examples
========
In the following, a few simple examples are shown.


Basic Usage
-----------
The following script contains an example in which a QMKP is defined and solved
by the implemented constructive procedure.

.. code-block:: python
   :caption: Defining and Solving a QMKP
   :linenos:

    import numpy as np
    from qmkpy import total_profit_qmkp, QMKProblem
    from qmkpy import algorithms

    weights = [5, 2, 3, 4]  # four items
    capacities = [10, 5, 12, 4, 2]  # five knapsacks
    profits = np.array([[3, 1, 0, 2],
                        [1, 1, 1, 4],
                        [0, 1, 2, 2],
                        [2, 4, 2, 3]])  # symmetric profit matrix

    qmkp = QMKProblem(profits, weights, capacities)
    qmkp.algorithm = algorithms.constructive_procedure
    assignments, total_profit = qmkp.solve()

    print(assignments)
    print(total_profit)


Saving a Problem Instance
-------------------------
It is possible to save a problem instance of a QMKP. This can be useful to
share examples as a benchmark dataset to compare different algorithms.

.. code-block:: python
   :caption: Saving a QMKProblem Instance
   :linenos:

    import numpy as np
    from qmkpy import total_profit_qmkp, QMKProblem
    from qmkpy import algorithms

    weights = [5, 2, 3, 4]  # four items
    capacities = [10, 5, 12, 4, 2]  # five knapsacks
    profits = np.array([[3, 1, 0, 2],
                        [1, 1, 1, 4],
                        [0, 1, 2, 2],
                        [2, 4, 2, 3]])  # symmetric profit matrix

    qmkp = QMKProblem(profits, weights, capacities)

    # Save the problem instance using the Numpy npz format
    qmkp.save("my_problem.npz", strategy="numpy") 
