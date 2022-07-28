Implementing a Novel Algorithm
==============================

**TL;DR**
Your function needs to look like: ``func(profits, weights, capacities, *args)``


If you want to implement and test a novel solution algorithm for the QMKP, you
simply need to write a Python function that takes ``profits`` as first
argument, ``weights`` as second, and ``capacities`` as third argument.
Beyond that, it can have an arbitrary number of addiotional arguments.
However, it needs to be possible to pass them positionally.

The return of the function needs to be the assignment matrix in binary form.


Example
-------
As an example, we want to implement the following algorithm

    Assign the item :math:`i` with the smallest weight :math:`w_i` to the first
    knapsack :math:`k` where it fits, i.e., where :math:`c_k \geq w_i`.

Obviously, this algorithm ignores the profits and will not yield very good
results. However, it only serves demonstration purposes.

Algorithm Implementation
************************
The above algorithm could be implemented as follows

.. code-block:: python
   :linenos:
   :caption: Example Algorithm

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

It should be emphasized that you should **not** modify any of the input arrays,
e.g., ``capacities`` inplace, since this could lead to unintended consequences.

Testing the Algorithm
*********************
The newly implemented algorithm can then easily be tested as follows.

.. code-block:: python
   :linenos:
   :caption: Testing the Novel Algorithm

    import numpy as np
    from qmkpy import total_profit_qmkp, QMKProblem
    from qmkpy import algorithms

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

