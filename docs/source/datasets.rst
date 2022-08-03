Datasets
========
One of the major benefits of this package is the possibility to quickly and
easily generate datasets of reference problems and test your algorithms against
(existing) datasets.

Especially when benchmarking your novel algorithm against commonly used
reference datasets, this will allow a simple reproducibility.

In the following, an example of how a repository for a research paper could
look like, is presented.

Research Paper Repository
-------------------------
The file structure can be as simple as shown in the following.

::
    
    project
    ├── dataset/          
    │   ├── problem1.txt
    │   ├── problem2.txt
    │   └── ...
    └── my_algorithm.py


The directory ``dataset/`` contains all problem instances of the reference
dataset, which are saved by one of the functions in :mod:`qmkpy.io`.

The file ``my_algorithm.py`` contains the implementation of your algorithm.
It could look something like the following. Details on how to implement new
algorithms can also be found on the `Implementing a Novel Algorithm
<developing.html>`_ page.

.. code-block:: python
    :linenos:

    import os
    import numpy as np
    import qmkpy

    def my_algorithm(profits, weights, capacities):
        # DOING SOME STUFF
        return assignments

    def main():
        results = []
        for root, dirnames, filenames in os.walk("dataset"):
            for problem in filenames:
                qmkp = qmkpy.QMKProblem.load(problem, strategy="txt")
                qmkp.algorithm = my_algorithm
                solution, profit = qmkp.solve()
                results.append(profit)
        print(f"Average profit: {np.mean(results):.2f}")

    if __name__ == "__main__":
        main()


This simple script solves all problems of the dataset using your algorithm and
prints the average total profit at the end.
