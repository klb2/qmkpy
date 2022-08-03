.. QMKPy documentation master file, created by
   sphinx-quickstart on Thu Jul 21 15:45:55 2022.

Welcome to QMKPy's documentation!
=================================

**QMKPy** is a Python library for modeling and solving quadratic multiple
knapsack problems (QMKP).
It provides a framework that allows quickly implementing and testing novel
algorithms to solve the QMKP.
It is therefore primarily targeting researchers working in the area of
operations research/optimization.

Additionally, it can be used to easily generate datasets of QMKP instances
which can be used as reference test set to fairly compare different algorithms.


The QMKP that can be solved by this package is defined as follows

.. math::

    \begin{alignat}{3}
        \max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}_{u}} p_{i} &+&\sum_{\substack{j\in\mathcal{A}_{u}\\ j\neq i}} p_{ij}\Bigg)\\
        \mathrm{s.t.}\quad & \sum_{i\in\mathcal{A}_{u}} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K} \\
        & \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall 1\leq i \leq N
    \end{alignat}

where :math:`\mathcal{K}` describes the set of knapsacks, :math:`\mathcal{A}_u`
is the set of items that are assigned to knapsack :math:`u`, and
:math:`a_{iu}\in\{0, 1\}` is the indicator whether item :math:`i` is assigned
to knapsack :math:`u`.
Each item :math:`i` has the weight :math:`w_i` and knapsack :math:`u` has the
weight capacity :math:`c_u`.
When assigning item :math:`i` to a knapsack, it yields the profit :math:`p_i`.
When assigning item :math:`j` (with :math:`j\neq i`) to the same knapsack, the
additional (joint) profit :math:`p_{ij}` is obtained.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   examples.rst
   conventions.rst
   developing.rst
   datasets.rst
   qmkpy.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
