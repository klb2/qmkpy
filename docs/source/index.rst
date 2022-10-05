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


The QMKP is an assignment problem where :math:`N\in\mathbb{N}` items are
assigned to :math:`K\in\mathbb{N}` knapsacks such that an overall profit is
maximized.
The exact formulation of the QMKP that can be solved by this package is given
as follows

.. math::

    \begin{alignat}{3}
        \max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}({u})} p_{i} &+&\sum_{\substack{j\in\mathcal{A}({u})\\ j\neq i}} p_{ij}\Bigg)\\
        \mathrm{s.t.}\quad & \sum_{i\in\mathcal{A}({u})} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K} \\
        & \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall i \in \{1, 2, \dots, N\}
    \end{alignat}

where :math:`\mathcal{K}=\{1, 2, \dots, K\}` describes the set of :math:`K`
knapsacks, :math:`\mathcal{A}({u})` is the set of items that are assigned to
knapsack :math:`u`, and :math:`a_{iu}\in\{0, 1\}` is the indicator whether item
:math:`i` is assigned to knapsack :math:`u`.
Each item :math:`i` has the weight :math:`w_i\in\mathbb{R}_{+}` and knapsack
:math:`u` has the weight capacity :math:`c_u\in\mathbb{R}_{+}`.
When assigning item :math:`i` to a knapsack, it yields the non-negative profit
:math:`p_i\in\mathbb{R}_{+}`.
When assigning item :math:`j` (with :math:`j\neq i`) to the same knapsack, the
additional (joint) profit :math:`p_{ij}\in\mathbb{R}_{+}` is obtained.

The objective of the above optimization problem is to maximize the total profit
such that each item is assigned to at most one knapsack and such that the
weight capacity constraints of the knapsacks are not violated.

*Remark:* The profits :math:`p` are also referred to as "values" in the
literature.

A detailed description of the way how the mathematical components of the QMKP
are implemented in the ``qmkpy`` framework can be found in the `code
conventions page <conventions.html#basic-arrays>`_.

For a basic overview on knapsack problems, see [KPP04]_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   examples.rst
   conventions.rst
   developing.rst
   datasets.rst
   qmkpy.rst
   references.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
