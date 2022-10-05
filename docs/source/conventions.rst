Code Conventions
================

The code in the library follows some conventions, which are specified in the
following.
We assume that there are :math:`N` items and :math:`K` knapsacks.

Basic Arrays
------------
In the following, we will discuss the essential elements of the QMKP and their
implementation in the ``qmkpy`` framework.

Mathematical Description
^^^^^^^^^^^^^^^^^^^^^^^^
The four essential components of the QMKP are the following.

Profit matrix :math:`P\in\mathbb{R}_{+}^{N\times N}`
    This symmetric matrix contains the profit values :math:`p_i` on the main
    diagonal and the joint profit values :math:`p_{ij}` as the other elements.

Weights :math:`w\in\mathbb{R}_{+}^{N}`
    This vector contains the weights of the items, where the :math:`i`-th
    component :math:`w_i` corresponds to the weight of item :math:`i`.

Capacities :math:`c\in\mathbb{R}_{+}^{K}`
    This vector contains the capacities of the knapsacks, where the
    :math:`i`-th component :math:`c_i` corresponds to the capacity of knapsack
    :math:`i`.

Assignments :math:`\mathcal{A}=\{\mathcal{A}_1, \mathcal{A}_2, \dots, \mathcal{A}_K\}` with :math:`\mathcal{A}_i\subseteq \{1, 2, \dots{}, N\}`
    The assignments of items to knapsacks are collected in the set
    :math:`\mathcal{A}`. It contains the individual sets :math:`\mathcal{A}_i`
    which contains the indices of all items that are assigned to knapsack
    :math:`i`.


Implementation
^^^^^^^^^^^^^^
The three main components described above are implemented in ``qmkpy`` as
arrays. The details are as follows.

``profits``
    The profit matrix is implemented as an array of size ``[N, N]`` which
    represents the symmetric :math:`N \times N` matrix :math:`P`.
    We have that the profits of the individual items :math:`p_i` are placed on
    the main diagonal ``profits[i-1, i-1] = p_i`` and the joint profits
    :math:`p_{ij}` make up the other elements as ``profits[i-1, j-1] =
    profits[j-1, i-1] = p_{ij}``. (The ``-1`` index shift is due to Python's
    0-based indexing.)

``weights``
    The weight vector is implemented as a list of length ``N``, where the
    weight :math:`w_i` corresponds to the index ``i-1``, i.e., ``weights[i-1] =
    w_i``.

``capacities``
    The capacities vector is implemented as a list of length ``K``, where the
    capacity :math:`c_i` corresponds to the index ``i-1``, i.e.,
    ``capacities[i-1] = c_i``.

``assignments``
    There are multiple ways of representing the assignment of items to
    knapsacks. *For all algorithms, the binary representation is used to
    represent the solution to a QMKP.*
    In this, the assignments :math:`\mathcal{A}` are represented by a binary
    array of size ``[N, K]`` where row :math:`i` stands for item :math:`i` and
    column :math:`u` represents knapsack :math:`u`.
    Thus, element ``assignments[i-1, u-1] = 1``, if item :math:`i` is assigned
    to knapsack :math:`u` and ``assignments[i-1, u-1] = 0`` otherwise.  


Argument Order
--------------
Functions that work on a QMKP always assume the argument order ``profits,
weights, capacities`` and they are expected to return ``assignments`` in the
binary form described above.

So if you want to write a function that solves a QMKP, the argument list of
your function needs to start with this.
More details on this can also be found on the `Implementing a Novel
Algorithm <developing.html>`_ page.


Alternative Representation of the Assignment Matrix
---------------------------------------------------
There are multiple ways of representing the final solution to a QMKP.
Essentially, we need to represent the assignment of the items to the knapsacks.

Besides the binary representation of the algorithms, which is described above,
another popular representation is the chromosome form :math:`C\in\{0, 1,
\dots{}, K\}^{N}` which is a vector of length :math:`N`, where the value of
entry :math:`i` specifies the knapsack to which item :math:`i` is assigned.
If the item is not assigned to any knapsack, the value 0 is used.
In the ``qmkpy`` framework, this is implemented such that ``chromosome`` is a
list of length ``N``, where index ``i-1`` represents item :math:`i`, i.e.,
``chromosome[i-1] = u-1`` indicates that item :math:`i` is assigned to knapsack
:math:`u`. If item :math:`i` is not assigned to any knapsack, we have
``chromosome[i-1] = -1``.

While the binary representation is dominantly used in this library, there exist
functions to convert the to representations (see
:meth:`qmkpy.util.assignment_from_chromosome` and
:meth:`qmkpy.util.chromosome_from_assignment`).
