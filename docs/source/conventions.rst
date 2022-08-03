Code Conventions
================

The code in the library follows some conventions, which are specified in the
following.
We assume that there are :math:`N` items and :math:`K` knapsacks.

Basic Arrays
------------
The basic arrays for every QMKP are

- ``profits``, which is a symmetric :math:`N \times N` array with :math:`p_i`
  on the main diagonal and :math:`p_{ij}` on the other elements.
- ``weights``, which is a vector/list of length :math:`N`, where the
  :math:`i`-th element specifies the weight :math:`w_i` of item :math:`i`.
- ``capacities``, which is a vector/list of length :math:`K`, where the
  :math:`i`-th element specifies the weight capacity :math:`c_i` of knapsack
  :math:`i`.


Argument Order
--------------
Functions that work on a QMKP always assume the argument order ``profits,
weights, capacities``.

So if you want to write a function that solves a QMKP, the argument list of
your function needs to start with this.
More details on this can also be found on the `Implementing a Novel
Algorithm <developing.html>`_ page.


Assignment Matrix / Solution of the QMKP
----------------------------------------
There are multiple ways of representing the final solution to a QMKP.
Essentially, we need to represent the assignment of the items to the knapsacks.

_For all algorithms, the binary representation is used to represent the
solution to a QMKP._
In this, the assignments :math:`A\in\{0, 1\}^{N\times K}` are represented by a
binary matrix where row :math:`i` stands for item :math:`i` and column
:math:`u` represents knapsack :math:`u`.
Thus, element :math:`a_{iu}=1`, if item :math:`i` is assigned to knapsack
:math:`u` and 0 otherwise.

Another popular representation is the chromosome :math:`C\in\{0, 1, \dots{},
K\}^{N}` which is a vector of length :math:`N`, where the value of entry
:math:`i` specifies the knapsack to which item :math:`i` is assigned.
If the item is not assigned to any knapsack, the value 0 is used.

While the binary representation is dominantly used in this library, there exist
functions to convert the to representations (see
:meth:`qmkpy.util.assignment_from_chromosome` and
:meth:`qmkpy.util.chromosome_from_assignment`).
