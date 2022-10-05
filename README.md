# QMKPy: A Python Framework for the Quadratic Multiple Knapsack Problem

[![Pytest](https://github.com/klb2/qmkpy/actions/workflows/pytest.yml/badge.svg)](https://github.com/klb2/qmkpy/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/klb2/qmkpy/branch/master/graph/badge.svg?token=NFBF1ZZEXQ)](https://codecov.io/gh/klb2/qmkpy)
[![Read the docs status](https://readthedocs.org/projects/qmkpy/badge/?version=latest&style=flat)](https://qmkpy.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/qmkpy)](https://pypi.org/project/qmkpy/)
![License](https://img.shields.io/github/license/klb2/qmkpy)


This software package primarily aims at research in the areas of operations
research and optimization.
It provides a way of quickly implementing and testing new algorithms to solve
the quadratic multiple knapsack problem (QMKP) and compare it with existing
solutions.


## Problem Description
The QMKP is defined as the following combinatorial optimization problem

$$
\begin{alignat}{3}
	\max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}(u)} p_{i} &+&\sum_{\substack{j\in\mathcal{A}(u), \\ j\neq i}} p_{ij}\Bigg)\\
	\mathrm{s.t.}\quad & \sum_{i\in\mathcal{A}(u)} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K} \\
	& \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall i \in \{1, 2, \dots, N\}
\end{alignat}
$$

This describes an assignment problem where one wants to assign $N\in\mathbb{N}$
items to $K\in\mathbb{N}$ knapsacks, which are described by the index set
$\mathcal{K}=\{1, 2, \dots, K\}$.
Item $i$ has the weight $w_i\in\mathbb{R_+}$ and knapsack $u$ has the weight
capacity $c_u\mathbb{R_+}$.
If item $i$ is assigned to a knapsack, it yields the (non-negative) profit
$p_i\in\mathbb{R_+}$.
If item $j$ (with $j\neq i$ ) is assigned _to the same_ knapsack, we get the
additional joint profit $p_{ij}\in\mathbb{R_+}$.

The set of items which are assigned to knapsack $u$ is denoted by
$\mathcal{A}(u)$ and $a_{iu}\in\{0, 1\}$ is an indicator whether item $i$ is
assigned to knapsack $u$.

The objective of the above problem is to maximize the total profit such that
each item is assigned to at most one knapsack and such that the weight capacity
constraints of the knapsacks are not violated.

_Remark:_ The profits $p$ are also referred to as "values" in the literature.


## Features

- Quick and simple creation of QMKP instances
- Saving/loading of problem instances for a simple creation and use of
  reference datasets
- Easy implementation of novel algorithms to solve the QMKP
- High reproducibility and direct comparison between different algorithms


The benefit of enabling a simple and direct way of implementing novel
algorithms is highlighted by an example in the provided Jupyter notebook in
[examples/Custom
Algorithm.ipynb](https://github.com/klb2/qmkpy/blob/master/examples/Custom%20Algorithm.ipynb).  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/qmkpy/HEAD?labpath=examples%2FCustom%20Algorithm.ipynb)


## Installation
The package can easily be installed via pip.
Either from the PyPI
```bash
pip3 install qmkpy
```
or from the GitHub repository
```bash
git clone https://github.com/klb2/qmkpy.git
cd qmkpy
git checkout dev  # optional for the latest development version
pip3 install -r requirements.txt
pip3 install .
pip3 install pytest  # optional if you want to run the unit tests
```

## Usage
In order to test the installation and get an idea of how to use the QMKPy
package, you can take a look at the `examples/` directory.
It contains some standalone scripts that can be executed and perform some
simple tasks.

More detailed descriptions of the implemented algorithms and a documentation of
the API can be found in the [documentation](https://qmkpy.readthedocs.io).

A collection of reference datasets can be found at
[https://github.com/klb2/qmkpy-datasets](https://github.com/klb2/qmkpy-datasets).


## Contributing
Please see
[CONTRIBUTING.md](https://github.com/klb2/qmkpy/blob/master/CONTRIBUTING.md)
for guidelines on how to contribute to this project.
In particular, novel algorithms are always welcome. Please check out the
[documentation](https://qmkpy.readthedocs.io/en/latest/developing.html#contributing-a-new-algorithm-to-the-package)
for a brief overview on how to implement new algorithms for the QMKPy
framework.
