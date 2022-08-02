# QMKPy: A Python Framework for Solving Quadratic Multiple Knapsack Problems

[![Pytest](https://github.com/klb2/qmkpy/actions/workflows/pytest.yml/badge.svg)](https://github.com/klb2/qmkpy/actions/workflows/pytest.yml)


This software package primarily aims at research in the areas of operations
research and optimization.
It provides a way of quickly implementing and testing new algorithms to solve
the quadratic multiple knapsack problem (QMKP) and compare it with existing
solutions.


## Problem Description
The QMKP is defined as the following combinatorial optimization problem
$$
\begin{alignat}{3}
	\max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}_u} p_{u,i} &+&\sum_{\substack{j\in\mathcal{A}_u\\j\neq i}} p_{u,ij}\Bigg)\\
	\mathrm{s.\,t.}\quad & \sum_{i\in\mathcal{A}_u} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K}\\%\text{for all}\; 1\leq u \leq K\\
	& \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall\; 1\leq i \leq N
\end{alignat}
$$

This describes an assignment problem where one wants to assign $N$ items to $K$
knapsacks. If item $i$ is assigned to a knapsack, it yields the profit $p_i$.
If item $j$ ($j\neq i$) is assigned _to the same_ knapsack, we get the
additional joint profit $p_{ij}$.

## Installation
The package can easily be installed via pip.
Either from the PyPI
```bash
pip3 install qkmpy
```
or from the GitHub repository
```bash
git clone https://github.com/klb2/qmkpy.git
cd qmkpy
git checkout dev  # optional for the latest development version
pip3 install .
```

