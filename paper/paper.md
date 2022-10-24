---
title: 'QMKPy: A Python Testbed for the Quadratic Multiple Knapsack Problem'
tags:
  - knapsack problems
  - QMKP
  - operations research
  - combinatorial optimization
authors:
  - name: Karl-Ludwig Besser
    orcid: 0000-0002-1597-8963
    affiliation: 1
  - name: Eduard A. Jorswieck
    orcid: 0000-0001-7893-8435
    affiliation: 1
affiliations:
 - name: Institute for Communications Technology, Technische Universität Braunschweig, Germany
   index: 1
date: 24 October 2022
bibliography: paper.bib
---

# Summary

QMKPy provides a Python framework for modeling and solving the quadratic
multiple knapsack problem (QMKP).
It is primarily aimed at researchers who develop new solution algorithms for
the QMKP.
QMKPy therefore mostly functions as a testbed to quickly implement novel
algorithms and compare their results with existing ones.
However, the package also already includes implementations of established
algorithms for those who only need to solve a QMKP as part of their research.

The QMKP is a type of knapsack problem which has first been analyzed by
@Hiley2006.
For a basic overview of other types of knapsack problems see, e.g.,
@Kellerer2004.
As in the regular multiple knapsack problem, the goal is to assign
$N\in\mathbb{N}$ items with given weights $w_i\in\mathbb{R}_{+}$ and
(non-negative) profit values $p_i\in\mathbb{R}_{+}$ to $K\in\mathbb{N}$
knapsacks with given weight capacities $c_u\in\mathbb{R}_{+}$, such that a
total profit is maximized.
In the QMKP, there exist additional joint profits $p_{ij}\in\mathbb{R}_{+}$
which are yielded when items $i$ and $j$ are packed into the same knapsack.

Mathematically, the QMKP is described by the following optimization problem
\begin{subequations}
\begin{alignat}{3}
	\max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}_u} p_{i} &+&\sum_{\substack{j\in\mathcal{A}_u \\ j\neq i}} p_{ij}\Bigg)\\
	\mathrm{s.t.}\quad & \sum_{i\in\mathcal{A}_u} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K} \\
	& \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall i\in\{1, 2, \dots{}, N\}
\end{alignat}
\end{subequations}
where $\mathcal{K}=\{1, 2, \dots{}, K\}$ describes the set of knapsacks,
$\mathcal{A}_u\subseteq\{1, 2, \dots{}, N\}$ is the set of items that are
assigned to knapsack $u$ and $a_{iu}\in\{0,1\}$ is a binary variable indicating
whether item $i$ is assigned to knapsack $u$.


# Statement of need

The QMKP is a NP-hard optimization problem and therefore, there exists a
variety of (heuristic) algorithms to find good solutions for it.
While Python frameworks already exist for the standard (multiple)
knapsack problem and the quadratic knapsack problem, they do not consider the
_quadratic multiple_ knapsack problem.
However, this type of problem arises in many areas of research. In addition to
the typical problems in operations research, it also occurs in distributed
computing [@Rust2020] and in the area of wireless communications
[@Besser2022wiopt].

For the classic knapsack problem and the quadratic _single_ knapsack problem,
many well-known optimization frameworks like Gurobi and @ortools provide
solution routines.
However, they are not directly applicable to the QMKP. Furthermore, it can be
difficult for researchers to reproduce results that rely on commerical
software.

Therefore, QMKPy aims to close that gap by providing an open source testbed to
easily implement and compare solution algorithms. Additionally, Python is
widely used among researchers and enables easy to read implementations.
This further supports the goal of QMKPy to promote sharable and reproducible
solution algorithms.

For initial comparisons, the software already implements multiple solution
algorithms for the QMKP, including a _constructive procedure (CP)_ based on
Algorithm 1 from @Aider2022 and the greedy heuristic from @Hiley2006. A second
algorithm that is included is the _fix and complete solution (FCS) procedure_
from Algorithm 2 in @Aider2022.
Additionally, a collection of reference QMKP instances that can be used with
QMKPy is provided at [@QMKPyDatasets].
This dataset includes the well-known reference problems from @Hiley2006, which
in turn are based on the quadratic single knapsack problems from
@Billionnet2004.

The open source nature of QMKPy and its aim at researchers encourages the
implementation of more algorithms for solving the QMKP that can become part of
the QMKPy framework.

The most notable benefits when implementing an algorithm using QMKPy are the
following:

- No additional overhead is required. Only a single function with the novel
  solution algorithm needs to be implemented.
- Generic unit tests are available to make sure that the novel algorithm
  fulfills basic criteria.
- The ability of loading and saving problem instances allows for quick and
  easy testing of any algorithm against reference datasets. This enables
  reproducible research and creates a high degree of comparability between
  different algorithms.


# References
