---
title: 'QMKPy: A Python Framework for the Quadratic Multiple Knapsack Problem'
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
date: 08 August 2022
bibliography: paper.bib
---

# Summary

QMKPy provides a Python framework for modeling and solving the quadratic
multiple knapsack problem (QMKP).
It primarily aims at researchers who develop new solution algorithms for the
QMKP. 
However, the package also already includes implementations of established
algorithms for those who only need to solve a QMKP as part of their research.

The QMKP is a type of knapsack problem which has first been analyzed by
[@Hiley2006].
For a basic overview of multiple types of knapsack problems see, e.g.,
[@Kellerer2004].
As in the regular multiple knapsack problem, the goal is to assign $N$ items
with given weights $w_i$ and profit values $p_i$ to $K$ knapsacks with given
weight capacities $c_u$, such that a total profit is maximized.
In the QMKP, there exist additional joint profits $p_{ij}$ which are yielded
when items $i$ and $j$ are packed into the same knapsack.

Mathematically, the QMKP is described by the following optimization problem
\begin{subequations}
\begin{alignat}{3}
	\max\quad & \sum_{u\in\mathcal{K}}\Bigg(\sum_{i\in\mathcal{A}_u} p_{i} &+&\sum_{\substack{j\in\mathcal{A}_u \\ j\neq i}} p_{ij}\Bigg)\\
	\mathrm{s.t.}\quad & \sum_{i\in\mathcal{A}_u} w_{i} \leq c_u & \quad & \forall u\in\mathcal{K} \\
	& \sum_{u=1}^{K} a_{iu} \leq 1  & & \forall 1\leq i \leq N
\end{alignat}
\end{subequations}
where $\mathcal{A}_u$ is the set of items that are assigned to knapsack $u$ and
$a_{iu}\in\{0,1\}$ is a binary variable indicating whether item $i$ is assigned
to knapsack $u$.


# Statement of need

The QMKP is a NP-hard optimization problem and therefore, there exists a
variety of (heuristic) algorithms to find good solutions for it.
While there already exist Python frameworks for the standard (multiple)
knapsack problem, they do not consider the _quadratic_ multiple knapsack
problem.
However, this type of problem arises in many areas of research. Besides the
typical problems in operations research, it also occurs in distributed
computing [@Rust2020] and in the area of wireless communications
[@Besser2022wiopt].

For solving QMKPs, the software already implements multiple solution
algorithms, including a _constructive procedure (CP)_ based on Algorithm 1 from
[@Aider2022] and the greedy heuristic from [@Hiley2006]. A second algorithm
that is included is the _fix and complete solution (FCS) procedure_ from
Algorithm 2 in [@Aider2022].

However, most of the benefits of QMKPy appear when implementing a novel
algorithm.
Most notably are the following:

- No additional overhead is required. Only a single function with the novel
  solution algorithm needs to be implemented.
- Generic unit tests are available to make sure that your algorithm fulfills
  basic criteria.
- The ability of loading and saving problem instances allows for a quick and
  easy testing of any algorithm against reference datasets. This enables
  reproducible research and creates a high degree of comparability between
  different algorithms. A collection of reference QMKP instances that can be
  used with QMKPy can be found at
  [https://github.com/klb2/qmkpy-datasets](https://github.com/klb2/qmkpy-datasets).


# References
