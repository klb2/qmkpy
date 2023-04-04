# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Add support of heterogeneous profits. This includes the extension of existing
  solution algorithms to support 3D profit matrices.
- Add new `utils.get_possible_assignments` function that shows which unassigned
  items could fit into which knapsacks.


## [1.2.0] - 2022-10-25
### Added
- Add new `json` strategy to save and load problem instances
- Add new (optional) `name` attribute to the `QMKProblem` class
- Add custom `__str__` method to the `QMKProblem` class which prints the name
  attribute, if it exists, or a string with the number of items and knapsacks
  otherwise.

### Updated
- The `io.save_problem_txt` will now first try to use the `problem.name`
  atrribute if the `name` keyword is not given.
- Change the (sub)title of the package to "QMKPy: A Python Testbed for the
  Quadratic Multiple Knapsack Problem"


## [1.1.0] - 2022-08-08
### Added
- Add example scripts to repository
- Add new utility functions
  - `util.get_unassigned_items`
  - `util.get_empty_knapsacks`
  - `util.get_remaining_capacities`
- Add new check function
  - `checks.is_symmetric_profits`
- Update documentation about datasets and add example

### Fixed
- Fix docstrings


## [1.0.0] - 2022-08-04
### Added
- Class `QMKProblem` for defining a QMKP
- Functions to save and load QMKProblem instances. The available strategies are
  - `numpy` (using numpy's npz format)
  - `pickle` (using Python's pickle library)
  - `txt` (using a text-based format)
- Module with some checks for feasibility of solutions, if an array is binary
- Some util functions, e.g., to convert the binary assignment matrix to the
  chromosome representation
- Solution algorithms for the QMKP
  - Constructive procedure
  - FCS procedure
  - Simple round robin scheme
  - Random assignment
