# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Add example scripts to repository


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
