# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Class `QMKProblem` for defining a QMKP
- Module with some checks for feasibility of solutions, if an array is binary
- Some util functions, e.g., to convert the binary assignment matrix to the
  chromosome representation
- Solution algorithms for the QMKP
  - Constructive procedure
  - FCS procedure
  - Simple round robin scheme
  - Random assignment
