# How to Contribute

Contributions are welcome and very much appreciated.

There are multiple ways in which you can contribute to the project.


## Report Bugs
Please report bugs by opening a new issue on Github.

If you are reporting a bug, please include the following information:

- Version of the package and the versions of the required packages
- Your operating system
- Detailed steps to reproduce the bug

## Fix Bugs
Any open issue that is tagged with "bug" is open to whoever wants to fix it.


## Implementing New Algorithms
You can always implement new algorithms to solve the QMKP. If you want to do
this, please check the documentation on style guidelines like the order of
arguments.

Please also make sure that:

- you write documentation (Numpy-style docstrings), which should include
  references to literature, if the algorithm is taken from any published work.
- you add your algorithm to `SOLVERS` in the `tests/test_algorithms.py`
  unit test file and that your implementation passes all the tests.
- you write additional unit tests in `tests/test_algorithm_<ALG>.py`.
