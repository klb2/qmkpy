# QMKPy Examples

In this directory, you can find some example script of how to use the QMKPy
library.

Make sure that all requirements are installed by running
```bash
pip3 install -r requirements.txt
```

All of the examples can simply be run via
```bash
python3 <scriptName>.py
```

Additionally, there is a Jupyter notebook provided, which illustrates how a new
algorithm can be easily implemented.  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/qmkpy/HEAD?labpath=examples%2FCustom%20Algorithm.ipynb)


## Example List
The following examples are provided

| **File Name** | **Description** |
|---------------|-----------------|
| `algorithm_comparison.py` | This file contains a simulation where multiple algorithms solve the same randomly generated QMKProblem instances and the mean profit is compared. |
| `basic_usage.py` | Simple script where a QMKP is specified and solved. |
| `custom_algorithm.py` | Illustration of how a custom solution algorithm can be implemented. |
| `save_problem.py` | Example of how to save a problem instance (using the `txt` strategy) |
| `Custom Algorithm.ipynb` | Jupyter notebook to illustrate how a custom solution algorithm can be implemented. |
