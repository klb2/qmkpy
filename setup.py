from setuptools import setup, find_packages

from qmkpy import __version__, __author__, __email__

with open("README.md", encoding="utf8") as rm:
    long_desc = rm.read()

setup(
    name = "qmkpy",
    version = __version__,
    author = __author__,
    author_email = __email__,
    description = "Framework for solving the Quadratic Multiple Knapsack Problem (QMKP)",
    keywords = ["quadratic multiple knapsack problem", "knapsack problem",
                "operations research"],
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url='https://github.com/klb2/qmkpy',
    project_urls={
        'Documentation': "https://qmkpy.readthedocs.io",
        'Source Code': 'https://github.com/klb2/qmkpy',
        },
    classifiers = [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        ],
    packages=find_packages(),
    python_requires=">=3.7",
    tests_require=['pytest', 'pytest-cov'],
    install_requires=['numpy'],
    zip_safe=True,
)
