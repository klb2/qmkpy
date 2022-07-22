__author__ = "Karl Besser"
__email__ = "k.besser@tu-bs.de"
__version__ = "0.1.0"


from . import algorithms
from . import checks
from . import util
from .qmkp import QMKProblem, total_profit_qmkp
from .util import value_density, chromosome_from_assignment

__all__ = ["__version__",
           "algorithms",
           "checks",
           "util",
           "QMKProblem",
           "total_profit_qmkp",
           "value_density",
           "chromosome_from_assignment",
          ]
