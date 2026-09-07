import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from petitRADTRANS import physical_constants as cst

# Import the class used to set up the retrieval.
from petitRADTRANS.retrieval import Retrieval,RetrievalConfig
# Import Prior functions, if necessary.
from petitRADTRANS.retrieval.utils import gaussian_prior
# Import an atmospheric model function
from petitRADTRANS.retrieval.models import isothermal_transmission 