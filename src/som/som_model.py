""" Self Organizing Maps module
"""

import numpy as np
from typing import Dict

from src.utils.distances import *


class SOM:
    def __init__(
            self,
            n_dim: int,
            topology_shape: tuple
    ):
        if isinstance(topology_shape, int):
            topology_shape = (topology_shape)

        elif isinstance(topology_shape, tuple) or isinstance(topology_shape, list):
            topology_shape = [size for size in topology_shape]
            topology_shape.append(n_dim)

        self.__prototypes = np.zeros(topology_shape)
        self.__structure = {
            "distance": euclidean_squared,
            "neighborhood": euclidean
        }

    def attach(self, structure: Dict) -> None:
        pass