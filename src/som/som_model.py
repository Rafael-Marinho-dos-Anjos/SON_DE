""" Self Organizing Maps module
"""

import numpy as np
from typing import Dict

from src.utils.distances import *
from src.utils.neighborhood import *
from src.utils.exceptions import *


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
            "neighborhood": gaussian(sigma=1)
        }

    def attach(self, structure: Dict) -> None:
        for key in structure.keys():
            if key not in self.__structure.keys():
                raise UnknownStructureException(f"{key} are not a known SOM structural element.")
            
            if not callable(structure[key]):
                raise NotCallableElementException(f"{key} is not a callable element.")
            
            self.__structure[key] = structure[key]
    
    def bmu(self, data: np.ndarray, return_dists: bool = False) -> tuple:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim() > 1:
            data = data.flatten()

        if data.shape[0] != self.__prototypes[-1]:
            raise WrongShapeException("The shape of given input is not accepted.")
        
        dists = self.__structure["distance"](self.__prototypes, data)

        if return_dists:
            return np.argmin(dists), dists
    
        return np.argmin(dists)
