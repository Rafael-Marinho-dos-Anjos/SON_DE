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

        self.__alpha = 0.01
        self.__prototypes = np.zeros(topology_shape)
        self.__structure = {
            "distance": euclidean_squared,
            "neighborhood": gaussian(sigma=1)
        }

    def attach(self, structure: Dict) -> None:
        for key in structure.keys():
            if key == "alpha":
                self.__alpha = structure[key]
                continue

            if key not in self.__structure.keys():
                raise UnknownStructureException(f"{key} are not a known SOM operator.")
            
            if not callable(structure[key]):
                raise NotCallableElementException(f"{key} is not a callable element.")
            
            self.__structure[key] = structure[key]
    
    def bmu(self, input: np.ndarray, return_dists: bool = False) -> tuple:
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        
        if input.ndim() > 1:
            input = input.flatten()

        if input.shape[0] != self.__prototypes[-1]:
            raise WrongShapeException("The shape of given input is not accepted.")
        
        dists = self.__structure["distance"](self.__prototypes, input)

        if return_dists:
            return np.argmin(dists), dists
    
        return np.argmin(dists)

    def neighborhood(self, bmu: tuple) -> np.ndarray:
        return self.__structure["neighborhood"](
            self.__prototypes.shape,
            bmu
        )
    
    def update(self, input: np.ndarray) -> None:
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        
        if input.ndim() > 1:
            input = input.flatten()

        if input.shape[0] != self.__prototypes[-1]:
            raise WrongShapeException("The shape of given input is not accepted.")
        
        bmu = self.bmu(input)
        neighborhood = self.neighborhood(bmu)

        diff = self.__prototypes - input
        diff = diff * neighborhood[:, :, np.newaxis]
        diff = diff * self.__alpha

        self.__prototypes = self.__prototypes - diff

    def get_prototype(self, index: tuple) -> np.ndarray:
        prototype = self.__prototypes.copy()
        for i in range(len(index)):
            prototype = prototype[i]

        return prototype
