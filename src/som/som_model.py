""" Self Organizing Maps module
"""

import numpy as np
from typing import Dict, Any

from src.utils.distances import *
from src.utils.neighborhood import *
from src.utils.exceptions import *


class SOM:
    def __init__(
            self,
            n_dim: int,
            topology_shape: tuple,
            init_method: Any
    ):
        if isinstance(topology_shape, int):
            topology_shape = (topology_shape)

        elif isinstance(topology_shape, tuple) or isinstance(topology_shape, list):
            topology_shape = [size for size in topology_shape]
            topology_shape.append(n_dim)

        self._alpha = 0.01

        if init_method:
            self.__init_method = lambda: init_method(topology_shape)
        else:
            self.__init_method = lambda: np.zeros(topology_shape)

        self._prototypes = self.__init_method()

        self._structure = {
            "distance": euclidean_squared,
            "neighborhood": gaussian(sigma=1)
        }

    def reset_prototypes(self):
        self._prototypes = self.__init_method()

    def attach(self, structure: Dict) -> None:
        for key in structure.keys():
            if key == "alpha":
                self._alpha = structure[key]
                continue

            if key not in self._structure.keys():
                raise UnknownStructureException(f"{key} are not a known SOM operator.")
            
            if not callable(structure[key]):
                raise NotCallableElementException(f"{key} is not a callable element.")
            
            self._structure[key] = structure[key]
    
    def bmu(self, input: np.ndarray, return_dists: bool = False) -> tuple:
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        
        if input.ndim > 1:
            input = input.flatten()

        if input.shape[0] != self._prototypes.shape[-1]:
            raise WrongShapeException("The shape of given input is not accepted.")
        
        dists = self._structure["distance"](self._prototypes, input)

        if return_dists:
            return np.argmin(dists), dists
    
        return np.argmin(dists)

    def neighborhood(self, bmu: tuple) -> np.ndarray:
        return self._structure["neighborhood"](
            self._prototypes.shape,
            bmu
        )
    
    def update(self, input: np.ndarray) -> None:
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        
        if input.ndim > 1:
            input = input.flatten()

        if input.shape[0] != self._prototypes.shape[-1]:
            raise WrongShapeException("The shape of given input is not accepted.")
        
        bmu = self.bmu(input)
        bmu = np.unravel_index(bmu, self._prototypes.shape[:-1])
        neighborhood = self.neighborhood(bmu)

        diff = self._prototypes - input
        diff = diff * neighborhood
        diff = diff * self._alpha

        self._prototypes = self._prototypes - diff

    def get_prototype(self, index: tuple) -> np.ndarray:
        prototype = self._prototypes.copy()
        for i in range(len(index)):
            prototype = prototype[i]

        return prototype

    def get_prototypes(self) -> np.ndarray:
        return self._prototypes.copy()