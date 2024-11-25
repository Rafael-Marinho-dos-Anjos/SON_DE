""" Neighborhood functions
"""

import numpy as np
from typing import Any


def gaussian(sigma: float) -> Any:
    def __gaussian(neighborhood_shape: tuple, bmu_loc: tuple) -> np.ndarray:
        def __neighbor(*location):
            d = np.asarray(location)

            for i in range(len(bmu_loc)):
                d[i] = d[i] - bmu_loc[i]
            
            d = np.square(d)
            d = np.sum(d, axis=0)

            return np.exp(-d / (2 * sigma ** 2))

        neighbors = np.fromfunction(__neighbor, neighborhood_shape)

        return neighbors
    
    return __gaussian


def boltzmann(sigma: float, t: float) -> Any:
    def __boltzmann(neighborhood_shape: tuple, bmu_loc: tuple) -> np.ndarray:
        def __neighbor(*location):
            d = np.asarray(location)
            
            for i in range(len(bmu_loc)):
                d[i] = d[i] - bmu_loc[i]
            
            d = np.linalg.norm(d, axis=0)
            
            return 1 / (1 + np.exp((d - sigma) / t))

        neighbors = np.fromfunction(__neighbor, neighborhood_shape)

        return neighbors
    
    return __boltzmann


def linear(sigma: float) -> Any:
    def __linear(neighborhood_shape: tuple, bmu_loc: tuple) -> np.ndarray:
        def __neighbor(*location):
            d = np.asarray(location)
            
            for i in range(len(bmu_loc)):
                d[i] = d[i] - bmu_loc[i]
            
            d = np.linalg.norm(d, axis=0)
            
            return d * ((1 - (d / sigma)) > 0).astype(np.float32)

        neighbors = np.fromfunction(__neighbor, neighborhood_shape)

        return neighbors
    
    return __linear


def retangular(sigma: float) -> Any:
    def __retangular(neighborhood_shape: tuple, bmu_loc: tuple) -> np.ndarray:
        def __neighbor(*location):
            d = np.asarray(location)
            
            for i in range(len(bmu_loc)):
                d[i] = d[i] - bmu_loc[i]
            
            d = np.linalg.norm(d, axis=0)
            
            return (d <= sigma).astype(np.float32)

        neighbors = np.fromfunction(__neighbor, neighborhood_shape)

        return neighbors
    
    return __retangular


if __name__ == "__main__":
    print(np.round(100*gaussian(0.75)([3, 3], [2, 0]))/100)
    