""" Self Organizing Maps with penalized activation module
"""

from src.som.som_model import SOM, np
from src.utils.exceptions import *


class PenalizedActivationSOM(SOM):
    def __init__(self, n_dim, topology_shape, init_method, curve_smoothing = 10):
        super().__init__(n_dim, topology_shape, init_method)

        self._acc_matrix = np.zeros(topology_shape, dtype=int)
        self._penalization_curve = lambda x: np.exp((x * -1) / curve_smoothing)

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
        penalization = self._penalization_curve(self._acc_matrix)

        diff = self._prototypes - input
        diff = diff * neighborhood
        diff = diff * penalization[:, :, np.newaxis]
        diff = diff * self._alpha

        self._prototypes = self._prototypes - diff
        self._acc_matrix[bmu] += 1

    def get_acc_matrix(self):
        return self._acc_matrix
