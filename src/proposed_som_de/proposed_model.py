""" SOM_DE Module
"""

import numpy as np
from typing import Any, Dict

from src.utils.exceptions import *
from src.de.de_model import DE
from src.som.som2_model import PenalizedActivationSOM as SOM
from src.proposed_som_de.external_file import ExternalFile
from src.proposed_som_de.operators import grouping, group_selection
from src.utils.distances import cosine
from src.utils.mutation import proposed_rand_1
from src.utils.crossing import binary
from src.utils.neighborhood import gaussian, exponential


class Model:
    def __init__(
        self,
        dim: int,
        NP: int,
        fitness_func: Any,
        topology_shape: tuple,
        init_method: Any = None,
        is_maximization: bool = False,
        adjust = None,
        sigma0 = None,
        tau0 = None,
        neig_func = None,
    ):
        self.__de = DE(dim, NP, init_method, is_maximization)
        self.__som = SOM(dim + 1, topology_shape, init_method)
        self.__is_maximization = is_maximization
        self.__NP=NP

        self.__de.attach(
            {
                "mutation": proposed_rand_1,
                "crossing": binary(0.5),
                "fitness": fitness_func,
                "F": 0.5
            }
        )
        self.__som.attach(
            {
                "distance": cosine,
                "neighborhood": gaussian(sigma=1)
            }
        )

        self.__external_file = ExternalFile()
        self.__external_file.append(
            np.concatenate(
                (self.__de.get_population(), self.__de.get_pop_fitness()[:, np.newaxis]),
                axis=1
            ))
        self.__ef_ratio = 0

        self.__operators = {
            "grouping": grouping,
            "group_selection": group_selection,
            "adjust": adjust,
            "sigma0": sigma0,
            "tau0": tau0
        }
    
    def new_gen(self, som_epochs: int, delta: float, limits: tuple = (3, 10), **kwargs) -> None:
        last_fit = self.__de.get_pop_fitness()

        if len(self.__external_file) // self.__NP > self.__ef_ratio:
            self.__ef_ratio = len(self.__external_file) // self.__NP

            for epoch in range(som_epochs):
                for i, x in enumerate(self.__external_file.get_not_acessed()):
                    if self.__operators["adjust"]:
                        sigma = self.__operators["sigma0"] * self.__operators["adjust"](i=i, t=epoch, T=som_epochs, NP=self.__NP)
                        tau = self.__operators["tau0"] * self.__operators["adjust"](i=i, t=epoch, T=som_epochs, NP=self.__NP)
                        self.__som.attach(
                            {
                                "neighborhood": exponential(sigma),
                                "alpha": tau
                            }
                        )

                    self.__som.update(x)

        clusters = self.__operators["grouping"](self.__de.get_pop_fitness(), self.__de.get_population(), self.__som)
        cluster, alternative_1, alternative_2 = self.__operators["group_selection"](clusters, self.__is_maximization)

        self.__de.new_generation(
            **{
                "cluster": cluster,
                "alternative_1": alternative_1,
                "alternative_2": alternative_2
            }
        )

        current_fit = self.__de.get_pop_fitness()
        new_indviduals = self.__de.get_population()[current_fit != last_fit]
        new_indviduals = np.concatenate((new_indviduals, current_fit[current_fit != last_fit][:, np.newaxis]), axis=1)

        if len(new_indviduals) > 0:
            self.__external_file.append(new_indviduals)
    
    def attach_de(self, structure):
        self.__de.attach(structure)
    
    def attach_som(self, structure):
        self.__som.attach(structure)
    
    def get_pop_fitness(self):
        return self.__de.get_pop_fitness()
    
    def get_population(self):
        return self.__de.get_population()

    def best_individual_index(self):
        return self.__de.best_individual_index()
    
    def get_acc_matrix(self):
        return self.__som.get_acc_matrix()


if __name__ == "__main__":
    print(
        np.concatenate((np.zeros((3, 4)), np.ones(3)[:, np.newaxis]), axis=1)
    )