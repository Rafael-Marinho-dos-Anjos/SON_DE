""" SON_DE Module
"""

import numpy as np
from typing import Any, Dict

from src.utils.exceptions import *
from src.de.de_model import DE
from src.som.som_model import SOM
from src.son_de.operators import *
from src.utils.distances import cosine
from src.utils.mutation import son_de_rand_1
from src.utils.crossing import binary
from src.utils.neighborhood import gaussian, exponential


class SON_DE:
    def __init__(
        self,
        dim: int,
        NP: int,
        fitness_func: Any,
        topology_shape: tuple,
        init_method: Any = None,
        is_maximization: bool = False,
        reset_prototypes: bool = False,
        adjust = None,
        sigma0 = None,
        tau0 = None,
        neig_func = None,
    ):
        self.__de = DE(dim, NP, init_method, is_maximization)
        self.__som = SOM(dim, topology_shape, init_method)
        self.__is_maximization = is_maximization
        self.__NP=NP
        self.__reset_prototypes = reset_prototypes

        self.__de.attach(
            {
                "mutation": son_de_rand_1,
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

        self.__operators = {
            "relationship_building": relationship_building,
            "neighborhood_size": neighborhood_size,
            "locating": locating,
            "grouping": grouping,
            "adjust": adjust,
            "sigma0": sigma0,
            "tau0": tau0
        }
    
    def new_gen(self, som_epochs: int, delta: float, limits: tuple = (3, 10), **kwargs) -> None:
        for epoch in range(som_epochs):
            for i, x in enumerate(self.__de.get_population()):
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
        
        lb = self.__operators["relationship_building"](
            self.__de.get_population(),
            self.__som.get_prototypes()
        )

        ranges = self.__operators["neighborhood_size"](
            self.__de.get_pop_fitness(),
            delta,
            limits,
            self.__is_maximization
        )

        nei = self.__operators["locating"](
            lb,
            ranges
        )

        sg, tg = self.__operators["grouping"](
            self.__de.get_pop_fitness(),
            nei,
            self.__is_maximization
        )

        sg = [self.__de.get_population()[ind] for ind in sg]
        tg = [self.__de.get_population()[ind] for ind in tg]

        self.__de.new_generation(
            **{
                "sg": sg,
                "tg": tg
            }
        )

        if self.__reset_prototypes:
            self.__som.reset_prototypes()
    
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
