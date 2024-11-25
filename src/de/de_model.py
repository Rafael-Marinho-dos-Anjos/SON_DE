""" Diferencial Evolution Module
"""

import numpy as np
from typing import Any, Dict

from src.utils.exceptions import *


class DE:
    def __init__(self, dim: int, NP: int, init_method: Any = None, is_maximization: bool = False):
        self.__population = np.zeros((NP, dim))
        self.__fitness = np.zeros(NP)
        self.__have_fit = np.zeros(NP).astype(np.bool)
        self.__is_maximization = is_maximization
        
        if init_method:
            self.__population = init_method(self.__population)

        self.__structure = {
            "mutation": None,
            "crossing": None,
            "fitness": None,
            "F": 0.5
        }

    def __calc_fitness(self):
        for i in range(self.__population.shape[0]):
            if not self.__have_fit[i]:
                self.__fitness[i] = self.__structure["fitness"](self.__population[i])
    
    def get_pop_fitness(self):
        self.__calc_fitness()
        
        return self.__fitness

    def get_population(self):
        return self.__population

    def attach(self, structure: Dict) -> None:
        for key in structure.keys():
            if key == "F":
                self.__structure[key] = structure[key]
                continue

            if key not in self.__structure.keys():
                raise UnknownStructureException(f"{key} are not a known DE operator.")
            
            if not callable(structure[key]):
                raise NotCallableElementException(f"{key} is not a callable element.")
            
            self.__structure[key] = structure[key]

    def best_individual_index(self):
        self.__calc_fitness()

        if self.__is_maximization:
            return np.max(self.__fitness)
        else:
            return np.min(self.__fitness)
    
    def new_generation(self):
        for i in range(self.__population.shape[0]):
            v = self.__structure["mutation"](
                self.__population,
                self.best_individual_index(),
                i
            )
        
            trial = self.__structure["crossing"](self.__population[i], v)
            trial_fit = self.__structure["fitness"](trial)

            if trial_fit >= self.__fitness[i]:
                self.__population[i] = trial
                self.__fitness[i] = trial_fit
