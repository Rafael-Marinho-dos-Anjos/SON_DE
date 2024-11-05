""" Diferencial Evolution Module
"""

import numpy as np


class DE:
    def __init__(self, dim: int, NP: int):
        self.__population = np.zeros((NP, dim))
        