""" DE operators for mutation
"""

import numpy as np


def de_rand_1(pop: np.ndarray, f: float, *args):
    xr1, xr2, xr3 = pop[np.random.choice(pop.shape[0], 3, replace=False)]

    return xr1 + f * (xr2 - xr3)


def de_rand_2(pop: np.ndarray, f: float, *args):
    xr1, xr2, xr3, xr4, xr5 = pop[np.random.choice(pop.shape[0], 5, replace=False)]

    return xr1 + f * (xr2 - xr3 + xr4 - xr5)


def de_best_1(pop: np.ndarray, f: float, index_best: int, *args):
    x_best = pop[index_best]
    pop = np.delete(pop, index_best, axis=0)
    xr1, xr2 = pop[np.random.choice(pop.shape[0], 2, replace=False)]

    return x_best + f * (xr1 - xr2)


def de_best_2(pop: np.ndarray, f: float, index_best: int, *args):
    x_best = pop[index_best]
    pop = np.delete(pop, x_best, axis=0)
    xr1, xr2, xr3, xr4 = pop[np.random.choice(pop.shape[0], 4, replace=False)]

    return x_best + f * (xr1 - xr2 + xr3 - xr4)


def de_current_to_best_1(pop: np.ndarray, f: float, index_best: int, index_curr:int, *args):
    x_curr = pop[index_curr]
    x_best = pop[index_best]
    pop = np.delete(pop, (index_best, index_curr), axis=0)
    xr1, xr2 = pop[np.random.choice(pop.shape[0], 2, replace=False)]

    return x_curr + f * (x_best - x_curr + xr1 - xr2)


def de_rand_to_best_1(pop: np.ndarray, f: float, index_best: int, *args):
    x_best = pop[index_best]
    pop = np.delete(pop, index_best, axis=0)
    xr1, xr2, xr3 = pop[np.random.choice(pop.shape[0], 3, replace=False)]

    return xr1 + f * (x_best - xr1 + xr2 - xr3)


pop = np.arange(100)
pop.resize((25, 4))
de_current_to_best_1(pop, 0.5, 9, 10)
