""" DE operators for mutation
"""

import numpy as np
from random import random


def de_rand_1(pop: np.ndarray, f: float, *args, **kwargs):
    xr1, xr2, xr3 = pop[np.random.choice(pop.shape[0], 3, replace=False)]

    return xr1 + f * (xr2 - xr3)


def de_rand_2(pop: np.ndarray, f: float, *args, **kwargs):
    xr1, xr2, xr3, xr4, xr5 = pop[np.random.choice(pop.shape[0], 5, replace=False)]

    return xr1 + f * (xr2 - xr3 + xr4 - xr5)


def de_best_1(pop: np.ndarray, f: float, index_best: int, *args, **kwargs):
    x_best = pop[index_best]
    pop = np.delete(pop, index_best, axis=0)
    xr1, xr2 = pop[np.random.choice(pop.shape[0], 2, replace=False)]

    return x_best + f * (xr1 - xr2)


def de_best_2(pop: np.ndarray, f: float, index_best: int, *args, **kwargs):
    x_best = pop[index_best]
    pop = np.delete(pop, x_best, axis=0)
    xr1, xr2, xr3, xr4 = pop[np.random.choice(pop.shape[0], 4, replace=False)]

    return x_best + f * (xr1 - xr2 + xr3 - xr4)


def de_current_to_best_1(pop: np.ndarray, f: float, index_best: int, index_curr:int, *args, **kwargs):
    x_curr = pop[index_curr]
    x_best = pop[index_best]
    pop = np.delete(pop, (index_best, index_curr), axis=0)
    xr1, xr2 = pop[np.random.choice(pop.shape[0], 2, replace=False)]

    return x_curr + f * (x_best - x_curr + xr1 - xr2)


def de_rand_to_best_1(pop: np.ndarray, f: float, index_best: int, *args, **kwargs):
    x_best = pop[index_best]
    pop = np.delete(pop, index_best, axis=0)
    xr1, xr2, xr3 = pop[np.random.choice(pop.shape[0], 3, replace=False)]

    return xr1 + f * (x_best - xr1 + xr2 - xr3)


def son_de_rand_1(pop: np.ndarray, f: float, *args, **kwargs):
    index = kwargs["index"]
    sg = kwargs["sg"][index]
    tg = kwargs["tg"][index]

    if len(sg) == 0:
        xr2, xr3 = tg[np.random.choice([i for i in range(len(tg))], 2, replace=False)]
    elif len(tg) == 0:
        xr2, xr3 = sg[np.random.choice([i for i in range(len(sg))], 2, replace=False)]
    else:
        xr2 = tg[np.random.choice([i for i in range(len(tg))], replace=False)]
        xr3 = sg[np.random.choice([i for i in range(len(sg))], replace=False)]
    
    nei = np.concatenate((sg, tg), axis=0)
    nei = nei[[i for i in range(len(nei)) if (nei[i] != xr2).any() and (nei[i] != xr3).any()]]
    
    if len(nei) == 0:
        xr1 = xr2 if random() < 0.5 else xr3
    else:
        xr1 = nei[np.random.randint(0, len(nei))]

    return xr1 + f * (xr2 - xr3)


def proposed_rand_1(pop: np.ndarray, f: float, *args, **kwargs):
    cluster = kwargs["cluster"]
    alternative_1 = kwargs["alternative_1"]
    alternative_2 = kwargs["alternative_2"]

    if len(cluster) < 3:
        cluster = np.concatenate((cluster, alternative_1), axis=0)
    if len(cluster) < 3:
        cluster = np.concatenate((cluster, alternative_2), axis=0)

    xr1, xr2, xr3 = cluster[np.random.choice([i for i in range(len(cluster))], 3, replace=False)]

    return xr1 + f * (xr2 - xr3)


def proposed_f_rand_1(pop: np.ndarray, f: tuple[float, float], *args, **kwargs):
    cluster = kwargs["cluster"]
    alternative_1 = kwargs["alternative_1"]
    alternative_2 = kwargs["alternative_2"]

    if len(cluster) < 3:
        cluster = np.concatenate((cluster, alternative_1), axis=0)
    if len(cluster) < 3:
        cluster = np.concatenate((cluster, alternative_2), axis=0)

    xr1, xr2, xr3 = cluster[np.random.choice([i for i in range(len(cluster))], 3, replace=False)]

    min_f = min(f)
    delta_f = abs(f[0] - f[1])

    return xr1 + (min_f + np.random.rand() * delta_f) * (xr2 - xr3)


if __name__ == "__main__":
    pop = np.arange(100)
    pop.resize((25, 4))
    de_current_to_best_1(pop, 0.5, 9, 10)

    son_de_rand_1(None, 0.5, 
                **{
                    "sg": np.array([[1, 1], [2, 2]]),
                    "tg": np.array([[2, 1], [3, 2]]),
                })