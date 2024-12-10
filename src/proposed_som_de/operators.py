""" SON-DE operators module
"""
from math import floor
from random import random

import numpy as np

from src.utils.distances import euclidean_squared
from src.som.som_model import SOM


def grouping(fitness: np.ndarray, pop: np.ndarray, som: SOM) -> dict[float, list[np.ndarray]]:
    pop = np.concatenate((pop, fitness[:, np.newaxis]), axis=1)
    clusters = dict()

    for ind in pop:
        bmu = som.bmu(ind)
        # bmu = ",".join(list(map(str, bmu)))
        bmu = str(bmu)

        if bmu in clusters:
            clusters[bmu] = np.concatenate((clusters[bmu], ind[np.newaxis, :]), axis=0)
        else: 
            clusters[bmu] = ind[np.newaxis, :]
    
    valuated_clusters = {sum(val[:, -1]) / len(val): val[:, :-1] for val in clusters.values()}

    return valuated_clusters


def group_selection(clusters: dict, is_maximization: bool) -> tuple[np.ndarray]:
    if not is_maximization:
        max_ = max(clusters.keys())
        clusters = {max_ - key: clusters[key] for key in clusters.keys()}

    clusters = {np.exp(key / 5): clusters[key] for key in clusters.keys()}

    a = random() * sum(clusters.keys())
    acc = 0
    for key in clusters.keys():
        acc += key
        if acc >= a:
            cluster = clusters.pop(key)
            break

    alternative_1 = None
    alternative_2 = None
    a = random() * sum(clusters.keys())
    acc = 0
    for key in clusters.keys():
        acc += key
        if acc >= a:
            alternative_1 = clusters.pop(key)
            break

    a = random() * sum(clusters.keys())
    acc = 0
    for key in clusters.keys():
        acc += key
        if acc >= a:
            alternative_2 = clusters.pop(key)
            break

    return cluster, alternative_1, alternative_2


if __name__ == "__main__":
    pass
