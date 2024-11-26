""" SON-DE operators module
"""
from math import floor

import numpy as np

from src.utils.distances import euclidean_squared


def relationship_building(de_pop: np.ndarray, som_prototypes: np.ndarray) -> np.ndarray:
    lb = np.zeros(som_prototypes.shape[:-1], dtype=np.int32) - 1
    for i, x in enumerate(de_pop):
        distances = euclidean_squared(x, som_prototypes).astype(np.float32)
        distances[np.unravel_index(lb[lb != -1], lb.shape)] = np.inf
    
        lb[np.unravel_index(i, lb.shape)] = np.argmin(distances)
    
    return lb


def neighborhood_size(fitness: np.ndarray, delta: float, limits: tuple, is_maximization: bool) -> np.ndarray:
    rank = np.argsort(fitness)

    if is_maximization:
        rank = rank[::-1]
    
    ranges = max(limits) - delta * rank
    ranges[ranges < min(limits)] = min(limits)

    return ranges


def locating(lb: np.ndarray, ranges:np.ndarray) -> list[np.ndarray]:
    topology_loc = list()
    for i in range(len(ranges)):
        topology_loc.append(
            np.where(
                lb==i
            )
        )
    topology_loc = np.array(topology_loc).squeeze()
    
    neighbors = list()
    for i in range(len(ranges)):
        loc = topology_loc[i]
        distances = euclidean_squared(loc, topology_loc)

        n_num = floor(ranges[i]) + 1
        n_num = len(ranges) if n_num > len(ranges) else n_num
        neighbors.append(np.argsort(distances)[1: n_num])

    return neighbors


def grouping(fitness: np.ndarray, nei: list[np.ndarray], is_maximization: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sg = list()
    tg = list()
    for i, neighbors in enumerate(nei):
        if is_maximization:
            tg_indexes = fitness[neighbors] >= fitness[i]
        else:
            tg_indexes = fitness[neighbors] <= fitness[i]

        tg.append(neighbors[tg_indexes])
        sg.append(neighbors[tg_indexes==False])
    
    return sg, tg


if __name__ == "__main__":
    a = np.array(
        [
            [2, 2],
            [1, 1],
            [0, 0],
            [3, 3]
        ]
    )
    b = np.array(
        [
            [[0, 0],
            [1, 1]],
            [[2, 2],
            [3, 3]]
        ]
    )
    nei = locating(relationship_building(a, b), [1, 2, 3, 4])
    fit = np.array([3, 2, 1, 4])

    print(grouping(fit, nei, False))
