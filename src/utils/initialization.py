""" Population initialization algorithms
"""

import numpy as np


def gaussian(pop_shape: tuple, sdt_dev: tuple):
    """
    Gaussian distribution.
    
    Params:
        pop_shape (tuple): Population shape.
        std_dev (tuple): Standart deviation.
    
    Return:
        np.ndarray.
    """
    if not isinstance(sdt_dev, np.ndarray):
        sdt_dev = np.array(sdt_dev, dtype=np.float32)
    
    population = np.random.normal(
        loc=0,
        scale=sdt_dev,
        size=pop_shape
    )

    return population

def random(pop_shape: tuple, limits: tuple = None):
    """
    Random distribution.
    
    Params:
        pop_shape (tuple): Population shape.
        limits (tuple): Every individual dimension limit.
    
    Return:
        np.ndarray.
    """
    if limits is None:
        limits = np.array([[0, 1]])

    if not isinstance(limits, np.ndarray):
        limits = np.array(limits, dtype=np.float32)
    
    population = np.random.random(pop_shape)
    population = population * (limits[:, 1] - limits[:, 0])
    population = population + limits[:, 0]

    return population

def lhs(pop_shape: tuple, limits: tuple = None):
    """
    Latin Hypercube Sampling (LHS).
    
    Params:
        pop_shape (tuple): Population shape.
        limits (tuple): Every individual dimension limit.
    
    Return:
        np.ndarray.
    """
    if limits is None:
        limits = np.array([[0, 1]])

    if not isinstance(limits, np.ndarray):
        limits = np.array(limits, dtype=np.float32)
    
    n_pop = 1
    for i in pop_shape[:-1]:
        n_pop *= i
    
    n_dim = pop_shape[-1]
    population = np.zeros((n_pop, n_dim))
    
    for i in range(n_dim):
        intervals = np.random.permutation(n_pop)
        population[:, i] = (intervals + np.random.rand(n_pop)) / n_pop
        
    population = population * (limits[:, 1] - limits[:, 0])
    population = population + limits[:, 0]

    population = np.reshape(population, pop_shape)
    
    return population


if __name__ == "__main__":
    lim = np.array(
        [
            [0, 1],
            [1, 1],
            [3, 5],
            [6, 9]
        ]
    )

    print(random((2, 5, 4)))
