""" Distance functions used by SOM
"""

import numpy as np


def euclidean_squared(a: np.ndarray, b: np.ndarray) -> float:
    """ 
    Euclidean distance squared
    
    Params:
        a: ndarray: First input vector
        b: ndarray: Seccond input vector
    
    Return: The euclidean distance squared between a and b vectors
    Return type: float | ndarray
    """
    dist = a - b
    dist = np.square(dist)
    dist = np.sum(dist, axis=-1)

    return dist


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """ 
    Euclidean distance
    
    Params:
        a: ndarray: First input vector
        b: ndarray: Seccond input vector
    
    Return: The euclidean distance between a and b vectors
    Return type: float | ndarray
    """
    dist = euclidean_squared(a, b)
    dist = np.sqrt(dist)

    return dist


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """ 
    Manhattan's distance
    
    Params:
        a: ndarray: First input vector
        b: ndarray: Seccond input vector
    
    Return: The manhattan's distance between a and b vectors
    Return type: float | ndarray
    """
    dist = a - b
    dist = np.abs(dist)
    dist = np.sum(dist, axis=-1)

    return dist


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """ 
    Cosine distance
    
    Params:
        a: ndarray: First input vector
        b: ndarray: Seccond input vector
    
    Return: The cosine distance between a and b vectors
    Return type: float | ndarray
    """
    dist = np.dot(a, b)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)

    return 1 - dist / norms


if __name__ == "__main__":
    a = np.array(
        [
            [[1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4]]
        ]
    )
    b = np.array([1, 2, 3])

    print(euclidean_squared(a, b))
