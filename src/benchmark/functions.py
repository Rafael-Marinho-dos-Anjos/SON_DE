import numpy as np


def sphere(o: np.ndarray):
    def __function(x: np.ndarray):
        z = x - o
        res = np.sum(z ** 2, axis=0)

        return res
    
    return __function


def rotated_high_conditioned_elliptic(o: np.ndarray, m: np.ndarray):
    def __function(x: np.ndarray):
        z = np.matmul(m, (x - o))
        d = len(x)
        i = np.arange(d) # Como a contagem começa de 0, não precisa fazer i-1 depois
        res = 10 ** (6 * i / (d - 1))
        res = res * z ** 2

        return res
    
    return __function


def different_powers(o: np.ndarray):
    def __function(x: np.ndarray):
        z = np.abs(x - o)
        d = len(x)
        i = np.arange(d) # Como a contagem começa de 0, não precisa fazer i-1 depois
        res = z ** (2 + 4 * i / (d - 1))

        return np.sum(res, axis=0)
    
    return __function
