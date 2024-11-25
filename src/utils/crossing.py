""" DE operators for crossing
"""

import numpy as np


def binary(CR: float):
    def __crossing(a: np.ndarray, b: np.ndarray):
        j = np.random.randint(0, len(a))
        trial = np.zeros(len(a))

        for i in range(len(a)):
            if i == j or np.random.rand() <= CR:
                trial[i] = b[i]
            else:
                trial[i] = a[i]

        return trial

    return __crossing


pop = np.arange(100)
pop.resize((25, 4))
