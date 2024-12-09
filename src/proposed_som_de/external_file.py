""" External file module
"""

import numpy as np


class ExternalFile:
    def __init__(self):
        self.__cache = None
        self.__not_acessed = list()
    
    def append(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        if self.__cache is None:
            self.__cache = x

            self.__not_acessed = [i for i in range(len(x))]

        else:
            a = len(self.__cache)
            b = len(x)
            self.__cache = np.concatenate((self.__cache, x), axis=0)

            for i in range(b):
                self.__not_acessed.append(a + i)
    
    def get_not_acessed(self) -> np.ndarray:
        not_acessed = self.__cache[self.__not_acessed]
        self.__not_acessed = list()

        return not_acessed

    def get_total_cache(self):
        return self.__cache

    def __len__(self):
        if not self.__cache:
            return 0

        return len(self.__cache)


if __name__ == "__main__":
    ef = ExternalFile()

    ef.append(np.zeros(3))
    ef.append(np.ones(3))
    print(ef.get_not_acessed())
    ef.append(np.zeros((2, 3)))
    ef.append(np.ones((3, 3)))
    print(ef.get_not_acessed())
    print(ef.get_total_cache())
