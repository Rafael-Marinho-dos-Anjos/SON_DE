import numpy as np


def sphere(o: np.ndarray, **kwargs):
    def __function(x: np.ndarray):
        z = (x - o).astype(np.float64)
        res = np.sum(z ** 2, axis=0)

        return res
    
    return __function


def rotated_high_conditioned_elliptic(o: np.ndarray, **kwargs):
    m = kwargs["m"]

    def __function(x: np.ndarray):
        z = np.matmul(m, (x - o), dtype=np.longdouble)
        d = len(x)
        i = np.arange(d) # Como a contagem começa de 0, não precisa fazer i-1 depois
        res = 10 ** (6 * i / (d - 1))
        res = res * z ** 2

        return res.sum()
    
    return __function


def different_powers(o: np.ndarray, **kwargs):
    def __function(x: np.ndarray):
        z = np.abs(x - o).astype(np.longdouble)
        d = len(x)
        i = np.arange(d) # Como a contagem começa de 0, não precisa fazer i-1 depois
        res = z ** (2 + 4 * i / (d - 1))

        return np.sum(res, axis=0)
    
    return __function


def rotated_bent_cigar(o: np.ndarray, **kwargs):
    m = kwargs["m"]

    def __function(x: np.ndarray):
        z = np.dot(m, x - o).astype(np.float64)
        res = z[0] ** 2 + 10 ** 6 * np.sum(z[1:] ** 2, axis=0)
        return res
    return __function


def rotated_discus(o: np.ndarray, **kwargs):
    m = kwargs["m"].astype(np.longdouble)
    
    def __function(x: np.ndarray):
        z = np.dot(m, x - o).astype(np.longdouble)
        res = 10 ** 6 * z[0] ** 2 + np.sum(z[1:] ** 2, axis=0)

        return res
    
    return __function


# multimodal functions

def rotated_rosenbrock(o: np.ndarray, **kwargs):
    m = kwargs["m"]

    def __function(x: np.ndarray):
        z = (x - o).astype(np.float64) / 2.048
        z = np.dot(m, z) + 1
        res = np.sum(100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2, axis=0)
        return res
    return __function


def rotated_schaffer_f7(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        y = np.dot(m, np.matmul(lambda_, (x - o))).astype(np.longdouble)
        z = np.sqrt(y[:-1]**2 + y[1:]**2)
        res = np.sum(np.sqrt(z) * (1 + 0.001 * np.sin(50 * z)**2), axis=0)
        return res
    return __function


def rotated_ackley(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        z = np.dot(m, np.matmul(lambda_, (x - o))).astype(np.longdouble)
        D = z.shape[0]
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2, axis=0) / D))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * z), axis=0) / D)
        res = term1 + term2 + 20 + np.e
        return res
    return __function


def rotated_weierstrass(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    a, b, k_max = 0.5, 3, 20
    k = np.arange(1, 21)

    def __function(x: np.ndarray):
        z = np.dot(m, (x - o) / 100)
        D = z.shape[0]
        term1 = np.sum((a ** k)[:, np.newaxis] * np.cos(2 * np.pi * (b ** k)[:, np.newaxis] * (z + 0.5)))
        term2 = D * np.sum((a ** k)[:, np.newaxis] * np.cos(2 * np.pi * (b ** k)[:, np.newaxis] * 0.5))
        res = term1 - term2

        return res
    
    return __function


def rotated_griewank(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 100 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        z = np.dot(m, np.matmul(lambda_, (x - o))).astype(np.float64)
        D = z.shape[0]
        term1 = np.sum(z**2, axis=0) / 4000
        term2 = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))), axis=0)
        res = term1 - term2 + 1

        return res
    
    return __function


def rastrigin(o: np.ndarray, **kwargs):
    def __function(x: np.ndarray):
        z = x - o
        res = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10, axis=0)
        return res
    return __function


def rotated_rastrigin(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    alpha = kwargs["alpha"]

    def __function(x: np.ndarray):
        z = np.dot(m, np.diag(alpha) @ (x - o))
        res = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10, axis=0)
        return res
    return __function


def non_continuous_rotated_rastrigin(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    alpha = kwargs["alpha"]

    def __function(x: np.ndarray):
        y = np.where(np.abs(x - o) > 0.5, np.round(x - o), x - o)
        z = np.dot(m, np.diag(alpha) @ y)
        res = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10, axis=0)
        return res
    return __function


def schwefel(o: np.ndarray, **kwargs):
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))
    
    def __g(x):
        x_ = np.mod(np.abs(x), 500) - 500
        a = x_ * np.sin(np.sqrt(np.abs(x_))) - np.square(x + 500) / (10000 * len(o))
        b = -x_ * np.sin(np.sqrt(np.abs(x_))) - np.square(x - 500) / (10000 * len(o))
        c = x * np.sin(np.sqrt(np.abs(x)))

        a = a * (x < -500)
        b = b * (x > 500)
        c = c * (np.abs(x) <= 500)

        return a + b + c
    
    def __function(x: np.ndarray):
        z = 10 * (x - o)
        z = np.matmul(lambda_, x) + 420.968746227503
        res = 418.9829 * z.shape[0] - np.sum(__g(z), axis=0)

        return res

    return __function


def rotated_schwefel(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))
    
    def __g(x):
        x_ = np.mod(np.abs(x), 500) - 500
        a = x_ * np.sin(np.sqrt(np.abs(x_))) - np.square(x + 500) / (10000 * len(o))
        b = -x_ * np.sin(np.sqrt(np.abs(x_))) - np.square(x - 500) / (10000 * len(o))
        c = x * np.sin(np.sqrt(np.abs(x)))

        a = a * (x < -500)
        b = b * (x > 500)
        c = c * (np.abs(x) <= 500)

        return a + b + c
    
    def __function(x: np.ndarray):
        z = 10 * (x - o)
        z = np.matmul(lambda_, np.matmul(m, x)) + 420.968746227503
        res = 418.9829 * z.shape[0] - np.sum(__g(z), axis=0)

        return res

    return __function


def rotated_katsuura(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    m2 = kwargs["m2"]
    j = np.arange(1, 33)
    i_ = np.arange(1, len(o) + 1)
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 100 ** (i / (2 * (len(o) - 1)))
    
    def __function(x: np.ndarray):
        z = np.matmul(m, (x - o) / 20)
        z = np.matmul(lambda_, z)
        z = np.matmul(m2, z)
        D = z.shape[0]
        term = i_ * np.sum(np.abs((2 ** j)[:, np.newaxis] * z - np.round((2 ** j)[:, np.newaxis] * z)) / (2 ** j)[:, np.newaxis], axis=0)
        term = (1 + term) ** (10 / (D ** 1.2))
        res = (10 / (D ** 2)) * np.prod(term) - 10 / (D ** 2)

        return res
    
    return __function



def lunacek_bi_rastrigin(o: np.ndarray, mu0: float = 2.5, d: float = 1, **kwargs):
    def __function(x: np.ndarray):
        z = x - o
        D = z.shape[0]
        mu1 = -np.sqrt(mu0**2 - d)
        term1 = np.sum((z - mu0)**2, axis=0)
        term2 = d * D + np.sum((z - mu1)**2, axis=0)
        term3 = 10 * np.sum(1 - np.cos(2 * np.pi * (z - mu0)), axis=0)
        res = np.minimum(term1, term2) + term3
        return res
    return __function


def rotated_lunacek_bi_rastrigin(o: np.ndarray, mu0: float = 2.5, d: float = 1, **kwargs):
    m = kwargs["m"]
    
    def __function(x: np.ndarray):
        z = np.dot(m, x - o)
        D = z.shape[0]
        mu1 = -np.sqrt(mu0**2 - d)
        term1 = np.sum((z - mu0)**2, axis=0)
        term2 = d * D + np.sum((z - mu1)**2, axis=0)
        term3 = 10 * np.sum(1 - np.cos(2 * np.pi * (z - mu0)), axis=0)
        res = np.minimum(term1, term2) + term3
        return res
    return __function


def expanded_griewank_rosenbrock(o: np.ndarray, **kwargs):
    def __function(x: np.ndarray):
        z = x - o + 1
        term1 = 100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2
        res = np.sum(term1 / 4000 - np.prod(np.cos(term1 / np.sqrt(np.arange(1, term1.size + 1)))) + 1)
        return res
    return __function


def rotated_expanded_schaffer_f6(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    
    def __function(x: np.ndarray):
        z = np.dot(m, x - o)
        term = 0.5 + (np.sin(np.sqrt(z[:-1]**2 + z[1:]**2))**2 - 0.5) / (1 + 0.001 * (z[:-1]**2 + z[1:]**2))
        res = np.sum(term, axis=0)
        return res
    return __function


# composition functions:

def calculate_weights(z: np.ndarray, o: np.ndarray, sigma: float):
    w = np.exp(-np.sum((z - o)**2, axis=0) / (2 * sigma**2))
    w /= np.sum(w)
    return w


def composition_function_any(sigmas: list, lambdas: list, biases: list, functions: list):
    functions = functions
    
    def __composition_function(o: np.ndarray, m: list):
        functions = [func(o=o, m=m[i]) for i, func in enumerate(functions)]

        def __function(x: np.ndarray):
            results = []
            for i in range(len(functions)):
                z = (x - o[i]) / lambdas[i]
                z = np.dot(m[i], z)
                f_i = functions[i](z)
                results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))

            res = np.sum(results, axis=0)
            return res
        
        return __function
    
    return __composition_function


def composition_function_1(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_2(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_3(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_4(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_5(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_6(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_7(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function


def composition_function_8(o: np.ndarray, M: list, sigmas: list, lambdas: list, biases: list, functions: list):
    def __function(x: np.ndarray):
        results = []
        for i in range(len(functions)):
            z = (x - o[i]) / lambdas[i]
            z = np.dot(M[i], z)
            f_i = functions[i](z)
            results.append((f_i + biases[i]) * calculate_weights(z, o[i], sigmas[i]))
        res = np.sum(results, axis=0)
        return res
    return __function

if __name__ == "__main__":
    k = np.arange(1, 21)
    a = 1
    b = 1
    c = np.arange(3)

    print(np.sum((a ** k)[:, np.newaxis] * np.sin((b ** k)[:, np.newaxis] * c)))