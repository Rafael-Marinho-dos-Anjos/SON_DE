import numpy as np


def t_asy(x: np.ndarray, beta):
    D = x.shape[-1]
    i = np.arange(1, D + 1)
    x_ = np.pow(x, (1 + np.sqrt(np.abs(x)) * beta * (i - 1) / (D - 1)))

    x_[x <= 0] = x[x <= 0]

    return x_


def t_osz(x: np.ndarray):
    D = x.shape[-1]
    x_ = np.zeros(D)
    x_[x != 0] = np.log(np.abs(x))[x != 0]

    c_1 = np.ones(D) * 5.5
    c_1[x > 0] = c_1[x > 0] + 4.5

    c_2 = np.ones(D) * 3.1
    c_2[x > 0] = c_2[x > 0] + 4.8

    res = np.sign(x) * np.exp(x_ + 0.049 * (np.sin(c_1 * x_) + np.sin(c_2 * x_)))

    return res


def sphere(o: np.ndarray, **kwargs):
    def __function(x: np.ndarray):
        z = (x - o).astype(np.float64)
        res = np.sum(z ** 2, axis=0)

        return res
    
    return __function


def rotated_high_conditioned_elliptic(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    d = o.shape[-1]

    def __function(x: np.ndarray):
        z = np.matmul(m, (x - o)).astype(np.longdouble)
        z = t_osz(z)
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
    m2 = kwargs["m2"]
    
    def __function(x: np.ndarray):
        z = np.matmul(m, x - o)
        z = np.matmul(m2, t_asy(z, 0.5))
        res = z[0] ** 2 + 10 ** 6 * np.sum(z[1:] ** 2, axis=0)

        return res

    return __function


def rotated_discus(o: np.ndarray, **kwargs):
    m = kwargs["m"].astype(np.longdouble)
    
    def __function(x: np.ndarray):
        z = np.dot(m, x - o).astype(np.longdouble)
        z = t_osz(z)
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
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        z = 5.12 * (x - o) / 100
        z = t_asy(t_osz(z), 0.2)
        z = np.matmul(lambda_, z)

        res = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10, axis=0)

        return res

    return __function


def rotated_rastrigin(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    m2 = kwargs["m2"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        z = np.matmul(m, 5.12 * (x - o) / 100)
        z = np.matmul(m2, t_asy(t_osz(z), 0.2))
        z = np.matmul(lambda_, z)
        z = np.matmul(m, z)
        
        res = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10, axis=0)

        return res

    return __function


def non_continuous_rotated_rastrigin(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    m2 = kwargs["m2"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 10 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        z = np.matmul(m, 5.12 * (x - o) / 100)
        z[z > 0.5] = (np.round(2 * z) / 2)[z > 0.5]
        z = np.matmul(m2, t_asy(t_osz(z), 0.2))
        z = np.matmul(lambda_, z)
        z = np.matmul(m, z)
        
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



def lunacek_bi_rastrigin(o: np.ndarray, **kwargs):
    mu_0 = 2.5
    D = o.shape[-1]
    d = 1
    s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu_1 = -np.sqrt((mu_0 ** 2 - d) / s)
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 100 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        x = x.astype(np.longdouble)
        y = (x - o) / 10
        x_ = 2 * np.sign(x) * y + mu_0
        z = np.matmul(lambda_, x_ - mu_0)

        res = np.min(
            [
                np.sum((x_ - mu_0) ** 2),
                d * D + s * np.sum((x_ - mu_1) ** 2)
            ]
        ) + 10 * (D - np.sum(np.cos(2 * np.pi * z)))

        return res

    return __function


def rotated_lunacek_bi_rastrigin(o: np.ndarray, **kwargs):
    mu_0 = 2.5
    D = o.shape[-1]
    d = 1
    s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu_1 = -np.sqrt((mu_0 ** 2 - d) / s)
    m = kwargs["m"]
    m2 = kwargs["m2"]
    lambda_ = np.identity(len(o))
    for i in range(len(o)):
        lambda_[i, i] = 100 ** (i / (2 * (len(o) - 1)))

    def __function(x: np.ndarray):
        x = x.astype(np.longdouble)
        y = (x - o) / 10
        x_ = 2 * np.sign(x) * y + mu_0
        z = np.matmul(m2, np.matmul(lambda_, np.matmul(m, x_ - mu_0)))

        res = np.min(
            [
                np.sum((x_ - mu_0) ** 2),
                d * D + s * np.sum((x_ - mu_1) ** 2)
            ]
        ) + 10 * (D - np.sum(np.cos(2 * np.pi * z)))

        return res

    return __function


def rotated_expanded_griewank_rosenbrock(o: np.ndarray, **kwargs):
    m = kwargs["m"]

    g_1 = lambda x: np.square(x) / 4000 - np.cos(x) + 1
    g_2 = lambda x: np.sum((x[:, :-1] ** 2 - x[:, 1:]) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)

    def __function(x: np.ndarray):
        z = np.matmul(m, (x - o) / 20 + 1)
        z_ = np.stack((z, np.roll(z, -1)), axis=1)
        
        res = np.sum(g_1(g_2(z_)))

        return res

    return __function


def rotated_expanded_schaffer_f6(o: np.ndarray, **kwargs):
    m = kwargs["m"]
    m2 = kwargs["m2"]
    f_6 = lambda x, y: 0.5 + (np.sin(np.sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) \
        / np.square(1 + 0.001 * (x ** 2 + y ** 2))
    
    def __function(x: np.ndarray):
        z = np.matmul(m, x - o)
        z = np.matmul(m2, t_asy(z, 0.5))
        z_ = np.roll(z, -1)
        
        res = np.sum(f_6(z, z_))

        return res

    return __function


# composition functions:

def calculate_weights(x: np.ndarray, o: np.ndarray, sigma: float):
    if (x != o).sum() == 0:
        return 1
    
    D = o.shape[0]
    w = np.exp(-np.sum((x - o)**2, axis=0) / (2 * D * sigma**2))
    w = w / np.sqrt(np.sum((x - o)**2))
    w /= np.sum(w)

    return w


def composition_function_any(sigmas: list, lambdas: list, biases: list, functions: list):
    __functions = functions
    def __composition_function(o: np.ndarray, **kwargs):
        functions = [func(o=o, **kwargs) for i, func in enumerate(__functions)]

        def __function(x: np.ndarray):
            results = []
            w_acc = 0
            for i in range(len(functions)):
                f_i = functions[i](x)
                w = calculate_weights(x, o, sigmas[i])
                w_acc += w
                results.append((lambdas[i] * f_i + biases[i]) * w)

            res = np.sum(results, axis=0) / w_acc
            return res
        
        return __function
    
    return __composition_function


composition_function_1 = composition_function_any(
    sigmas=[10, 20, 30, 40, 50],
    lambdas=[1, 1e-6, 1e-26, 1e-6, 0.1],
    biases=[0, 100, 200, 300, 400],
    functions=[
        rotated_rosenbrock,
        different_powers,
        rotated_bent_cigar,
        rotated_discus,
        sphere
    ]
)

composition_function_2 = composition_function_any(
    sigmas=[20, 20, 20],
    lambdas=[1, 1, 1],
    biases=[0, 100, 200],
    functions=[
        schwefel,
        schwefel,
        schwefel
    ]
)

composition_function_3 = composition_function_any(
    sigmas=[20, 20, 20],
    lambdas=[1, 1, 1],
    biases=[0, 100, 200],
    functions=[
        rotated_schwefel,
        rotated_schwefel,
        rotated_schwefel
    ]
)

composition_function_4 = composition_function_any(
    sigmas=[20, 20, 20],
    lambdas=[0.25, 1, 0.25],
    biases=[0, 100, 200],
    functions=[
        rotated_schwefel,
        rotated_rastrigin,
        rotated_weierstrass
    ]
)

composition_function_5 = composition_function_any(
    sigmas=[10, 30, 50],
    lambdas=[0.25, 1, 0.25],
    biases=[0, 100, 200],
    functions=[
        rotated_schwefel,
        rotated_rastrigin,
        rotated_weierstrass
    ]
)

composition_function_6 = composition_function_any(
    sigmas=[10, 10, 10, 10, 10],
    lambdas=[ 0.25, 1, 1e-7, 2.5, 10],
    biases=[0, 100, 200, 300, 400],
    functions=[
        rotated_schwefel,
        rotated_rastrigin,
        rotated_high_conditioned_elliptic,
        rotated_weierstrass,
        rotated_griewank
    ]
)

composition_function_7 = composition_function_any(
    sigmas=[10, 10, 10, 10, 10],
    lambdas=[100, 10, 2.5, 25, 0.1],
    biases=[0, 100, 200, 300, 400],
    functions=[
        rotated_griewank,
        rotated_rastrigin,
        rotated_schwefel,
        rotated_weierstrass,
        sphere,
    ]
)

composition_function_8 = composition_function_any(
    sigmas=[10, 20, 30, 40, 50],
    lambdas=[2.5, 2.5e-3, 2.5, 5e-4,0.1],
    biases=[0, 100, 200, 300, 400],
    functions=[
        rotated_griewank,
        rotated_schaffer_f7,
        rotated_schwefel,
        rotated_expanded_schaffer_f6,
        sphere,
    ]
)


if __name__ == "__main__":
    k = np.arange(1, 21)
    a = 1
    b = 1
    c = np.arange(3)

    print(np.sum((a ** k)[:, np.newaxis] * np.sin((b ** k)[:, np.newaxis] * c)))