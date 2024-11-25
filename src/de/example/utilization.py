import numpy as np
import matplotlib.pyplot as plt

from src.de.de_model import DE
from src.utils.initialization import random
from src.utils.mutation import de_rand_1
from src.utils.crossing import binary


# Definição de uma função de aptidão de uma hiperesfera de raio 3
def fitness(input):
    return np.abs(np.sum(input ** 2) - 9)

# DE com 20 induvíduos
de = DE(
    dim=3,
    NP=50,
    init_method=lambda x: random(x, [[-20, 20]]),
    is_maximization=False
)

# Definindo a distância utilizada como Manhattan,
# uma função de vizinhança gaussiana e um passo
# de aprendizado inicial de 0.5
cr_0 = 0.5
f_0 = 0.9
de.attach(
    {
        "mutation": de_rand_1,
        "crossing": binary(cr_0),
        "fitness": fitness,
        "F": f_0
    }
)

# Ajuste da população
GEN_MAX = 25
for gen in range(GEN_MAX):
    de.new_generation()

    cr = cr_0 * np.exp(-gen)
    f = f_0 * np.exp(-gen)

    de.attach(
        {
            "crossing": binary(cr_0),
            "F": f_0
        }
    )
    
    # Visualização dos resultados
    best = de.best_individual_index()
    pop = de.get_population()
    fit = de.get_pop_fitness()

    print(f"Melhor Individuo: {pop[best]}\tFitness: {fit[best]}")
