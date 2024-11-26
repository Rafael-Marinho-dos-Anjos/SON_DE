import numpy as np
import matplotlib.pyplot as plt

from src.son_de.son_de_model import SON_DE
from src.utils.initialization import random
from src.utils.mutation import de_rand_1
from src.utils.crossing import binary


# Definição de uma função de aptidão de uma hiperesfera de raio 3
def fitness(input):
    return np.abs(np.sum(input ** 2) - 9)

# DE com 20 induvíduos
son_de = SON_DE(
    dim=3,
    NP=64,
    fitness_func=fitness,
    topology_shape=(8, 8),
    init_method=lambda x: random(x, [[-20, 20]]),
    is_maximization=False
)

# Ajuste da população
GEN_MAX = 100
for gen in range(GEN_MAX):
    son_de.new_gen(10, 0.5, (10, 25))
    
    # Visualização dos resultados
    best = son_de.best_individual_index()
    pop = son_de.get_population()
    fit = son_de.get_pop_fitness()

    print(f"Melhor Individuo: {pop[best]}\tFitness: {fit[best]}")
