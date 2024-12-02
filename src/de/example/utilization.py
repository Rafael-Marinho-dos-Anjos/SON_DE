import numpy as np
import matplotlib.pyplot as plt

from src.de.de_model import DE
from src.utils.initialization import random
from src.utils.mutation import de_rand_1
from src.utils.crossing import binary
from src.benchmark.functions import sphere


dim = 20
# DE com 20 induvíduos
de = DE(
    dim=dim,
    NP=100,
    init_method=lambda x: random(x, [[-80, 80]]),
    is_maximization=False
)

global_optmium = np.random.randint(-80, 80, size=dim)

# Definindo a distância utilizada como Manhattan,
# uma função de vizinhança gaussiana e um passo
# de aprendizado inicial de 0.5
cr_0 = 0.9
f_0 = 0.5
de.attach(
    {
        "mutation": de_rand_1,
        "crossing": binary(cr_0),
        "fitness": sphere(global_optmium),
        "F": f_0
    }
)

# Ajuste da população
GEN_MAX = 1000
for gen in range(GEN_MAX):
    de.new_generation()
    
    # Visualização dos resultados
    best = de.best_individual_index()
    pop = de.get_population()
    fit = de.get_pop_fitness()

    print(f"t: {gen+1}\tMelhor Individuo: {pop[best]}\tFitness: {float(fit[best]):.2f}")

print(f"Global optimum: {global_optmium}")

