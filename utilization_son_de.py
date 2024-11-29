import numpy as np
import matplotlib.pyplot as plt

from src.son_de.son_de_model import SON_DE
from src.utils.initialization import lhs
from src.benchmark.functions import different_powers


# Função de aptidão, ótimo global em [51, 74, -3]
fitness = different_powers(np.array([51, 74, -3]))

# DE com 100 induvíduos, utilizada a distribuição do CEC [-80, 80]
son_de = SON_DE(
    dim=3,
    NP=100,
    fitness_func=fitness,
    topology_shape=(10, 10),
    init_method=lambda x: lhs(x, [[-80, 80]]),
    is_maximization=False
)

f_0 = 0.8
son_de.attach_de(
    {
        "F": f_0
    }
)

# Ajuste da população
GEN_MAX = 1000
for gen in range(GEN_MAX):
    son_de.new_gen(25, 0.5, (5, 20))

    f = f_0 * np.exp(-gen/25)

    son_de.attach_de(
        {
            "F": f
        }
    )

    # Visualização dos resultados
    if (gen + 1) % 25 == 0:
        best = son_de.best_individual_index()
        pop = son_de.get_population()
        fit = son_de.get_pop_fitness()

        print(f"Ger: {gen+1}  Melhor Individuo: {pop[best]}\tFitness: {fit[best]}")
