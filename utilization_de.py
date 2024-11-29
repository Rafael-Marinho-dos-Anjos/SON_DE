import numpy as np
import matplotlib.pyplot as plt

from src.de.de_model import DE
from src.utils.initialization import lhs
from src.utils.mutation import de_rand_1
from src.utils.crossing import binary
from src.benchmark.functions import different_powers


# Função de aptidão, ótimo global em [51, 74, -3]
fitness = different_powers(np.array([51, 74, -3]))

# DE com 100 induvíduos, utilizada a distribuição do CEC [-80, 80]
de = DE(
    dim=3,
    NP=100,
    init_method=lambda x: lhs(x, [[-80, 80]]),
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
GEN_MAX = 1000
for gen in range(GEN_MAX):
    de.new_generation()

    f = f_0 * np.exp(-gen/100)

    de.attach(
        {
            "F": f
        }
    )
    
    # Visualização dos resultados
    if (gen + 1) % 25 == 0:
        best = de.best_individual_index()
        pop = de.get_population()
        fit = de.get_pop_fitness()

        print(f"Ger: {gen+1}  Melhor Individuo: {pop[best]}\tFitness: {fit[best]}")
