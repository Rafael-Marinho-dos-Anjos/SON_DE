import numpy as np
import matplotlib.pyplot as plt

from src.son_de.son_de_model import SON_DE
from src.utils.initialization import random as initialization # Seleção do método de inicialização dos pesos do SOM
from src.benchmark.functions import sphere as fit_function # Seleção da função de fitness


# Número de dimensões do indivíduo
dim=30

# Função de aptidão
global_optimum = np.random.randint(-80, 80, size=30)
fitness = fit_function(global_optimum)
print(f"Ótimo global: {global_optimum}")

def update_adjust(i, t, T, NP):
    return 1 - ((t * NP + i) / (T * NP))

# DE com 100 induvíduos, utilizada a distribuição do CEC [-80, 80]
sigma_0 = 5
tau_0 = 0.9
SOM_EPOCHS = 10
delta = 1
r_min_max = (5, 70)
son_de = SON_DE(
    dim=30,
    NP=100,
    fitness_func=fitness,
    topology_shape=(10, 10),
    init_method=lambda x: initialization(x, [[-100, 100]]),
    is_maximization=False,
    reset_prototypes=True,
    sigma0=sigma_0,
    tau0=tau_0,
    adjust=update_adjust
)


f_0 = 0.5
son_de.attach_de(
    {
        "F": f_0
    }
)

best_fit_per_epoch = []
# Ajuste da população
GEN_MAX = 20000
for gen in range(GEN_MAX):
    son_de.new_gen(SOM_EPOCHS, delta, r_min_max)

    best = son_de.best_individual_index()
    fit = son_de.get_pop_fitness()
    # Visualização dos resultados
    if (gen + 1) % 1 == 0:
        pop = son_de.get_population()

        print(f"Ger: {gen+1}\tFitness: {fit[best]}")
    
    best_fit_per_epoch.append(fit[best])

    if fit[best] < 1:
        break

best_ind = [int(100*char)/100 for char in pop[best]]
print(f"Melhor Individuo: {best_ind}\nÓtimo global: {global_optimum}")
