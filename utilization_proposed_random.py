import numpy as np
import matplotlib.pyplot as plt

from src.proposed_som_de.proposed_model import Model
from src.utils.initialization import random as initialization # Seleção do método de inicialização dos pesos do SOM
from src.benchmark.functions import sphere as fit_function # Seleção da função de fitness
from src.utils.crossing import binary_random
from src.utils.mutation import proposed_f_rand_1


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
delta = 1
r_min_max = (5, 70)

acc_mat_list = list()
fits = [list() for _ in range(30)]

for run in range(30):
    model = Model(
        dim=30,
        NP=100,
        fitness_func=fitness,
        topology_shape=(4, 4),
        init_method=lambda x: initialization(x, [[-100, 100]]),
        is_maximization=False,
        sigma0=sigma_0,
        tau0=tau_0,
        adjust=update_adjust
    )


    f = (0.2, 0.8)
    cr = (0.1, 0.9)
    model.attach_de(
        {
            "mutation": proposed_f_rand_1,
            "F": f,
            "crossing": binary_random(cr)
        }
    )

    best_fit_per_epoch = []
    # Ajuste da população
    GEN_MAX = 1000
    for gen in range(GEN_MAX):
        som_epochs = 25 if gen == 0 else 10
        model.new_gen(som_epochs, delta, r_min_max)

        best = model.best_individual_index()
        fit = model.get_pop_fitness()
        # Visualização dos resultados
        if (gen + 1) % 1 == 0:
            pop = model.get_population()

            print(f"Running: [{run+1}/30]\tGer: [{gen+1}/1000]\tFitness: {fit[best]}")
            fits[run].append(fit[best])
        
        best_fit_per_epoch.append(fit[best])

        # if fit[best] < 1:
        #     break

    acc_mat_list.append(model.get_acc_matrix())

acc_mat_list = np.stack(acc_mat_list, axis=0)
print(np.mean(acc_mat_list, axis=0))

import matplotlib.pyplot as plt

for i in range(30):
    plt.plot(fits[i])

plt.show()
