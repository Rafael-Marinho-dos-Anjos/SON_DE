import os
import json

import numpy as np
import matplotlib.pyplot as plt

from src.proposed_som_de.proposed_model import Model
from src.utils.initialization import random as initialization # Seleção do método de inicialização dos pesos do SOM
from src.benchmark.functions import sphere as fit_function # Seleção da função de fitness
from src.utils.crossing import binary_random
from src.utils.mutation import proposed_f_rand_1


# Número de dimensões do indivíduo
PATH = "src/son_de/runs/runs.json"
if os.path.exists(PATH):
    with open(PATH, "r") as file:
        runs = json.loads(file.read())
else:
    runs = dict()

func_name = "F14"
runs[func_name] = list()

# Número de dimensões do indivíduo
dim=30

m = np.identity(dim)

# Função de aptidão
global_optimum = np.random.randint(-80, 80, size=30)
fitness = fit_function(global_optimum)
opt_fit = fitness(global_optimum)
print(f"Ótimo global: {global_optimum}")
print(f"Aptidão ótima: {opt_fit}")

def update_adjust(i, t, T, NP):
    return 1 - ((t * NP + i) / (T * NP))

# DE com 100 induvíduos, utilizada a distribuição do CEC [-80, 80]
sigma_0 = 5
tau_0 = 0.9
delta = 1
r_min_max = (5, 70)

for run in range(30):
    best_fit = None
    runs[func_name].append({
        "fitness": list(),
        "epoch": list()
    })

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
            adjust=None
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
            
            if best_fit is None or fit[best] < best_fit:
                runs[func_name][-1]["fitness"].append(fit[best] - opt_fit)
                runs[func_name][-1]["epoch"].append(gen + 1)
                best_fit = fit[best]

        print(f"\nrun [{run + 1}/30]\tError: {fit[best] - opt_fit}")

with open(PATH, "w") as file:
    file.write(json.dumps(runs))
