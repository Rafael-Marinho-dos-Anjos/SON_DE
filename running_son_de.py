import os
import json

import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm

from src.son_de.son_de_model import SON_DE
from src.utils.initialization import random as initialization # Seleção do método de inicialização dos pesos do SOM
from src.benchmark.functions import schwefel as fit_function # Seleção da função de fitness

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
SOM_EPOCHS = 10
delta = 1
r_min_max = (5, 70)

for run in range(30):
    best_fit = None
    runs[func_name].append({
        "fitness": list(),
        "epoch": list()
    })

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
    GEN_MAX = 3000
    for gen in tqdm(range(GEN_MAX)):
        son_de.new_gen(SOM_EPOCHS, delta, r_min_max)

        best = son_de.best_individual_index()
        fit = son_de.get_pop_fitness()
        
        if best_fit is None or fit[best] < best_fit:
            runs[func_name][-1]["fitness"].append(fit[best] - opt_fit)
            runs[func_name][-1]["epoch"].append(gen + 1)
            best_fit = fit[best]

    print(f"\nrun [{run + 1}/30]\tError: {fit[best] - opt_fit}")

with open(PATH, "w") as file:
    file.write(json.dumps(runs))
