import json
import numpy as np


with open("src/proposed_som_de/runs/runs.json", "r") as file:
    runs = json.loads(file.read())

gambiarra = 10 ** 100

for func in runs.keys():
    best_fits = list()
    for run in runs[func]:
        best_fits.append(run["fitness"][-1] * gambiarra)
    
    best_fits = np.array(best_fits, dtype=np.float64)
    print(f"{func} -> mean: {best_fits.mean()/gambiarra}\tstd: {best_fits.std()/gambiarra}")
