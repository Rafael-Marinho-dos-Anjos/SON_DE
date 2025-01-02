import json
import numpy as np


with open("src/proposed_som_de/runs/runs.json", "r") as file:
    runs = json.loads(file.read())

gambiarra = 10 ** 100

for i in range(1, 29):
    func = "F" + str(i)
    if func not in runs.keys():
        continue

    best_fits = list()
    for run in runs[func]:
        best_fits.append(run["fitness"][-1])

    best_fits = np.array(best_fits, dtype=np.float64)
    print(f"{func} -> mean: {best_fits.mean()}    \tstd: {best_fits.std()}")
