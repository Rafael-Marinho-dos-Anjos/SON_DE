import json
import numpy as np


with open("src/son_de/runs/runs.json", "r") as file:
    runs = json.loads(file.read())

for func in runs.keys():
    best_fits = list()
    for run in runs[func]:
        best_fits.append(run["fitness"][-1])
    
    best_fits = np.array(best_fits)
    print(f"{func} -> mean: {best_fits.mean()}\tstd: {best_fits.std()}")
