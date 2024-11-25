import numpy as np
import matplotlib.pyplot as plt

from src.som.som_model import SOM
from src.utils.initialization import random
from src.utils.distances import manhattan
from src.utils.neighborhood import gaussian


# Dados com 2 dimensões
data = np.random.rand(200 * 2).reshape((200, 2))

# SOM com topologia retangular 3x3
som = SOM(2, (3, 3), random)

# Definindo a distância utilizada como Manhattan,
# uma função de vizinhança gaussiana e um passo
# de aprendizado inicial de 0.5
sigma = 1
alpha = 0.5
som.attach(
    {
        "distance": manhattan,
        "neighborhood": gaussian(1),
        "alpha": alpha
    }
)

EPOCHS = 10

for epoch in range(EPOCHS):
    for x in data:
        som.update(x)
    
    # Atualização do alcance da vizinhança e do passo de aprendizagem
    sigma = np.exp(-epoch)
    alpha = 0.5 * np.exp(-epoch)

    som.attach(
        {
            "neighborhood": gaussian(1),
            "alpha": alpha
        }
    )


# Visualização dos agrupamentos
groups = {}
for x in data:
    bmu = som.bmu(x)
    bmu = str(bmu)

    if bmu in groups.keys():
        groups[bmu].append(x)
    else:
        groups[bmu] = [x]

fig, ax = plt.subplots()
for cluster in groups.keys():
    p = np.array(groups[cluster])
    ax.scatter(p[:, 0], p[:, 1], label=cluster)

ax.legend()
plt.show()
