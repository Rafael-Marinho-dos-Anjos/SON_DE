import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do algoritmo
D = 30                 # Dimensão (número de variáveis)
NP = 100               # Tamanho da população
F = 0.5                # Fator de amplificação diferencial
CR = 0.9               # Taxa de cruzamento
maxGen = 300           # Número máximo de gerações
bounds = [-100, 100]   # Limites para cada variável

# Função objetivo: Esfera
def sphere(x):
    return np.sum(x ** 2)

# Inicialização da população
pop = np.random.uniform(bounds[0], bounds[1], (NP, D))  # População inicial
fitness = np.array([sphere(ind) for ind in pop])        # Avaliação inicial

# Salva a população inicial para o gráfico
initialPop = pop.copy()

# Loop do algoritmo
for gen in range(maxGen):
    newPop = pop.copy()  # Nova população
    for i in range(NP):
        # 1. Seleção dos índices para mutação
        idxs = np.random.permutation(NP)[:3]  # Seleciona 3 indivíduos aleatórios
        while i in idxs:    # Garante que os índices não incluam o atual
            idxs = np.random.permutation(NP)[:3]
        
        a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

        # 2. Mutação
        mutant = a + F * (b - c)
        mutant = np.clip(mutant, bounds[0], bounds[1])  # Garante que os limites sejam respeitados

        # 3. Cruzamento binomial
        trial = pop[i].copy()  # Copia o vetor alvo
        for j in range(D):
            if np.random.rand() <= CR or j == np.random.randint(D):  # Pelo menos um componente é alterado
                trial[j] = mutant[j]

        # 4. Seleção
        trialFitness = sphere(trial)  # Avaliação do vetor trial
        if trialFitness < fitness[i]:  # Substitui se a solução trial for melhor
            newPop[i] = trial
            fitness[i] = trialFitness

    pop = newPop.copy()  # Atualiza a população para a próxima geração

    # Exibe progresso a cada 10 gerações
    if (gen + 1) % 10 == 0:
        print(f'Geração {gen + 1}: Melhor fitness = {np.min(fitness)}')

# Salva a população final para o gráfico
finalPop = pop

# Resultado final
bestFitness = np.min(fitness)
bestIdx = np.argmin(fitness)
bestSolution = pop[bestIdx]
print('--------------------')
print(f'Melhor fitness: {bestFitness}')

'''
# Gráficos de Dispersão para a Primeira Dimensão
selectedDim = 0  # Dimensão a ser analisada (por exemplo, a primeira variável)

# Gráfico de dispersão para a população inicial
plt.figure()
plt.scatter(np.arange(NP), initialPop[:, selectedDim], 50, c='b', label='Inicial', marker='o')
plt.title(f'Distribuição da Dimensão {selectedDim + 1} - População Inicial')
plt.xlabel('Indivíduos')
plt.ylabel('Valor da Variável')
plt.grid(True)
plt.legend()

# Gráfico de dispersão para a população final
plt.figure()
plt.scatter(np.arange(NP), finalPop[:, selectedDim], 50, c='r', label='Final', marker='x')
plt.title(f'Distribuição da Dimensão {selectedDim + 1} - População Final')
plt.xlabel('Indivíduos')
plt.ylabel('Valor da Variável')
plt.grid(True)
plt.legend()

plt.show()
'''