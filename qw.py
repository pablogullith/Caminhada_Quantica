import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays e operações matemáticas
import matplotlib.pyplot as plt  # Importa a biblioteca Matplotlib para criar gráficos
from scipy.stats import norm  # Importa a função norm do módulo scipy.stats para a distribuição normal

def quantum_walk(t, initial_state):
    size = 2 * t + 1  # Define o tamanho do array para acomodar a caminhada, considerando o intervalo de -t a t
    alpha = np.zeros(size, dtype=complex)  # Inicializa o array alpha com zeros (tipos complexos)
    beta = np.zeros(size, dtype=complex)  # Inicializa o array beta com zeros (tipos complexos)
    
    mid = t  # Define o índice central
    if initial_state == '0':
        alpha[mid] = 1.0  # Define o estado inicial |0> no índice central
    elif initial_state == '1':
        beta[mid] = 1.0  # Define o estado inicial |1> no índice central
    else:
        raise ValueError("O estado inicial deve ser '0' ou '1'.")  # Levanta um erro se o estado inicial não for válido
    
    for _ in range(t):  # Executa o loop por t passos
        # Atualiza alpha e beta usando operações vetorizadas para melhorar a eficiência
        new_alpha = np.roll(alpha, shift=-1) + np.roll(beta, shift=-1)  # Calcula o novo alpha
        new_beta = np.roll(alpha, shift=1) - np.roll(beta, shift=1)  # Calcula o novo beta
        new_alpha /= np.sqrt(2)  # Normaliza o novo alpha
        new_beta /= np.sqrt(2)  # Normaliza o novo beta
        alpha, beta = new_alpha, new_beta  # Atualiza alpha e beta para o próximo passo
    
    # Calcula a distribuição de probabilidade final como a soma dos quadrados das magnitudes de alpha e beta
    prob_distribution_quantum = np.abs(alpha)**2 + np.abs(beta)**2
    prob_distribution_quantum /= prob_distribution_quantum.sum()  # Normaliza a distribuição para que a soma seja 1
    
    return prob_distribution_quantum  # Retorna a distribuição de probabilidade quântica

def classical_walk(t):
    size = 2 * t + 1  # Define o tamanho do array para acomodar a caminhada, considerando o intervalo de -t a t
    positions = np.arange(-t, t+1)  # Cria um array de posições de -t a t
    mean = 0  # Define a média da distribuição normal como 0
    std_dev = np.sqrt(t)  # Define o desvio padrão da distribuição normal como sqrt(t)
    
    # Calcula a distribuição de probabilidade clássica usando a função norm.pdf
    prob_distribution_classical = norm.pdf(positions, loc=mean, scale=std_dev)
    prob_distribution_classical /= prob_distribution_classical.sum()  # Normaliza a distribuição para que a soma seja 1
    
    return prob_distribution_classical  # Retorna a distribuição de probabilidade clássica

# Parâmetros da simulação
t = 100  # Número de passos na caminhada

# Calcula as distribuições de probabilidade
prob_distribution_quantum_0 = quantum_walk(t, '0')  # Caminhada quântica com estado inicial |0>
prob_distribution_quantum_1 = quantum_walk(t, '1')  # Caminhada quântica com estado inicial |1>
prob_distribution_classical = classical_walk(t)  # Caminhada clássica

# Gera o gráfico
positions = np.arange(-t, t+1)  # Cria um array de posições de -t a t
plt.plot(positions, prob_distribution_quantum_0, color='orange', label='Caminhada Quântica (Estado |0>)')  # Plota a distribuição quântica para estado |0>
plt.plot(positions, prob_distribution_quantum_1, color='green', label='Caminhada Quântica (Estado |1>)')  # Plota a distribuição quântica para estado |1>
plt.plot(positions, prob_distribution_classical, color='blue', label='Caminhada Clássica', linestyle='--')  # Plota a distribuição clássica com linha tracejada
plt.title('Distribuição de Probabilidade')  # Define o título do gráfico
plt.xlabel('Posição')  # Define o rótulo do eixo x
plt.ylabel('Probabilidade')  # Define o rótulo do eixo y
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.legend()  # Adiciona uma legenda ao gráfico
plt.show()  # Exibe o gráfico
