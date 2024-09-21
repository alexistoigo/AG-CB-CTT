import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt  # Importação do matplotlib


# Parâmetros do problema
NUM_AULAS = 100       # Número total de aulas
NUM_SALAS = 10        # Número total de salas
NUM_PERIODOS = 20     # Número total de períodos
CAPACIDADE_SALAS = [30 for _ in range(NUM_SALAS)]  # Capacidade de cada sala
ALUNOS_POR_AULA = [random.randint(10, 50) for _ in range(NUM_AULAS)]  # Número de alunos por aula

# Criação dos tipos básicos
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

# Função para criar um indivíduo
def cria_individuo():
    # Cada gene representa a atribuição de uma aula a uma sala e período
    return [random.randrange(NUM_SALAS * NUM_PERIODOS) for _ in range(NUM_AULAS)]

# Função para avaliar um indivíduo
def avalia_individuo(individuo):
    penalidade = 0
    horario_ocupado = {}
    for idx, gene in enumerate(individuo):
        sala = gene // NUM_PERIODOS
        periodo = gene % NUM_PERIODOS

        # Verifica se a capacidade da sala é suficiente
        if ALUNOS_POR_AULA[idx] > CAPACIDADE_SALAS[sala]:
            penalidade += 10  # Penalidade por violação de capacidade

        # Verifica se a sala já está ocupada no período
        if (sala, periodo) in horario_ocupado:
            penalidade += 100  # Penalidade alta por conflito de sala
        else:
            horario_ocupado[(sala, periodo)] = idx

        # Verifica se há conflito de professor (supondo que cada aula tem um professor único)
        # Aqui simplificado, mas pode ser estendido com dados reais

    # Penalidades adicionais podem ser adicionadas aqui (restrições suaves)
    return (penalidade,)


def gera_grafico_convergencia(logbook):
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    
    plt.figure(figsize=(10, 5))
    plt.plot(gen, min_fitness, label="Penalidade Mínima")
    plt.plot(gen, avg_fitness, label="Penalidade Média")
    plt.xlabel("Geração")
    plt.ylabel("Penalidade")
    plt.title("Curva de Convergência do Algoritmo Genético")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("curva_convergencia.png")  # Salva o gráfico como uma imagem
    plt.show()  # Exibe o gráfico na tela


# Registro dos operadores genéticos
toolbox = base.Toolbox()
toolbox.register("individuo", tools.initIterate, creator.Individuo, cria_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individuo)
toolbox.register("evaluate", avalia_individuo)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_SALAS*NUM_PERIODOS-1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    populacao = toolbox.population(n=100)
    NGEN = 500
    HOF = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Algoritmo evolutivo simples
    populacao, logbook = algorithms.eaSimple(populacao, toolbox, cxpb=0.8, mutpb=0.1,
                                             ngen=NGEN, stats=stats, halloffame=HOF, verbose=True)

    # Resultados
    melhor_individuo = HOF[0]
    print("Melhor solução encontrada:")
    print("Penalidade:", melhor_individuo.fitness.values[0])

    # Decodificação da solução
    for idx, gene in enumerate(melhor_individuo):
        sala = gene // NUM_PERIODOS
        periodo = gene % NUM_PERIODOS
        print(f"Aula {idx}: Sala {sala}, Período {periodo}")


    gera_grafico_convergencia(logbook)


if __name__ == "__main__":
    main()
