import numpy as np

from multiprocessing import Pool
from game import SnakeGame
from network import NeuralNetwork

game = SnakeGame()
net = NeuralNetwork()

def evaluate(net, game, n_games=3):
    total = 0
    for i in range(n_games):
        game.reset()
        steps = 0
        while not game.done:
            state = game.get_state()
            action = net.get_action(state)
            game.step(action)
            steps += 1
        total += steps + (game.score ** 2) * 100
    return total / n_games

def evaluate_worker(net):
    game = SnakeGame()
    return evaluate(net, game)

def tournament_select(population, fitnesses, k=5):
    idxs = np.random.choice(len(population), k, replace=False)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]

def crossover(parent_a, parent_b):
    child_weights = []
    child_biases = []
    for wa, wb in zip(parent_a.weights, parent_b.weights):
        mask = np.random.rand(*wa.shape) < 0.5
        child_weights.append(np.where(mask, wa, wb))
    for ba, bb in zip(parent_a.biases, parent_b.biases):
        mask = np.random.rand(*ba.shape) < 0.5
        child_biases.append(np.where(mask, ba, bb))
    return NeuralNetwork(weights=(child_weights, child_biases))

def mutate(net, rate=0.05, strength=0.3):
    for w in net.weights:
        mask = np.random.rand(*w.shape) < rate
        w += mask * np.random.randn(*w.shape) * strength
    for b in net.biases:
        mask = np.random.rand(*b.shape) < rate
        b += mask * np.random.randn(*b.shape) * strength

def save_network(net, path="best_net.npz"):
    arrays = {}
    for i, w in enumerate(net.weights):
        arrays[f"w{i}"] = w
    for i, b in enumerate(net.biases):
        arrays[f"b{i}"] = b
    np.savez(path, **arrays)

def load_network(path="best_net.npz", layer_sizes=[11, 16, 16, 3]):
    data = np.load(path)
    n = len(layer_sizes) - 1
    weights = [data[f"w{i}"] for i in range(n)]
    biases = [data[f"b{i}"] for i in range(n)]
    return NeuralNetwork(layer_sizes=layer_sizes, weights=(weights, biases))

# if __name__ == "__main__":

#     POP_SIZE = 300
#     GENERATIONS = 500
#     ELITE = 10

#     population = [NeuralNetwork() for _ in range(POP_SIZE)]
#     game = SnakeGame()

#     best_ever = -1
#     with Pool() as pool:
#         for gen in range(GENERATIONS):
#             fitnesses = pool.map(evaluate_worker, population)
#             ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])

#             if ranked[0][0] > best_ever:
#                 best_ever = ranked[0][0]
#                 save_network(ranked[0][1], "best_net.npz")
#                 print(f"  New record: {best_ever:.0f} — saved")

#             new_pop = [net for _, net in ranked[:ELITE]]

#             while len(new_pop) < POP_SIZE:
#                 parent_a = tournament_select(population, fitnesses)
#                 parent_b = tournament_select(population, fitnesses)
#                 child = crossover(parent_a, parent_b)
#                 mutate(child)
#                 new_pop.append(child)

#             population = new_pop

#             best_score = max(fitnesses)
#             avg = sum(fitnesses) / len(fitnesses)
#             print(f"Gen {gen}: best={best_score:.0f} avg={avg:.0f}")