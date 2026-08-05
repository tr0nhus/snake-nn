import pygame
import numpy as np
from game import SnakeGame
from network import NeuralNetwork

CELL = 25

BG      = (20, 20, 25)
SNAKE   = (80, 200, 120)
HEAD    = (140, 240, 170)
FOOD    = (220, 80, 90)
TEXT    = (230, 230, 235)

def load_network(path="best_net.npz", layer_sizes=[11, 16, 16, 3]):
    data = np.load(path)
    n = len(layer_sizes) - 1
    weights = [data[f"w{i}"] for i in range(n)]
    biases  = [data[f"b{i}"] for i in range(n)]
    return NeuralNetwork(layer_sizes=layer_sizes, weights=(weights, biases))

def run(net_path, fps):
    game = SnakeGame()
    net = load_network(net_path)

    pygame.init()
    size = game.grid_size * CELL
    screen = pygame.display.set_mode((size, size + 40))
    pygame.display.set_caption("Snake NN")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 18)

    game.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()   # tryk R for nyt spil

        if not game.done:
            action = net.get_action(game.get_state())
            game.step(action)

        screen.fill(BG)

        fx, fy = game.food
        pygame.draw.rect(screen, FOOD, (fx * CELL, fy * CELL, CELL - 1, CELL - 1))

        for i, (x, y) in enumerate(game.snake):
            color = HEAD if i == 0 else SNAKE
            pygame.draw.rect(screen, color, (x * CELL, y * CELL, CELL - 1, CELL - 1))

        status = f"SCORE: {game.score}   LENGTH: {len(game.snake)}"
        if game.done:
            status += "   [R new game]"
        screen.blit(font.render(status, True, TEXT), (8, size + 10))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
