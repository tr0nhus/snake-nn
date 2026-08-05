import argparse
import numpy as np
from multiprocessing import Pool

from game import SnakeGame
from network import NeuralNetwork
import ga
import visualize


def cmd_train(args):
    population = [NeuralNetwork() for _ in range(args.pop)]
    best_ever = -1

    with Pool() as pool:
        for gen in range(args.generations):
            fitnesses = pool.map(ga.evaluate_worker, population)

            order = np.argsort(fitnesses)[::-1]
            ranked = [population[i] for i in order]
            best_fitness = fitnesses[order[0]]

            if best_fitness > best_ever:
                best_ever = best_fitness
                ga.save_network(ranked[0], args.out)
                print(f"  ny rekord: {best_ever:.0f} — gemt til {args.out}")

            new_pop = ranked[:args.elite]
            while len(new_pop) < args.pop:
                pa = ga.tournament_select(population, fitnesses, args.k)
                pb = ga.tournament_select(population, fitnesses, args.k)
                child = ga.crossover(pa, pb)
                ga.mutate(child, args.mutation_rate, args.mutation_strength)
                new_pop.append(child)

            population = new_pop

            avg = sum(fitnesses) / len(fitnesses)
            print(f"Gen {gen}: best={best_fitness:.0f} avg={avg:.0f}")


def cmd_play(args):
    visualize.run(net_path=args.net, fps=args.fps)


def main():
    parser = argparse.ArgumentParser(description="Snake NN")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a population with GA")
    p_train.add_argument("--generations", type=int, default=500)
    p_train.add_argument("--pop", type=int, default=300)
    p_train.add_argument("--elite", type=int, default=10)
    p_train.add_argument("--k", type=int, default=5)
    p_train.add_argument("--mutation-rate", type=float, default=0.05)
    p_train.add_argument("--mutation-strength", type=float, default=0.3)
    p_train.add_argument("--out", default="best_net.npz")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play", help="Watch the best network play")
    p_play.add_argument("--net", default="best_net.npz")
    p_play.add_argument("--fps", type=int, default=60)
    p_play.set_defaults(func=cmd_play)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()