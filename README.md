# snake-nn

Training a neural network to play Snake using a genetic algorithm (neuroevolution).

A population of randomly-initialized networks each play the game, the best performers are selected, combined, and mutated, and the process repeats over many generations until an agent learns to survive and chase food.

## Project layout

| File | Purpose |
| --- | --- |
| [game.py](game.py) | `SnakeGame` — game logic, state encoding, and reward signal |
| [network.py](network.py) | `NeuralNetwork` — feedforward net mapping game state → action |
| [ga.py](ga.py) | Genetic algorithm: evaluation, selection, crossover, mutation, save/load |
| [main.py](main.py) | CLI entry point — `train` and `play` commands |
| [visualize.py](visualize.py) | Renders a trained agent with pygame |
| `best_net.npz` | Weights of the best network found so far (written during training) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pygame
```

`numpy` is required for everything; `pygame` is only needed for the `play` command.

> **Note:** on newer Python versions (e.g. 3.14), `pygame` may not yet publish a
> prebuilt wheel, causing `pip install` to fail trying to build it from source.
> If that happens, install [`pygame-ce`](https://pyga.me/) instead — it's an
> actively maintained, drop-in-compatible fork that still imports as `pygame`:
> ```bash
> pip install numpy pygame-ce
> ```

## Usage

The project is driven through [main.py](main.py), which has two subcommands.

### Train

Runs the genetic algorithm. Fitness is evaluated in parallel across CPU cores. Whenever a new best network is found it is saved to `--out` (default `best_net.npz`).

```bash
python main.py train
```

Options (defaults shown):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--generations` | `250` | Number of generations to evolve |
| `--pop` | `250` | Population size |
| `--elite` | `10` | Top networks copied unchanged into the next generation |
| `--k` | `5` | Tournament size for parent selection |
| `--mutation-rate` | `0.05` | Fraction of weights perturbed per child |
| `--mutation-strength` | `0.3` | Std-dev of the mutation noise |
| `--out` | `best_net.npz` | Where to save the best network |

Example — a shorter run with a larger population:

```bash
python main.py train --generations 100 --pop 500 --out my_net.npz
```

Each generation prints its best and average fitness, and announces a new record when one is saved.

### Play

Watch a saved network play in a pygame window.

```bash
python main.py play
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--net` | `best_net.npz` | Network file to load |
| `--fps` | `60` | Playback speed |

Press **R** in the window to start a new game. Example:

```bash
python main.py play --net my_net.npz --fps 15
```
