# snake-nn

Training a neural network to play Snake using a genetic algorithm (neuroevolution).

Instead of learning through backpropagation, a population of networks plays the game and the best-performing ones are selected, combined, and mutated over successive generations.

## Project layout

| File | Purpose | Status |
| --- | --- | --- |
| [game.py](game.py) | `SnakeGame` — game logic, state encoding, and reward signal | ✅ implemented |
| [network.py](network.py) | Neural network that maps game state → action | 🚧 stub |
| [ga.py](ga.py) | Genetic algorithm: population, selection, crossover, mutation | 🚧 stub |
| [main.py](main.py) | Entry point — runs the training loop | 🚧 stub |
| [visualize.py](visualize.py) | Render a trained agent playing | 🚧 stub |

## The game

`SnakeGame` runs on a configurable square grid (default 20×20).

**Actions** are relative to the snake's current heading: `"left"`, `"right"`, or `"straight"` (any other value keeps the direction).

**State** (`get_state`) is an 11-element binary vector:

- Danger straight / left / right (collision one step ahead in each direction)
- Current direction (one-hot over the 4 directions)
- Food location relative to the head (left, right, up, down)

**Rewards** (`step`):

- `+10` for eating food
- `-10` for hitting a wall or itself
- `-10` if the episode runs too long without progress
- `0` otherwise

## Getting started

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy
```

### Quick sanity check

```python
from game import SnakeGame

env = SnakeGame(grid_size=20)
state = env.get_state()
reward, done = env.step("straight")
print(state, reward, done)
```

## Roadmap

- [ ] Implement the feedforward network in `network.py`
- [ ] Implement the genetic algorithm in `ga.py`
- [ ] Wire up the training loop in `main.py`
- [ ] Add visualization of a trained agent in `visualize.py`
