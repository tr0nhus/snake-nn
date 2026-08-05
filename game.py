import numpy as np
import random

class SnakeGame:
    DIRECTIONS = [(0,-1), (1, 0), (0, 1), (-1, 0)]
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # new snake, new food, reset score
        self.snake = [(10,10)]
        self.direction = (0, 1)
        self.food = self._spawn_food()
        self.score = 0
        self.done = False
        self.step_count = 0
        
    def step(self, action):
        # move head, check collision, update state
        idx = self.DIRECTIONS.index(self.direction)
        if action == "right":
            idx = (idx + 1) % 4
        elif action == "left":
            idx = (idx - 1) % 4
        self.direction = self.DIRECTIONS[idx]

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if self._is_collision(new_head):
            self.done = True
            return -10, True

        self.step_count += 1
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            reward = 10
        else:
            self.snake.pop()
            reward = 0

        if self.step_count > 100 * 50 + len(self.snake):
            self.done = True
            reward =  -10

        return reward, self.done

    def get_state(self):
        # return numpy array with game state
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        idx = self.DIRECTIONS.index(self.direction)
        left_idx = (idx - 1) % 4
        left_x, left_y = self.DIRECTIONS[left_idx]
        right_idx = (idx + 1) % 4
        right_x, right_y = self.DIRECTIONS[right_idx]

        straight = (head_x + dir_x, head_y + dir_y)
        left = (head_x + left_x, head_y + left_y)
        right = (head_x + right_x, head_y + right_y)

        state = [
            self._is_collision(straight),
            self._is_collision(left),
            self._is_collision(right),

            self.direction == self.DIRECTIONS[0],
            self.direction == self.DIRECTIONS[1],
            self.direction == self.DIRECTIONS[2],
            self.direction == self.DIRECTIONS[3],

            self.food[0] < head_x,
            self.food[0] > head_x,
            self.food[1] > head_y,
            self.food[1] < head_y
        ]

        return np.array(state, dtype=int)
    
    def _spawn_food(self):
        # spawn food somewhere the snake is not and return coords
        food = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        while food in self.snake:
            food = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        return food

    def _is_collision(self, pos):
        # check for collision and bounds
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False
