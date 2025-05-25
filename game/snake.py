import pygame
import sys
import random
import numpy as np

CELL_SIZE = 20
GRID_SIZE = 20
WIDTH, HEIGHT = CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE


class Snake:
    def __init__(self):
        self.body = [(10, 10), (9, 10), (8, 10)]
        self.direction = (1, 0)

    def move(self):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.body = [new_head] + self.body[:-1]

    def draw(self, surface):
        for x, y in self.body:
            pygame.draw.rect(
                surface,
                (0, 255, 0),
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )


class SnakeGameEnv:
    def __init__(self, fps=60):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
        self.food.position = self.food.random_position(self.snake.body)
        self.done = False
        self.fps = fps  # <--- nuevo parámetro

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.done = False
        return self.get_state()

    def step(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        if action is not None:
            if (directions[action][0], directions[action][1]) != tuple(
                -x for x in self.snake.direction
            ):
                self.snake.direction = directions[action]
        self.snake.move()
        reward = 0
        head_x, head_y = self.snake.body[0]
        if not (0 <= head_x < GRID_SIZE and 0 <= head_y < GRID_SIZE):
            self.done = True
            reward = -1
        elif self.snake.body[0] in self.snake.body[1:]:
            self.done = True
            reward = -1
        elif self.snake.body[0] == self.food.position:
            self.snake.body.append(self.snake.body[-1])
            self.food.position = self.food.random_position(self.snake.body)
            reward = 1
        else:
            reward = 0
        return self.get_state(), reward, self.done

    def render(self):
        self.display.fill((0, 0, 0))
        self.snake.draw(self.display)
        self.food.draw(self.display)
        pygame.display.update()
        self.clock.tick(self.fps)  # <--- usa self.fps

    def get_state(self):
        head_x, head_y = self.snake.body[0]
        dir_x, dir_y = self.snake.direction
        food_x, food_y = self.food.position

        dir_onehot = [
            int((dir_x, dir_y) == (0, -1)),
            int((dir_x, dir_y) == (0, 1)),
            int((dir_x, dir_y) == (-1, 0)),
            int((dir_x, dir_y) == (1, 0)),
        ]

        food_dx = food_x - head_x
        food_dy = food_y - head_y

        def obstacle(dx, dy):
            next_pos = (head_x + dx, head_y + dy)
            return int(
                not (0 <= next_pos[0] < GRID_SIZE and 0 <= next_pos[1] < GRID_SIZE)
                or next_pos in self.snake.body
            )

        moves = [
            self.snake.direction,
            (-self.snake.direction[1], self.snake.direction[0]),
            (self.snake.direction[1], -self.snake.direction[0]),
        ]
        obstacles = [obstacle(*move) for move in moves]

        state = np.array(dir_onehot + [food_dx, food_dy] + obstacles, dtype=np.float32)
        return state


class Food:
    def __init__(self):
        self.position = self.random_position()

    def random_position(self, snake_body=None):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if not snake_body or pos not in snake_body:
                return pos

    def draw(self, surface):
        x, y = self.position
        pygame.draw.rect(
            surface, (255, 0, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )


if __name__ == "__main__":
    env = SnakeGameEnv()
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action = 0
        if keys[pygame.K_DOWN]:
            action = 1
        if keys[pygame.K_LEFT]:
            action = 2
        if keys[pygame.K_RIGHT]:
            action = 3
        _, _, done = env.step(action)
        env.render()
        if done:
            pygame.time.wait(1000)
            env.reset()
