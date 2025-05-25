import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.snake_env import SnakeEnv
import numpy as np
import time

env = SnakeEnv()
episodes = 10

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.random.randint(0, 4)  # Acción aleatoria (sustituir luego por DQN)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        env.render()  # Visualiza en tiempo real
        time.sleep(
            0.03
        )  # Baja para ver rápido (~30 FPS). Sube el valor para más lento.

        state = next_state

    print(f"Episodio {ep + 1} terminado. Score: {info['score']}")

env.close()
