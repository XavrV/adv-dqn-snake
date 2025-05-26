import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from env.snake_env import SnakeEnv  # Asegúrate de que esta ruta sea correcta

# Hiperparámetros clave
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.005
EPSILON_DECAY = 1000
BATCH_SIZE = 64
MEM_SIZE = 10_000


# 1. Definir la red neuronal simple
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


# 3. Selección epsilon-greedy
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state)
        return q_values.argmax().item()


# 4. Entrenamiento DDQN loop (core)
def train():
    env = SnakeEnv(render_mode="human", fps=240, stack_size=8)

    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]
    policy_net = DQN(input_dim, n_actions).to(device)
    target_net = DQN(input_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(MEM_SIZE)

    epsilon = EPSILON_START
    steps = 0
    episode_rewards = []
    mlflow.set_experiment("snake-ddqn")
    with mlflow.start_run():
        for episode in range(2000):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Acción
                action = select_action(state, policy_net, epsilon, n_actions)
                next_state, reward, done, info = env.step(action)
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Entrenamiento
                if len(buffer) >= BATCH_SIZE:
                    batch = buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = batch
                    states = torch.tensor(states, dtype=torch.float32).to(device)
                    actions = (
                        torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                    )
                    rewards = (
                        torch.tensor(rewards, dtype=torch.float32)
                        .unsqueeze(1)
                        .to(device)
                    )
                    next_states = torch.tensor(next_states, dtype=torch.float32).to(
                        device
                    )
                    dones = (
                        torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
                    )

                    # DDQN: Selección de acción con policy_net, evaluación con target_net
                    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                    next_q_values = (
                        target_net(next_states).gather(1, next_actions).detach()
                    )
                    expected_q = rewards + GAMMA * next_q_values * (1 - dones)

                    q_values = policy_net(states).gather(1, actions)
                    loss = nn.MSELoss()(q_values, expected_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Actualización epsilon
                steps += 1
                epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
                    -1.0 * steps / EPSILON_DECAY
                )

                # Render (puedes comentar para hacerlo aún más rápido)
                env.render()

            episode_rewards.append(total_reward)
            mlflow.log_metric("reward", total_reward, step=episode)
            mlflow.log_metric("epsilon", epsilon, step=episode)
            mlflow.log_metric("episode_length", len(episode_rewards), step=episode)

            # Update target net
            if episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(
                    f"Ep {episode} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}"
                )

        mlflow.pytorch.log_model(policy_net, "model")
        env.close()


if __name__ == "__main__":
    train()
