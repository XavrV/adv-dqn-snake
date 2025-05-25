from env.snake_env import SnakeEnv
import time

env = SnakeEnv()
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)  # visualiza lento

env.close()
