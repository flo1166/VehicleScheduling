import gymnasium as gym
from Test import GridWorldEnv

gym.register(
    id='GridWorld-v0',
    entry_point='Test:GridWorldEnv'
)

env = gym.make("GridWorld-v0", render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)
env.render()

for _ in range(500):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()