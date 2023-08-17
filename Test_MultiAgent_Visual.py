import gymnasium as gym
from Test_MultiAgent import VehicleScheduling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gym.register(
    id='VehicleScheduling-v1',
    entry_point='Test_MultiAgent:VehicleScheduling'
)

env = gym.make("VehicleScheduling-v1", render_mode="human")
env.action_space.seed(42)

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, learning_rate, gamma):
  for episode in range(n_training_episodes):
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()[0]
    steps = 0
    done = False

    # repeat
    for steps in range(max_steps):
      for j in env.agents:
        #print(env.episode, j.name, j.active)
        if not j.done and j.active:
          j.epsilon_greedy_policy(env, epsilon)
          j.action_to_direction(j.action)
          #print(env.step_counter, j.name,j.direction)
        else:
          #print(env.step_counter, j.name,j.direction)
          j.new_state = j.current_state

      # check if three actions are active
      for k in env.agents:
        if k.active and k.action == None:
          raise ValueError("Missing action to active agent.")

      state, reward, done, truncated, info = env.step(None)

      # Update q values and state
      for n in env.agents:
        if n.active:
          n.update_q_table(learning_rate, gamma, env.dict_state)
        n.current_state = n.new_state

      if env.render_mode == "human":
        env.render_frame()
      # If done, finish the episode
      if done:
        #print('reward:', env.reward)
        break

  return env

def evaluate_agents(env, n_eval_episodes):
  for steps in range(env.max_steps):
      for j in env.agents:
        j.action = np.argmax(np.argmax(j.qtable[j.current_state[0]]))
        j.action_to_direction = j.action

      state, reward, done, truncated, info = env.step(None)

      # Update state
      for n in env.agents:
        n.current_state = n.new_state

      if env.render_mode == "human":
        env.render_frame()

      # If done, finish the episode
      if done:
        break

  return env

# Hyperparameter Tuning
'''
Results = pd.DataFrame(columns = ['training_episodes', 'max_epsilon', 'decay_rate', 'learning_rate', 'gamma', 'mean_reward', 'mean_lateness'])
n_training_episodes = [10000, 5000, 1000]
max_epsilon = [1, 0.7, 0.5]
min_epsilon =  0.05
decay_rate = [0.0005]
max_steps = 999
learning_rate = [0.7, 0.5, 0.3]
gamma = [0.99, 0.95, 0.9]

for a in n_training_episodes:
  for b in max_epsilon:
    for c in decay_rate:
      for d in learning_rate:
        for e in gamma:
          dict_frame = {'training_episodes': [a], 'max_epsilon': [b], 'decay_rate': [c], 'learning_rate': [d], 'gamma': [e], 'mean_reward': [np.mean(np.average(env.total_reward[2:])), 'mean_lateness': [np.mean(np.average(env.total_lateness))]}
          Results = pd.concat([Results, pd.DataFrame.from_dict(dict_frame)], ignore_index = True)
          print(Results[Results[']])
'''

Qtable_VehicleScheduling = train(env.n_training_episodes, env.min_epsilon, env.max_epsilon, env.decay_rate, env, env.max_steps, env.learning_rate, env.gamma)
#print(Results)

for i in Qtable_VehicleScheduling.agents:
  print(i.name, ':\n')
  print(i.qtable, '\n')
  print('Mean Reward: ', np.mean(Qtable_VehicleScheduling.total_reward))

# some plots
fig, ax = plt.subplots()
ax.step(range(len(Qtable_VehicleScheduling.total_reward[2:])), Qtable_VehicleScheduling.total_reward[2:], linewidth=2.5)
plt.plot(Qtable_VehicleScheduling.total_reward)
plt.show()

env.render()
env.close()