import gymnasium as gym
from Test_MultiAgent import VehicleScheduling
import numpy as np
import random
import pandas as pd

gym.register(
    id='VehicleScheduling-v1',
    entry_point='Test_MultiAgent:VehicleScheduling'
)

env = gym.make("VehicleScheduling-v1", render_mode="human")

# Training parameters
n_training_episodes = [100, 1000, 10000]
learning_rate = [0.7, 0.5, 0.1]        

# Evaluation parameters
n_eval_episodes = 1000      

# Environment parameters
env_id = "VehicleScheduling-v1"   
max_steps = 99             
gamma = [0.99, 0.95, 0.90]              
eval_seed = []             

# Exploration parameters
max_epsilon = [1, 0.7, 0.5]           
min_epsilon = 0.05           
decay_rate = [0.0005, 0.005, 0.05] 

dict_state = {}
iterator = 0

for k in range(env.n):
  for h in range(env.m):
    dict_state[iterator] = np.array([h,k])
    iterator += 1

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, learning_rate, gamma):
  for episode in range(n_training_episodes):
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()[0]
    steps = 0
    done = False

    # repeat
    for steps in range(max_steps):
      if env.n_agents == len(env.agents):
        for j in range(env.n_agents):
          if not env.agents[j].done:
            env.agents[j].epsilon_greedy_policy(env, epsilon)
            env.agents[j].action_to_direction(env.agents[j].action)
            print(env.step_counter, env.agents[j].name,env.agents[j].direction)
          else:
            env.agents[j].direction = np.array([0,0])
            print(env.step_counter, env.agents[j].name,env.agents[j].direction)
            env.agents[j].new_state = env.agents[j].current_state
      else:
        raise Exception("n_agents of environment doesn't match total agents") 
      
      action = None

      state, reward, done, truncated, info = env.step(action)

      env.reward = reward

      # Update q vaues and state
      for n in env.agents:
        n.update_q_table(learning_rate, gamma, dict_state)
        n.current_state = n.new_state

      if env.render_mode == "human":
        env.render_frame()
      # If done, finish the episode
      if done:
        print('reward:', env.reward)
        break

  return env.total_reward

# Hyperparameter Tuning
Results = pd.DataFrame(columns = ['training_episodes', 'max_epsilon', 'decay_rate', 'learning_rate', 'gamma', 'mean_reward'])
for a in n_training_episodes:
  for b in max_epsilon:
    for c in decay_rate:
      for d in learning_rate:
        for e in gamma:
          current_rewards = train(a, min_epsilon, b, c, env, max_steps, d, e)
          dict_frame = {'training_episodes': [a], 'max_epsilon': [b], 'decay_rate': [c], 'learning_rate': [d], 'gamma': [e], 'mean_reward': [np.mean(current_rewards[3:])]}
          Results = pd.concat([Results, pd.DataFrame.from_dict(dict_frame)], ignore_index = True)
          print(Results)
#Qtable_VehicleScheduling = train(n_training_episodes[0], min_epsilon, max_epsilon[-1], decay_rate[0], env, max_steps, learning_rate[-2], gamma[-1])
print(Results)

for i in Qtable_VehicleScheduling:
  print(i.name, ':\n')
  print(i.qtable, '\n')

env.render()
env.close()