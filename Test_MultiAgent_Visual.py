import gymnasium as gym
from Test_MultiAgent import VehicleScheduling
import numpy as np
import random

gym.register(
    id='VehicleScheduling-v1',
    entry_point='Test_MultiAgent:VehicleScheduling'
)

env = gym.make("VehicleScheduling-v1", render_mode="human")

# Training parameters
n_training_episodes = 1000
learning_rate = 0.5        

# Evaluation parameters
n_eval_episodes = 1000      

# Environment parameters
env_id = "VehicleScheduling-v1"   
max_steps = 99             
gamma = 0.90              
eval_seed = []             

# Exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.05           
decay_rate = 0.0005 

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
          else:
            env.agents[j].direction = np.array([0,0])
            env.agents[j].new_state = env.agents[j].current_state
      else:
        raise Exception("n_agents of environment doesn't match total agents") 
      
      action = None

      state, reward, done, truncated, info = env.step(action)

      # Update q vaues and state
      for n in env.agents:
        n.update_q_table(learning_rate, gamma, dict_state)
        n.current_state = n.new_state

      # If done, finish the episode
      if done:
        break

  return env.agents

Qtable_VehicleScheduling = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, learning_rate, gamma)

for i in Qtable_VehicleScheduling:
  print(i.qtable)

env.render()
env.close()