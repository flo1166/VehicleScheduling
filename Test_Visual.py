import gymnasium as gym
from Test import VehicleScheduling
import numpy as np
import random

gym.register(
    id='VehicleScheduling-v0',
    entry_point='Test:VehicleScheduling'
)

env = gym.make("VehicleScheduling-v0", render_mode="human")

def print_env(env):
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample()) # display a random observation

    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    state_space = len(env.observation_space['agent'].sample()[0])
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")
    return state_space, action_space

state_space, action_space = print_env(env)

def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable

Qtable_VehicleScheduling = initialize_q_table(state_space, action_space)

def epsilon_greedy_policy(Qtable, state, epsilon):
  random_int = random.uniform(0,1)
  if random_int > epsilon:
    action = np.argmax(Qtable[state['agent'][0]])
  else:
    action = env.action_space.sample()
  return action

def greedy_policy(Qtable, state):
  action = np.argmax(Qtable[state])
  return action

# Training parameters
n_training_episodes = 1000
learning_rate = 0.7        

# Evaluation parameters
n_eval_episodes = 1000      

# Environment parameters
env_id = "VehicleScheduling-v1"   
max_steps = 99             
gamma = 1              
eval_seed = []             

# Exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.05           
decay_rate = 0.0005   

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state = env.reset()[0]
        steps = 0
        done = False

        # repeat
        for steps in range(max_steps):
          action = epsilon_greedy_policy(Qtable, state, epsilon)
          
          new_state, reward, done, truncated, info = env.step(action)
          
          Qtable[state['agent'][0]][action] = Qtable[state['agent'][0]][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state['agent'][0]]) - Qtable[new_state['agent'][0]][action])

          # If done, finish the episode
          if done:
            break
        
          # Our state is the new state
          state = new_state
      
    return Qtable

Qtable_VehicleScheduling = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_VehicleScheduling)

print(Qtable_VehicleScheduling)

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):

  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state = env.reset(seed=seed[episode])
    else:
      state = env.reset()[0]
    steps = 0
    done = False
    total_rewards_ep = 0
   
    for steps in range(max_steps):
      # Take the action (index) that have the maximum reward
      action = np.argmax(Q[state['agent'][0]][:])
      new_state, reward, done, truncated, info = env.step(action)
      total_rewards_ep += reward
       
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_VehicleScheduling, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

env.render()
env.close()