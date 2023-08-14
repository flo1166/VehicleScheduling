import numpy as np
import random

class Agent():
  def __init__(self, name, name_int, start, end, state_space, action_space, priority, timetable_start, duration):
    self.name = name
    self.name_int = name_int
    self.start_location = start
    self.end_location = end
    self.current_state = start
    self.new_state = None
    self.reward = 0
    self.qtable = self.initialize_q_table(state_space, action_space)
    self.done = False
    self.direction = None
    self.action = None
    self.prio = priority
    self.timetable_start = timetable_start
    self.duration = duration
    self.lateness = 0
    self.prev_lateness = 0
  
  def initialize_q_table(self, state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable
  
  def update_q_table(self, learning_rate, gamma, dict_state):
    for i in range(len(list(dict_state.values()))):
      if np.array_equal(list(dict_state.values())[i], self.current_state):
        current_state = list(dict_state.keys())[i]
      if np.array_equal(list(dict_state.values())[i], self.new_state):
        new_state = list(dict_state.keys())[i]
    self.qtable[current_state][self.action] = (1 - learning_rate) * self.qtable[current_state][self.action] + learning_rate * (self.reward + gamma * np.max(self.qtable[new_state]) - self.qtable[new_state][self.action])
  
  def epsilon_greedy_policy(self, env, epsilon):
    random_int = random.uniform(0,1)
    if random_int > epsilon:
      action = np.argmax(self.qtable[self.current_state[0]])
    else:
      action = env.action_space.sample()
    self.action = action

  def greedy_policy(self):
    action = np.argmax(self.qtable[self.current_state])
    return action
  
  def action_to_direction(self, action):
    action_to_direction = {
          0: np.array([1, 0]),
          1: np.array([0, 0]),
          2: np.array([-1, 0])
      }
    self.direction = action_to_direction[action]
    self.new_state = self.current_state + self.direction
    