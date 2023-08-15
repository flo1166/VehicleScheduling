# Train Environment and Q-learning Agent

import numpy as np
import matplotlib.pyplot as plt

class Train:
    def __init__(self, start, goal, departure_time, arrival_time):
        self.start = start
        self.position = start
        self.goal = goal
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.actual_departure_time = None
        self.actual_arrival_time = None

class TrainEnvironment:
    def __init__(self, n_states, trains):
        self.n_states = n_states
        self.trains = trains
        self.stations = {'A': 0, 'B': 4, 'C': 8}  # Define the train stations
        self.total_states = n_states #* len(trains)
        self.reset()

    def reset(self):
        for train in self.trains:
            train.position = train.start
            train.actual_arrival_time = None
        return [train.position for train in self.trains]

    def step(self, train_id, action, step):
        train = self.trains[train_id]
     
        if action == 0:  # Move left
            train.position = max(0, train.position - 1)
        elif action == 1:  # Stay
            pass
        elif action == 2:  # Move right
            train.position = min(self.n_states - 1, train.position + 1)
            
        # Check for collisions
        reward = 0
        done = False
        positions = [t.position for t in self.trains]
        if positions.count(train.position) > 1:
            if train.position not in self.stations.values():
                reward = -100  # Collision penalty
                done = True
                self.reset()
                return [t.position for t in self.trains], reward, done  # Return immediately after reset
            else:
                pass
        # Check if the train has reached its goal
        if train.position == train.goal:
            reward = 100  # Goal reward
            # Update actual arrival time if the train reaches its goal
            if train.actual_arrival_time is None:
                train.actual_arrival_time = step
        ## Delay Berechnung
        if train.actual_arrival_time is not None and train.actual_arrival_time > train.arrival_time:
            reward -= (train.actual_arrival_time - train.arrival_time)
        return [t.position for t in self.trains], reward, done


class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.8, discount_factor=0.8, exploration_rate=0.8, exploration_decay=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = self.q_table[state, action] + self.lr * (target - predict)

def train_agents(env, agents, n_episodes=10000000, max_steps=50):
    rewards_per_episode = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = [False] * len(agents)
        step = 0
        crashed = False  # Flag to check if the episode should end due to a collision
        
        while not all(done) and step < max_steps:
            for i, agent in enumerate(agents):
                if not done[i] and step >= env.trains[i].departure_time:
                    action = agent.choose_action(state[i])
                    next_state, reward, crashed = env.step(i, action, step)
                    #print(f"Train {i} Action: {action}")
                    agent.learn(state[i], action, reward, next_state[i])
                    total_reward += reward
                    if next_state[i] == env.trains[i].goal:
                        done[i] = True
                if crashed:  # Check if the episode should end due to a collision
                    break
            state = next_state  # Update the state
            #print(f"Train State: {state}")
            step += 1
            if crashed:  # Break out of the outer loop if the episode should end
                break
        rewards_per_episode.append(total_reward)
        env.reset()  # Reset the environment at the end of each episode
    return rewards_per_episode


def print_q_table(q_table, agent_id):
    print(f"\nAgent {agent_id} Q-Table:")
    print("-" * 40)
    print("State | Action 0 | Action 1 | Action 2")
    print("-" * 40)
    for state, actions in enumerate(q_table):
        print(f"{state:5} | {actions[0]:8.2f} | {actions[1]:8.2f} | {actions[2]:8.2f}")
    print("-" * 40)









# Initialize environment and agents
trains = [
    Train(0, 4, departure_time=0, arrival_time=5),
    Train(4, 8, departure_time=0, arrival_time=5),
    Train(0, 8, departure_time=0, arrival_time=10)
]

env = TrainEnvironment(9, trains)
agents = [QLearningAgent(env.total_states, 3) for _ in env.trains]

# Train the agents
rewards = train_agents(env, agents)







# Output Q-Table
for i, agent in enumerate(agents):
    print_q_table(agent.q_table, i)
    
# Visualize the training progress
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show()