import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from Agent import Agent

class VehicleScheduling(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10000}

    def __init__(self, render_mode=None, n = 2, m = 9, n_agents = 3, max_steps = 99):
        # grid set up
        self.n = n
        self.m = m
        self._grid_shape = (m,n)
        self.grid = -1 * np.ones(self._grid_shape)
        self.stations = [0, 4, 8]
        self._create_grid()
        
        # visualization set up
        self.window_size = 512  # The size of the PyGame window

        # We have 3 actions, corresponding to "right", "stop", "left"
        self.action_space = spaces.Discrete(3)

        # agents
        self.agents = [Agent('agent1', 1, 1, np.array([0,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 1, 8, 9, True),
                       Agent('agent2', 2, 2, np.array([0,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 2, 4, 6, True),
                       Agent('agent3', 3, 3, np.array([8,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 1, 4, 5, True),
                       Agent('agent4', 4, 1, np.array([8,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 10, 8, 18, False),
                       Agent('agent5', 5, 2, np.array([4,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 6, 4, 10, False),
                       Agent('agent6', 6, 3, np.array([4,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 6, 4, 10, False)#,
                       #Agent('agent7', 7, 1, np.array([0,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 19, 8, 27, False),
                       #Agent('agent8', 8, 2, np.array([0,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 10, 4, 14, False),
                       #Agent('agent9', 9, 3, np.array([8,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 11, 4, 15, False),
                       #Agent('agent10', 10, 1, np.array([8,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 28, 8, 36, False),
                       #Agent('agent11', 11, 2, np.array([4,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 15, 4, 19, False),
                       #Agent('agent12', 12, 3, np.array([4,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 15, 4, 19, False),
                       #Agent('agent13', 13, 2, np.array([0,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 20, 4, 24, False),
                       #Agent('agent14', 14, 2, np.array([4,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 24, 4, 28, False),
                       #Agent('agent15', 15, 2, np.array([0,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 28, 4, 32, False),
                       #Agent('agent16', 16, 2, np.array([4,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 33, 4, 37, False),
                       #Agent('agent17', 17, 3, np.array([8,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 19, 4, 23, False),
                       #Agent('agent18', 18, 3, np.array([4,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 24, 4, 28, False),
                       #Agent('agent19', 19, 3, np.array([8,0]), np.array([4,0]), self.m * self.n, self.action_space.n, 29, 4, 33, False),
                       #Agent('agent20', 20, 3, np.array([4,0]), np.array([8,0]), self.m * self.n, self.action_space.n, 33, 4, 37, False)
                       ]
        
        self.n_agents = len(self.agents)
        # observations
        self.observation_space = spaces.Dict(
            {"agent" + str(i + 1): spaces.Box(0, 1, shape=(2, ), dtype=int) for i in range(self.n_agents)}
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.max_steps = max_steps

        # Counter
        self.step_counter = 0
        self.previous_time = 0
        self.episode = 0
        self.reward = 0
        self.total_reward = [0,0]
        self.crash = [False]
        self.total_lateness = []
        self.total_dones = []

        # Hyperparameters
        self.learning_rate = 0.7
        self.n_training_episodes = 10000    

        # Environment parameters
        self.max_steps = 999             
        self.gamma = 0.99                 

        # Exploration parameters
        self.max_epsilon = 1       
        self.min_epsilon = 0.05           
        self.decay_rate = 0.0005 
        
        # dictionary for states
        dict_state = {}
        iterator = 0

        for k in range(self.n):
            for h in range(self.m):
                dict_state[iterator] = np.array([h,k])
                iterator += 1
        self.dict_state = dict_state
        
    def _create_grid(self):
        for i in self.stations:
            self.grid[i, 0] = 100
        self.grid[1:-1,0] = 0

    def _get_obs(self):
        return {i.name: i.current_state for i in self.agents}
                
    def _get_info(self):
        return {"distance": np.linalg.norm(self.agents[0].current_state - self.agents[1].current_state, ord=1)}
    
    def check_if_station(self, new_state):
        checks = []
        for i in self.stations:
            checks.append(np.array_equal(np.array([i, 0]), new_state))
        return True in checks

    def valid_action(self):
        for i in self.agents:
            i.new_state = np.clip(i.new_state, [0, 0], [self.m - 1, self.n - 1])

    def reset(self, seed = 42, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # concat rewards, latenesses and dones
        reward = [j.reward for j in self.agents]
        self.total_reward.append(reward)
        lateness = [j.lateness for j in self.agents]
        self.total_lateness.append(lateness)
        dones = [j.done for j in self.agents]
        self.total_dones.append(dones)
        self.reward = 0
        # Choose the agent's location uniformly at random
        for i in self.agents:
            i.current_state = i.start_location
            i.done = False
            i.reward = 0
            i.lateness = 0
            i.crash = False
            i.active = False
        
        for j in range(3):
            self.agents[j].active = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()

        # reset counters
        self.previous_time = self.step_counter
        self.step_counter = 0
        self.episode += 1

        return observation, info
    
    def if_crash(self, agent, neighbor):
        if not self.check_if_station(agent.new_state):
            agent.reward = -200
            agent.lateness = self.max_steps
            agent.done = True
            agent.crash = True
        elif not self.check_if_station(neighbor.new_state):
            agent.reward = -200
            agent.lateness = self.max_steps
            agent.done = True
            agent.crash = True

    def crash_neighbors(self, i, left_neighbor, right_neighbor, left_neighbor2, right_neighbor2, same_state):
        if left_neighbor != []:
            for l in left_neighbor:
                if (i.action == 1 and l.action == 0) or (i.action == 2 and l.action <= 1):
                    self.if_crash(i, l)

        if right_neighbor != []:
            for r in right_neighbor:
                if (i.action == 0 and 1 <= r.action <= 2) or (i.action == 1 and r.action == 2):
                    self.if_crash(i, r) 

        if left_neighbor2 != []:
            for l2 in left_neighbor2:
                if (i.action == 2 and l2.action == 0):  
                   self.if_crash(i, l2)
               
        if right_neighbor2 != []:
            for r2 in right_neighbor2:
                if (i.action == 0 and r2.action == 2):
                    self.if_crash(i, r2)
        
        if same_state != []:
            for same in same_state:
                if self.check_if_station(i.current_state):
                    if (i.action == same.action and not self.check_if_station(i.new_state)):
                        self.if_crash(i, same)
                else:
                    self.if_crash(i, same)

    def step(self, action):
        self.step_counter += 1
        truncated = False

        # check if agent is in grid world
        self.valid_action()

        # An episode is done if all agents have reached the target
        dones = []
        reward = 0
        crash = False

        # Check if neighbors
      
        for i in self.agents:
            if i.active:
                left_neighbor = []
                right_neighbor = []
                left_neighbor2 = []
                right_neighbor2 = []
                same_state = []
                
                for j in self.agents:
                    if i != j and j.active:               
                        if (i.current_state + np.array([1,0]) == j.current_state).all():
                            right_neighbor.append(j)
                        elif (i.current_state + np.array([-1,0]) == j.current_state).all():
                            left_neighbor.append(j)
                        elif (i.current_state + np.array([2,0]) == j.current_state).all():
                            right_neighbor2.append(j)
                        elif (i.current_state + np.array([-2,0]) == j.current_state).all():
                            left_neighbor2.append(j)
                        elif np.array_equal(i.current_state, j.current_state):
                            same_state.append(j)

                # check for crash
                self.crash_neighbors(i, left_neighbor, right_neighbor, left_neighbor2, right_neighbor2, same_state)

                # Rewards
                if i.done == False:
                    i.done = np.array_equal(i.new_state, i.end_location)
                    
                    # activate new train
                    if i.done:
                        i.update_q_table(self.learning_rate, self.gamma, self.dict_state)
                        i.action = None
                        i.active = False
                        
                        for j in self.agents:
                            if i != j and i.zug_id == j.zug_id and j.done == False:
                                j.active = True
                                j.action = 1
                                break

                    dones.append(i.done)

                    if i.done:
                        i.lateness = i.timetable_start + i.duration - self.step_counter
                        i.reward = 100 + i.lateness
                        reward += i.reward
                    else:
                        i.reward = 0
                        reward += i.reward
                
                if i.crash:
                    reward += i.reward
        
        # document crash
        for k in self.agents:
            curr_crash = []
            if k.crash == True:
                done = True
                crash = True
                curr_crash.append(crash)
                break
            
        if True in curr_crash:
            self.crash.append(True)
        else:
            self.crash.append(False)

        # is it done?
        if not crash:
            done = not (False in dones)

        # return gym things
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self._grid_shape[0]
        )  # The size of a single grid square in pixels
        
        
        colors = [[163, 28, 6], [189, 70, 62], [57, 35, 188], [26, 173, 189], [228, 139, 22], [151, 108, 8], [7, 23, 55], [59, 129, 154], [6, 143, 50], [183, 166, 179], [139, 107, 56], [114, 150, 71], [207, 222, 1], [194, 206, 40], [178, 108, 87], [71, 39, 55], [245, 195, 86], [26, 23, 97], [24, 91, 216], [88, 154, 67]]
        font = pygame.font.Font('freesansbold.ttf', 12)

        # First we draw the target
        rec_int = 0
        for j in range(self.n_agents):
            if self.agents[j].active:
                pygame.draw.rect(
                    canvas,
                    colors[j],
                    pygame.Rect(
                        ((pix_square_size + (rec_int)) * self.agents[j].end_location[0], pix_square_size * self.agents[j].end_location[1]),
                        (pix_square_size, pix_square_size),
                    ),
                )
            rec_int += 1

        # Now we draw the agent
        circle_int = 150
        for k in range(self.n_agents):
            if self.agents[k].active:

                pygame.draw.circle(
                    canvas,
                    colors[k],
                    ((self.agents[k].current_state[0] + (circle_int / 300)) * pix_square_size, (self.agents[k].current_state[1] + 0.5) * pix_square_size),
                    pix_square_size / 3,
                )
                pygame.draw.circle(
                    canvas,
                    colors[k],
                    (25,circle_int),
                    pix_square_size / 3,
                )
                canvas.blit(font.render(str(self.agents[k].name), False, (0, 0, 0)), (50, circle_int - 10))
                action_circle = None
                if self.agents[k].action == 0:
                    action_circle = 'right'
                elif self.agents[k].action == 1:
                    action_circle = 'stop'
                elif self.agents[k].action == 2:
                    action_circle = 'left'
                canvas.blit(font.render(action_circle, False, (0, 0, 0)), (125, circle_int - 10))
                circle_int += 50

        # Now we draw the non accessible area
        for l in range(self.m):
            for m in range(self.n):
                if self.grid[l,m] == -1:
                    pygame.draw.rect(
                        canvas,
                        (0,0,0),
                        pygame.Rect(
                            (pix_square_size * l, pix_square_size * m),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Finally, add some gridlines
        for x in range(self._grid_shape[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size * self._grid_shape[1]),
                width=3,
            )

        for x in range(self._grid_shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

        # Counter for steps
        green = (0, 255, 0)
        blue = (0, 0, 128)  
        X = 200
        Y = 200
        
        text = font.render("Step: " + str(self.step_counter), False, (0, 0, 0))
        text2 = font.render("Previous Steps: " + str(self.previous_time), False, (0, 0, 0))
        text3 = font.render("Episode: " + str(self.episode), False, (0, 0, 0))
        cur_reward = [j.reward for j in self.agents]
        text4 = font.render("Reward: " + str(np.sum(cur_reward)), False, (0, 0, 0))
        
        if self.agents == None:
            action = [None for i in range(self.n_agents)]
        else:
            action = ['right' if i.action == 0 else 'stop' if i.action == 1 else 'left' if i.action == 2 else None for i in self.agents]

        text5 = font.render("Action: " + str(action), False, (0, 0, 0))
        dones = [i.done for i in self.agents]
        text6 = font.render("Done: " + str(dones), False, (0, 0, 0))
        text7 = font.render("Previous Reward: " + str(self.total_reward[-1]), False, (0, 0, 0))
        lateness = [self.agents[j].lateness for j in range(self.n_agents)]
        text8 = font.render("Lateness: " + str(lateness), False, (0, 0, 0))
        #text9 = font.render("Previous Lateness: " + str(lateness_prev), False, (0, 0, 0))
        text10 = font.render("Reward: " + str(cur_reward), False, (0, 0, 0))
        text11 = font.render("Crash: " + str(self.crash[-1]), False, (0, 0, 0))

        canvas.blit(text, (300,300))
        canvas.blit(text2, (25,300))
        canvas.blit(text3, (300,325))
        canvas.blit(text4, (25,325))
        canvas.blit(text7, (25,350))
        canvas.blit(text5, (25,375))
        canvas.blit(text6, (25,400))
        canvas.blit(text8, (25,450))
        #canvas.blit(text9, (25,475))
        canvas.blit(text10, (25,425))
        canvas.blit(text11, (300,450))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()