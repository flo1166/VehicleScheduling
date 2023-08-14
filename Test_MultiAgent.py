import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from Agent import Agent
import random

class VehicleScheduling(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10000}

    def __init__(self, render_mode=None, n = 2, m = 10, n_agents = 2, max_steps = 99):
        
        # grid set up
        self.n = n
        self.m = m
        self._grid_shape = (m,n)
        self.grid = -1 * np.ones(self._grid_shape)
        self.stations = [0, -1]
        self._create_grid()
        
        # visualization set up
        self.window_size = 512  # The size of the PyGame window

        # We have 3 actions, corresponding to "right", "stop", "left"
        self.action_space = spaces.Discrete(3)

        # agents
        self.n_agents = n_agents
        self.agents = [Agent('agent1', 1, np.array([0,0]), np.array([9,0]), self.m * self.n, self.action_space.n, 0, 1, 9),
                       Agent('agent2', 2, np.array([9,0]), np.array([0,0]), self.m * self.n, self.action_space.n, 1, 11, 9)]
        
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

        # Counter
        self.step_counter = 0
        self.previous_time = 0
        self.episode = 0
        self.total_reward = [0,0]
 
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
#        shared_grid = self.grid
        for i in self.agents:
#            current_state = i.current_state
            i.new_state = np.clip(i.new_state, [0, 0], [self.m - 1, self.n])
#            new_state = i.new_state
#            shared_grid[current_state[0], current_state[1]] = i.name_int
#            shared_grid[new_state[0], new_state[1]] = i.name_int

        for k in self.agents:
            for l in self.agents:
                # check if they want to occupy same state
                if k != l and np.array_equal(k.new_state, l.new_state):
                    if k.prio > l.prio:
                        if k.action == 1:
                            l.new_state = l.current_state
                        else:
                            l.new_state = k.new_state + k.direction
                    else:
                        if l.action == 1:
                            k.new_state = k.current_state
                        else:
                            k.new_state = l.new_state + l.direction
                # check if they want to switch (as überholen is not allowed)
                elif k != l and np.array_equal(k.new_state, l.current_state) and np.array_equal(k.current_state, l.new_state):
                    # if k is in station
                    if self.check_if_station(k.new_state):
                        # check if k has higher prio, if so k can drive out of station
                        if k.prio > l.prio:
                            l.new_state = k.new_state + l.direction
                        else:
                            k.new_state = k.current_state
                    # if l is in station
                    elif self.check_if_station(l.new_state):
                        # check if l has higher prio, if so k can drive out of station
                        if l.prio > k.prio:
                            k.new_state = l.new_state + k.direction
                        else:
                            l.new_state = l.current_state
                    else:
                        # both are not in a station, check if k has higher prio
                        if k.prio > l.prio:
                            if k.action != 1:
                                l.new_state = k.new_state + k.direction
                            else:
                                l.new_state = l.current_state
                        else:
                            if l.action != 1:
                                k.new_state = l.new_state + l.direction
                            else:
                                k.new_state = k.current_state

        for n in self.agents:
            n.new_state = np.clip(n.new_state, [0, 0], [self.m - 1, self.n])

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # update reward
        reward = [j.reward for j in self.agents]
        reward = np.sum(reward)
        self.total_reward.append(reward)

        # Choose the agent's location uniformly at random
        for i in self.agents:
            i.current_state = i.start_location
            i.done = False
            i.reward = 0
            i.prev_lateness = i.lateness
            i.lateness = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # reset counters
        self.previous_time = self.step_counter
        self.step_counter = 0
        self.episode += 1
        
        return observation, info
    
    def step(self, action):
        self.step_counter += 1

        # check if the action is valid to not leave the grid (np.clip) nor to illegaly move
        self.valid_action()

        # An episode is done if the agent has reached the target
        dones = []
        reward = 0

        for i in self.agents:
            if i.done == False:
                i.done = np.array_equal(i.current_state, i.end_location)
                dones.append(i.done)
                if i.done:
                    i.lateness = i.timetable_start + i.duration - self.step_counter
                    i.reward += 100 + i.lateness
                else:
                    i.reward += -1
                reward += i.reward

        done = not (False in dones)

        # return gym things
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
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
        
        random.seed(42)
        colors = [random.sample(range(0, 255), 3) for i in range(self.n_agents)]
        # First we draw the target
        for j in range(self.n_agents):
            pygame.draw.rect(
                canvas,
                colors[j],
                pygame.Rect(
                    (pix_square_size * self.agents[j].end_location[0], pix_square_size * self.agents[j].end_location[1]),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        for k in range(self.n_agents):
            pygame.draw.circle(
                canvas,
                colors[k],
                ((self.agents[k].current_state[0] + 0.5) * pix_square_size, (self.agents[k].current_state[1] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )

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
        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render("Step: " + str(self.step_counter), False, (0, 0, 0))
        text2 = font.render("Previous Steps: " + str(self.previous_time), False, (0, 0, 0))
        text3 = font.render("Episode: " + str(self.episode), False, (0, 0, 0))
        text4 = font.render("Reward: " + str(self.total_reward[-1]), False, (0, 0, 0))
        
        if self.agents == None:
            action = [None for i in range(self.n_agents)]
        else:
            action = ['right' if i.action == 0 else 'stop' if i.action == 1 else 'left' if i.action == 2 else None for i in self.agents]

        text5 = font.render("Action: " + str(action), False, (0, 0, 0))
        dones = [i.done for i in self.agents]
        text6 = font.render("Done: " + str(dones), False, (0, 0, 0))
        text7 = font.render("Previous Reward: " + str(self.total_reward[-2]), False, (0, 0, 0))
        lateness = [self.agents[j].lateness for j in range(self.n_agents)]
        text8 = font.render("Lateness: " + str(lateness), False, (0, 0, 0))
        lateness_prev = [self.agents[j].prev_lateness for j in range(self.n_agents)]
        text9 = font.render("Previous Lateness: " + str(lateness_prev), False, (0, 0, 0))
        cur_reward = [self.agents[j].reward for j in range(self.n_agents)]
        text10 = font.render("Current Agent Reward: " + str(cur_reward), False, (0, 0, 0))

        canvas.blit(text, (300,300))
        canvas.blit(text2, (25,300))
        canvas.blit(text3, (300,325))
        canvas.blit(text4, (300,350))
        canvas.blit(text7, (25,350))
        canvas.blit(text5, (300,375))
        canvas.blit(text6, (300,400))
        canvas.blit(text8, (300,425))
        canvas.blit(text9, (25,425))
        canvas.blit(text10, (25,450))

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