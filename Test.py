import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class VehicleScheduling(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10000}

    def __init__(self, render_mode=None, size=10):
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2, ), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            }
        )

        # We have 3 actions, corresponding to "right", "stop", "left"
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 0]),
            2: np.array([-1, 0])
        }

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
        self.reward = 0
        self.action = ''
    
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0,0], dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([9,0], dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.previous_time = self.step_counter
        self.step_counter = 0
        self.episode += 1
        self.reward = 0
        return observation, info
    
    def step(self, action):
        self.step_counter += 1
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 100 - self.step_counter + self.size if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # save reward for displaying
        self.reward += reward
        self.action = action

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
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            if x <= 1:
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * x),
                    (self.window_size, pix_square_size * x),
                    width=3,
                )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size),
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
        text4 = font.render("Reward: " + str(self.reward), False, (0, 0, 0))

        if self.action == 0:
            action = 'right'
        elif self.action == 1:
            action = 'stop'
        else:
            action = 'left'
        text5 = font.render("Action: " + action, False, (0, 0, 0))

        canvas.blit(text, (300,300))
        canvas.blit(text2, (300,325))
        canvas.blit(text3, (300,350))
        canvas.blit(text4, (300,375))
        canvas.blit(text5, (300,400))

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