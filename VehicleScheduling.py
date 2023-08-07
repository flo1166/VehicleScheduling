import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import random
import numpy as np
import pygame

def forward_regio(state):
  state += 1
  return state

def wait_regio(state):
  pass

def backward_regio(state):
  state -= 1
  return state

actions = [forward_regio, wait_regio, backward_regio]

observations = [0 + i for i in range(100)]


class CustomEnv(gym.Env):
  metadata = {'render_modes': 'human'}
  """Custom Environment that follows gym interface"""

  def __init__(self, render_mode = None):
    self.actions = actions
    self.observations = observations

    # action space
    self.action_space = Discrete(len(actions))

    # observation space

    self.observation_space = gym.spaces.Dict(
            {
                "agent": Box(0, 100, shape=(2,), dtype=int),
                "target": Box(0, 100, shape=(2,), dtype=int),
            }
        )
    #self.observation_space = Box(low = 0,
    #                             high = 100,
    #                             shape =(len(observations), ),
    #                             dtype=int)
    
    # current state 
    self.state = 0

    # actions and their influence
    self._action_to_direction = {
            0: 1
            1: 0
            2: -1
        }
    
    # render of env
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    self.window = None
    self.clock = None

  def _get_obs(self):
    return {"agent": self._agent_location, "target": self._target_location}

  def info(self):
    return self._agent_location - self._target_location
  
  def step(self, action):
    '''
    Returns:the next observation, the reward, done and optionally additional info
    '''
    
    # map the action to the direction we move
    direction = self._action_to_direction[action]

    # check if we don't leave the grid
    self._agent_location = np.clip(
            self._agent_location + direction, 0, 100 - 1
        )
    
    # An episode is done iff the agent has reached the target
    terminated = np.array_equal(self._agent_location, self._target_location)
    
    # reward function
    reward = 1 if terminated else 0

    # observation
    observation = self._get_obs()

    # info to assess how far away
    info = self.info()

    return observation, reward, terminated, info

  
  def reset(self, seed = None):
    '''
    Returns: the observation of the initial state
    Reset the environment to initial state so that a new episode (indpendent of previous ones) may start
    '''

    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Choose the agent's and target location
    self._agent_location = 0
    self._target_location = 99

    observation = self._get_obs()
    info = self.info()

    if self.render_mode == "human":
                self._render_frame()

    return observation, info
    #return observation  # reward, done, info can't be included
  
  def render(self, mode='human'):
    '''
    Returns: None
    Show the current environment state e.g. the graphical window
    '''
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
            (pix_square_size * x, self.window_size),
            width=3,
        )

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

  def close (self):
    '''
    Returns: None
    This method is optional. Used to cleanup all resources
    '''
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()

env = CustomEnv()
env.reset()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

