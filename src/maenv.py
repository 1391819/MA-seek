"""

    Environment "handler" for the game.

"""
#-------------------------------------------------------------------------------
# Imports
import numpy as np
from game import GameAI
from gym.spaces import Box, Discrete

#-------------------------------------------------------------------------------

class MASeek:
    def __init__(self):
        super(MASeek, self).__init__()
        # initialising game instance
        self.pygame = GameAI()

        # action space
        self.action_space = Discrete(8)
        # obs space/state
        self.observation_space = Box(low=0, high=5000, shape=(74,), dtype=np.float32)

    # env reset
    def reset(self):
        # Resetting the environment
        del self.pygame
        self.pygame = GameAI()

        # returning observation
        observation_seekers = self.pygame.observe_seekers()
        observation_hiders = self.pygame.observe_hiders()

        return observation_seekers, observation_hiders

    # not used, step only seeker(s)
    def step_seekers(self, action):

        self.pygame.action_seekers(action)
        obs = self.pygame.observe_seekers()
        reward = self.pygame.evaluate_seekers()
        done = self.pygame.is_done_seekers()
        n_collisions = self.pygame.get_n_collisions_seekers()

        return obs, reward, done, n_collisions

    # not used, step only hider(s)
    def step_hiders(self, action):

        self.pygame.action_hiders(action)
        obs = self.pygame.observe_hiders()
        reward = self.pygame.evaluate_hiders()
        done = self.pygame.is_done_hiders()
        n_collisions = self.pygame.get_n_collisions_hiders()

        return obs, reward, done, n_collisions

    # perform env step (all agents)
    # implementation is really "static", mainly due to the presence
    # of only two agents
    #
    # future work -> use a list or dict containing all agents + IDs representing their
    # "type" (seeker/hider)
    def step(self, seeker_action, hider_action):

        # seeker
        self.pygame.action_seekers(seeker_action)
        seeker_obs = self.pygame.observe_seekers()
        seeker_reward = self.pygame.evaluate_seekers()
        seeker_done = self.pygame.is_done_seekers()
        seeker_n_collisions = self.pygame.get_n_collisions_seekers()

        # hider
        self.pygame.action_hiders(hider_action)
        hider_obs = self.pygame.observe_hiders()
        hider_reward = self.pygame.evaluate_hiders()
        hider_done = self.pygame.is_done_hiders()
        hider_n_collisions = self.pygame.get_n_collisions_hiders()

        return seeker_obs, seeker_reward, seeker_done, seeker_n_collisions, hider_obs, hider_reward, hider_done, hider_n_collisions

    # rendering environment
    def render(self, mode='human'):
        self.pygame.view()

    def close(self):
        pass
