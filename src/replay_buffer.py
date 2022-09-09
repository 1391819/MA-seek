"""

    Replay buffer class

"""

# Imports
from collections import deque
import numpy as np
import random

# Constants
BATCH_SIZE = 128
BUFFER_CAPACITY = 50_000 # max buffer capacity
TIME_STEPS = 20 # number of previous observations

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    # storing transition inside buffer
    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    # sample transition from buffer
    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(BATCH_SIZE, TIME_STEPS, -1)
        next_states = np.array(next_states).reshape(BATCH_SIZE, TIME_STEPS, -1)

        return states, actions, rewards, next_states, done

    # return buffer size
    def size(self):
        return len(self.buffer)
