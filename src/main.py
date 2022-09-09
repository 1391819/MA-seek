"""

    Main (agent class + main loop)

"""
#-------------------------------------------------------------------------------
#python 3.10.4
# imports
import tensorflow as tf
import numpy as np
import os

from maenv import MASeek
from drqn import DRQN
from replay_buffer import ReplayBuffer

#-------------------------------------------------------------------------------
# settings
TIME_STEPS = 20
BATCH_SIZE = 128
GAMMA = 0.99
# (66%) - 0.99955 for 10K episodes, 0.9985 for 3K episodes, 0.99977 for 20K episodes, 0.9991 for 5K episodes
EPS_DECAY = 0.9985
EPS_MIN = 0.05

MAX_EPISODES = 3000
MAX_TRY = 400
SAVING_MODEL_INTERVAL = 1000

# in case of longer training times,
# we could further "expand" the target update
# too many updates spanned close to each other
# could effectively have the opposite wanted effect
TARGET_UPDATE = 10
REPLAY_ITERATIONS = 10

#-------------------------------------------------------------------------------
# saving models
MODELS = "models"
SEEKER_MODELS_DIR = os.path.join(MODELS, "MASeekTest", "Seeker")
HIDER_MODELS_DIR = os.path.join(MODELS, "MASeekTest", "Hider")

# loading dirs
SEEKER_LOAD_MODEL = os.path.join(MODELS, "MASeek3K", "Seeker")
HIDER_LOAD_MODEL = os.path.join(MODELS, "MASeek3K", "Hider")

# logging for analysis
LOGDIR = "logs"
SEEKER_LOGDIR = os.path.join(LOGDIR, "MASeekTest", "Seeker")
HIDER_LOGDIR = os.path.join(LOGDIR, "MASeekTest", "Hider")

SEEKER_WRITER = tf.summary.create_file_writer(SEEKER_LOGDIR)
HIDER_WRITER = tf.summary.create_file_writer(HIDER_LOGDIR)

#-------------------------------------------------------------------------------
# agent class
class Agent:
    def __init__(self, env, model_to_load=None):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([TIME_STEPS, self.state_dim])

        # creating main network and target network
        self.model = DRQN(self.state_dim, self.action_dim, model_to_load)
        self.target_model = DRQN(self.state_dim, self.action_dim, model_to_load)
        # updating target network weights
        self.update_target()

        # initialising experience buffer
        self.buffer = ReplayBuffer()

    # updating target network weights based on main network
    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    # train main network using replay experience for a defined number
    # of times
    def replay_experience(self):
        for _ in range(REPLAY_ITERATIONS):
            # sample replay buffer
            states, actions, rewards, next_states, done = self.buffer.sample()
            # predict targets using main network
            targets = self.model.predict(states)
            # predict next q values using target network
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(BATCH_SIZE), actions] = (
                rewards + (1 - done) * next_q_values * GAMMA
            )
            # train main network
            self.model.train(states, targets)

    # adding states (w/ rolling window - earliest "timestep" state removed when
    # max timestep is reached)
    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

#-------------------------------------------------------------------------------
# main loop
if __name__ == "__main__":

    # env creation
    env = MASeek()

    # agent creation
    seeker_agent = Agent(env, SEEKER_LOAD_MODEL)
    hider_agent = Agent(env, HIDER_LOAD_MODEL)

    # analysis
    seeker_rewards, hider_rewards = [], []
    total_reward_seeker, total_reward_hider = 0, 0

    # main loop
    for episode in range(MAX_EPISODES + 1):

        # terminal conditions, seeker catches hider
        # or hider goes on top of the seeker
        seeker_done = False
        hider_done = False

        # analysis variables
        episode_seeker_reward, episode_hider_reward = 0, 0
        episode_seeker_ncollisions, episode_hider_ncollisions = 0, 0
        episode_seeker_success, episode_hider_success = 0, 0
        step = 0
        success = 0
        fail = 0

        # initialising states (time steps/previous obs - obs dim)
        seeker_agent.states = np.zeros([TIME_STEPS, seeker_agent.state_dim])
        hider_agent.states = np.zeros([TIME_STEPS, hider_agent.state_dim])

        # retrieving initial states, storing them (one for now - start)
        current_seeker_state, current_hider_state = env.reset()
        seeker_agent.update_states(current_seeker_state)
        hider_agent.update_states(current_hider_state)

        # game loop
        while not seeker_done and not hider_done:

            # retrieving action for both agents (w/ obviously main network)
            seeker_action = seeker_agent.model.get_action(seeker_agent.states)
            hider_action = hider_agent.model.get_action(hider_agent.states)

            # env step
            #seeker_next_state, seeker_reward, seeker_done, seeker_ncollisions = env.step_seekers(seeker_action)
            #hider_next_state, hider_reward, hider_done, hider_ncollisions = env.step_hiders(hider_action)
            seeker_next_state, seeker_reward, seeker_done, seeker_ncollisions, \
                hider_next_state, hider_reward, hider_done, hider_ncollisions = env.step(seeker_action, hider_action)

            # we save prev states so we can store in the exp replay past obs
            prev_seeker_states = seeker_agent.states
            prev_hider_states = hider_agent.states

            # updating curr state with next state (adding more than updating)
            seeker_agent.update_states(seeker_next_state)
            hider_agent.update_states(hider_next_state)

            # storing experience in replay buffer
            seeker_agent.buffer.store(prev_seeker_states, seeker_action, seeker_reward * 0.01, seeker_agent.states, seeker_done)
            hider_agent.buffer.store(prev_hider_states, hider_action, hider_reward * 0.01, hider_agent.states, hider_done)

            # analysis variables updates
            total_reward_seeker += seeker_reward
            episode_seeker_reward += seeker_reward
            total_reward_hider += hider_reward
            episode_hider_reward += hider_reward
            episode_seeker_ncollisions += seeker_ncollisions
            episode_hider_ncollisions += hider_ncollisions

            step += 1

            # rendering
            env.render()

            # max_try steps check + analysis
            if step >= MAX_TRY - 1:
                episode_hider_success += 1
                break

            # analysis
            if seeker_done or hider_done:
                episode_seeker_success += 1

        # experience replay
        if seeker_agent.buffer.size() >= BATCH_SIZE:
            seeker_agent.replay_experience()

        if hider_agent.buffer.size() >= BATCH_SIZE:
            hider_agent.replay_experience()

        # target network update (every TARGET_UPDATE)
        if episode % TARGET_UPDATE == 0:
            seeker_agent.update_target()
            hider_agent.update_target()

        # decaying epsilon
        seeker_agent.model.epsilon *= EPS_DECAY
        seeker_agent.model.epsilon = max(seeker_agent.model.epsilon, EPS_MIN)

        hider_agent.model.epsilon *= EPS_DECAY
        hider_agent.model.epsilon = max(hider_agent.model.epsilon, EPS_MIN)

        # logging data to tensorboard
        with SEEKER_WRITER.as_default():
            tf.summary.scalar("episode_reward", episode_seeker_reward, step=episode)
            tf.summary.scalar("cumulative_reward", np.mean(seeker_rewards)/episode, step=episode)
            tf.summary.scalar("episode_ncollisions", episode_seeker_ncollisions, step=episode)
            tf.summary.scalar("wins", episode_seeker_success, step=episode)
            tf.summary.scalar("epsilon", seeker_agent.model.epsilon, step=episode)
        with HIDER_WRITER.as_default():
            tf.summary.scalar("episode_reward", episode_hider_reward, step=episode)
            tf.summary.scalar("cumulative_reward", np.mean(hider_rewards)/episode, step=episode)
            tf.summary.scalar("episode_ncollisions", episode_hider_ncollisions, step=episode)
            tf.summary.scalar("wins", episode_hider_success, step=episode)
            tf.summary.scalar("epsilon", hider_agent.model.epsilon, step=episode)

        # analysis variables updates
        seeker_rewards.append(total_reward_seeker)
        hider_rewards.append(total_reward_hider)

        # saving models every tot episodes (fixed)
        if episode % SAVING_MODEL_INTERVAL == 0 and episode != 0:
            seeker_agent.model.model.save(SEEKER_MODELS_DIR)
            hider_agent.model.model.save(HIDER_MODELS_DIR)


    env.close()
