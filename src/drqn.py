"""

    DRQN class

"""
#-------------------------------------------------------------------------------
# imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM
from keras.models import load_model

import numpy as np
import random

tf.keras.backend.set_floatx("float32")

#-------------------------------------------------------------------------------
# settings
EPS = 0
LR = 0.0003 # 0.1, 0.001, 0.0005, 0.00146, 0.0003, 0.0004, 0.00005, 0.00003
TIME_STEPS = 20

#-------------------------------------------------------------------------------
# DRQN class
class DRQN:
    def __init__(self, state_dim, action_dim, model_to_load=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPS
        self.model_to_load = model_to_load

        self.opt = Adam(LR)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()
        #self.model.summary()

    # creating the model: main/target
    def nn_model(self):

        # loading model if specified
        if self.model_to_load is not None:

            print(f"Loading model {self.model_to_load}")
            model = load_model(self.model_to_load)
            print(f"Model {self.model_to_load} loaded")
            return model
        # else, creating model
        else:

            return tf.keras.Sequential(
                [
                    Input((TIME_STEPS, self.state_dim)),
                    LSTM(256, activation="tanh"),
                    Dense(128, activation="relu"),
                    Dense(self.action_dim),
                ]
            )

    # predict action: util
    def predict(self, state):
        return self.model.predict(state)

    # predict action (main network), using decayed-epsilon-greedy policy
    def get_action(self, state):

        # predicting q value
        state = np.reshape(state, [1, TIME_STEPS, self.state_dim])
        q_value = self.predict(state)[0]

        # exploration/exploitation return
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)

        return np.argmax(q_value)

    # training
    def train(self, states, targets):
        # performing gradient descent, and computing loss
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            assert targets.shape == logits.shape
            loss = self.compute_loss(targets, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
