import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
# import config # No longer need to import config directly for these parameters

class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=10000,
                 replay_buffer_size=10000,
                 batch_size=64,
                 target_update_freq=100): # Added more params here

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_buffer_size) # Use passed parameter
        self.gamma = gamma    # Use passed parameter
        self.epsilon = epsilon_start  # Use passed parameter
        self.epsilon_initial_start = epsilon_start # Store initial for robust decay calculation
        self.epsilon_min = epsilon_end # Use passed parameter
        self.epsilon_decay_steps = epsilon_decay_steps # Use passed parameter
        self.learning_rate = learning_rate # Use passed parameter
        
        self.batch_size = batch_size # Store for use in replay
        self.target_update_freq = target_update_freq # Store for use in replay

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.train_step_counter = 0

    def _build_model(self):
        # Simple MLP for Q-value approximation
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear') # Q-values per action
        ])
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)) # Uses self.learning_rate
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self): # batch_size is now self.batch_size
        if len(self.memory) < self.batch_size: # Use self.batch_size
            return 0 

        minibatch = random.sample(self.memory, self.batch_size) # Use self.batch_size
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        targets = rewards + self.gamma * np.max(q_values_next_target, axis=1) * (1 - dones)

        target_q_values = q_values_current
        batch_indices = np.arange(self.batch_size) # Use self.batch_size
        target_q_values[batch_indices, actions] = targets

        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0: # Use self.target_update_freq
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
             # Use stored initial epsilon for correct decay rate over potentially multiple training sessions/resets
             self.epsilon -= (self.epsilon_initial_start - self.epsilon_min) / self.epsilon_decay_steps
             self.epsilon = max(self.epsilon_min, self.epsilon)
        return loss

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)