# models/dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class DQNAgent:

    def __init__(self, state_size, action_size=3):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0][0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3][0] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# TRAINING FUNCTION
def train_dqn_agent(prices,
                    predicted_returns,
                    sentiments,
                    volumes,
                    sma50,
                    sma200,
                    episodes=15,
                    save_path="models/rl_model.keras"):

    # FORCE CLEAN 1D FLOAT ARRAYS
    prices = np.asarray(prices).reshape(-1).astype(float)
    predicted_returns = np.asarray(predicted_returns).reshape(-1).astype(float)
    sentiments = np.asarray(sentiments).reshape(-1).astype(float)
    volumes = np.asarray(volumes).reshape(-1).astype(float)
    sma50 = np.asarray(sma50).reshape(-1).astype(float)
    sma200 = np.asarray(sma200).reshape(-1).astype(float)

    state_size = 5
    agent = DQNAgent(state_size=state_size)

    # 🔥 SAFE RETURN CALCULATION
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)

    for e in range(episodes):

        position = 0

        for t in range(len(prices) - 1):

            state = np.array([[ 
                predicted_returns[t],
                sentiments[t],
                volumes[t],
                sma50[t],
                sma200[t]
            ]])

            action = agent.act(state)
            new_position = action - 1  # 0=short,1=hold,2=long

            reward = position * returns[t]

            next_state = np.array([[ 
                predicted_returns[t+1],
                sentiments[t+1],
                volumes[t+1],
                sma50[t+1],
                sma200[t+1]
            ]])

            done = (t == len(prices) - 2)

            agent.remember(state, action, reward, next_state, done)

            position = new_position

            agent.replay()

        print(f"Episode {e+1}/{episodes} completed")

    agent.model.save(save_path)
    print("RL model saved to", save_path)

    return agent