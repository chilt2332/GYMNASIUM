import gymnasium as gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CarRacing-v3", continuous = False)

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape = (1, states)))
model.add(Dense(24, activation = "relu"))
model.add(Dense(24, activation = "relu"))
model.add(Dense(actions, activation = "linear"))

agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit = 50000, window_length = 1),
    policy=BoltzmannQPolicy(),
    nb_actions = actions,
    nb_step_warmup = 10,
    target_model_update = 0.01
)

agent.compile(Adam(lr = 0.001), metrics = ["mae"])
agent.fit(env, nb_steps=10000, visualize = False, verbose = 1)

results = agent.test(env, nb_episodes=10, visualize = True)
print(np.mean(results.history["episode_reward"]))

env.close()