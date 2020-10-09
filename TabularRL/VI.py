# Value iteration
import gym
import operator as op
from classes import PolicyIterationTrainer


seed = 1
# Create the environment
env = gym.make('FrozenLake8x8-v0')
env.reset()