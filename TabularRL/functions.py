# policy function
import gym
import numpy as np
import matplotlib.pyplot as plt

action = np.random.randint(low=0, high=4, size=(64, ), dtype=np.int)


def policy_expert(obs):
    """ expert policy
    :param obs: observation
    :return: action
    """
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    return DOWN if (obs + 1) % 8 == 0 else RIGHT


def policy_random(obs):
    """ random policy
    :param obs: observation
    :return: action
    """
    return action[obs]


def policy_zero(obs):
    """ random policy
    :param obs: observation
    :return: action
    """
    action = 0
    return action


# evaluate function
def evaluate(policy, num_episode, seed=0, env_name='FrozenLake8x8-v0', render=False):
    """
    :param policy: policy function whose input is the observation
    :param num_episode: number of episodes
    :param seed: seed
    :param env_name: name of environment
    :param render: render the environment (animation)
    :return: the average return of each episode for a given policy
    """
    env = gym.make(env_name)
    env.seed(seed)
    rewards = []

    for i in range(num_episode):
        # reset the environment
        obs = env.reset()
        act = policy(obs)

        ep_reward = 0
        while True:
            # reward at each step and sum them to the episode reward.
            obs, reward, done, info = env.step(act)
            state_next = obs
            act = policy(state_next)

            ep_reward = ep_reward + reward
            if render:
                env.render()
            if done:
                break
        rewards.append(ep_reward)

    return np.mean(rewards)
