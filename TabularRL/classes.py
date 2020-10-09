# classes
from functions import evaluate, policy_random
import gym
import numpy as np


class TabularRLTrainerAbstract(object):
    def __init__(self, env_name='FrozenLake8x8-v0', model_based=True):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.n
        self.model_based = model_based
        self.table = None

    def _get_transitions(self, state, act):
        """
        Query the environment to get the transition probability,
        reward, the next state, and done given a pair of state and action.
        """
        self._check_env_name()
        assert self.model_based, "_get_transitions is not available in MF method"
        ret = []
        transitions = self.env.env.P[state][act]
        for prob, next_state, reward, done in transitions:
            ret.append({
                'prob': prob,
                'next_state': next_state,
                'reward': reward,
                'done': done
            })
        return ret

    def _check_env_name(self):
        assert self.env_name.startswith('FrozenLake')

    def print_table(self):
        """print beautiful table, only work for FrozenLake8X8-v0 env. We
        write this function for you."""
        self._check_env_name()
        print(self.table)

    def train(self):
        """Conduct one iteration of learning."""
        raise NotImplementedError("You need to override the "
                                  "Trainer.train() function.")

    def evaluate(self):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 1000 episodes when seed=0."""
        result = evaluate(self.policy, 1000, env_name=self.env_name)
        return result

    def render(self):
        """Reuse evaluate function, render current policy
        for one episode when seed=0"""
        evaluate(self.policy, 1, render=False, env_name=self.env_name)


class PolicyIterationTrainer(TabularRLTrainerAbstract):
    def __init__(self, gamma=1.0, eps=1.0e-10, env_name='FrozenLake8x8-v0'):
        super(PolicyIterationTrainer, self).__init__(env_name)
        self.gamma = gamma
        self.eps = eps
        self.table = np.zeros((self.obs_dim,))
        self.policy = policy_random

    def train(self):
        """
        conduct one iteration of learning
        :return: None
        """
        self.update_value_function()
        self.update_policy()

    def update_value_function(self):
        """ update the value table
        :return: None
        """
        print("----------Policy Evaluation------------")
        count = 0
        while True:
            old_table = self.table.copy()
            for state in range(self.obs_dim):
                act = self.policy(state)
                transition_list = self._get_transitions(state, act)

                state_value = 0.0
                for transition in transition_list:
                    prob = transition['prob']
                    next_state = transition['next_state']
                    reward = transition['reward']
                    done = transition['done']
                    state_value = state_value + prob * (reward + self.gamma * self.table[next_state])
                self.table[state] = state_value

            if np.sum(np.abs(old_table - self.table)) <= self.eps:
                break
            count = count + 1
            if count % 200 == 0:
                print("[DEBUG]\tUpdated values for {} steps. "
                      "Difference between new and old table is: {}".format(
                      count, np.sum(np.abs(old_table - self.table))
                ))
            if count > 6000:
                raise ValueError("Clearly the code has problem. Check it!")

    def update_policy(self):
        """
        update the policy function
        :return: None
        """
        policy_table = np.zeros([self.obs_dim, ], dtype=np.int)
        for state in range(self.obs_dim):
            state_action_value = [0] * self.action_dim
            for action in range(self.action_dim):
                transition_list = self._get_transitions(state, action)
                for transition in transition_list:
                    prob = transition['prob']
                    next_state = transition['next_state']
                    reward = transition['reward']
                    done = transition['done']
                    state_action_value[action] = state_action_value[action] + prob * (reward + self.gamma * self.table[next_state])

            best_action = np.argmax(state_action_value)
            policy_table[state] = best_action
        self.policy = lambda obs: policy_table[obs]


class ValueIterationTrainer(PolicyIterationTrainer):
    """Note that we inherate Policy Iteration Trainer, to resue the
    code of update_policy(). It's same since it get optimal policy from
    current state-value table (self.table).
    """

    def __init__(self, gamma=1.0, env_name='FrozenLake8x8-v0'):
        super(ValueIterationTrainer, self).__init__(gamma, None, env_name)
        self.table = np.zeros((self.obs_dim,))

    def train(self):
        """Conduct one iteration of learning."""
        self.update_value_function()
        self.update_policy()

    def update_value_function(self):
        old_table = self.table.copy()

        for state in range(self.obs_dim):
            state_value = 0.0
            act = self.policy(state)
            transition_list = self._get_transitions(state, act)
            for transition in transition_list:
                prob = transition['prob']
                next_state = transition['next_state']
                reward = transition['reward']
                done = transition['done']
                state_value = state_value + prob * (reward + self.gamma * self.table[next_state])
            self.table[state] = state_value

    def evaluate(self):
        """Since in value itertaion we do not maintain a policy function,
        so we need to retrieve it when we need it."""
        self.update_policy()
        return super().evaluate()

    def render(self):
        """Since in value itertaion we do not maintain a policy function,
        so we need to retrieve it when we need it."""
        self.update_policy()
        return super().render()