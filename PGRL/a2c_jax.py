"""policy gradient RIENFORCE with JAX"""
import numpy as np
import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp
import gym
from jax import random
from collections import OrderedDict
import optax

import matplotlib.pyplot as plt

# RL environment setup.
env = gym.make("InvertedPendulum-v4")
env.reset(seed=0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_steps = 600


class MultivariateNormalDiag(eqx.Module):
    # Cheap knockoff of `tensorflow_probability.substrates.jax.distributions.MultivariateNormalDiag`.
    mean: jnp.array
    std: jnp.array

    @jax.jit
    def sample(self, key, shape=()):
        return self.mean + self.std * jax.random.normal(key, shape + self.mean.shape)

    @jax.jit
    def log_prob(self, value):
        return jnp.sum(jax.scipy.stats.norm.logpdf(value, self.mean, self.std), -1)


class Policy(eqx.Module):
    trunk_layers: list
    action_mean_head: eqx.Module
    action_std_head: eqx.Module
    value_head: eqx.Module

    def __init__(self, key):
        # `PRNGKey`s for initializing NN layers.
        keys = jax.random.split(key, 5)
        # Embedding layers.
        self.trunk_layers = [
            eqx.nn.Linear(state_dim, 68, key=keys[0]),
            eqx.nn.Linear(68, 68, key=keys[1]),
        ]
        # Actor's layers.
        self.action_mean_head = eqx.nn.Linear(68, action_dim, key=keys[2])
        self.action_std_head = eqx.nn.Linear(68, action_dim, key=keys[3])
        # Critic's layers.
        self.value_head = eqx.nn.Linear(68, 1, key=keys[4])

    @jax.jit
    def __call__(self, x):
        for layer in self.trunk_layers:
            x = jax.nn.relu(layer(x))
        action_mean = self.action_mean_head(x)
        action_std = jax.nn.softplus(self.action_std_head(x))
        value = self.value_head(x)[0]
        return MultivariateNormalDiag(action_mean, action_std), value


def compute_returns(rewards, discount_factor):
    tail_return = 0.
    returns = []
    for r in rewards[::-1]:
        tail_return = r + discount_factor * tail_return
        returns.append(tail_return)
    returns.reverse()
    return np.array(returns)

@jax.jit
def train_loss_for_epsiode(policy: Policy, states, actions, returns, num_steps):
    action_distributions, values = jax.vmap(policy)(states)
    action_log_probs = action_distributions.log_prob(actions)
    advantages = returns - values
    mask = jnp.arange(max_steps) < num_steps
    actor_loss = jnp.sum(-action_log_probs * jax.lax.stop_gradient(advantages) * mask)
    critic_loss = jnp.sum(jnp.square(advantages) * mask)
    return actor_loss + critic_loss


def train_step_for_episode(opt_state, policy, states, actions, returns, num_steps):
    grads = jax.grad(train_loss_for_epsiode)(policy, states, actions, returns, num_steps)
    updates, opt_state = optimizer.update(grads, opt_state)
    policy = optax.apply_updates(policy, updates)
    return opt_state, policy


if __name__ == '__main__':
    render = bool(1)
    render_interval = 50
    # Training parameters.
    discount_factor = 0.99  # Discount factor for computing tail returns.
    ema_factor = 0.99  # Exponential moving average for standardizing returns.
    key = jax.random.PRNGKey(0)

    key, policy_key = jax.random.split(key)
    policy = Policy(policy_key)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(policy)

    episodic_rewards = []
    smooth_episodic_rewards = []
    episodic_reward_ema = None  # Exponential moving average of episodic rewards.
    return_ema = None           # Exponential moving average of returns (i.e., critic targets).
    return_emv = None           # Exponential moving variance of returns (i.e., critic targets).

    for i_episode in range(3001):
        # Reset environment at the start of each episode, and clear accumulators.
        state = env.reset()
        episodic_reward = 0
        states = []
        actions = []
        rewards = []

        # Sample a trajectory (episode) according to our stochastic policy/environment dynamics.
        for t in range(max_steps):
            states.append(state)
            action_distribution, _ = policy(state)
            key, sample_key = jax.random.split(key)
            action = np.array(action_distribution.sample(sample_key))  # Leave JAX to interact with gym.
            state, reward, done, _ = env.step(action)
            episodic_reward += reward
            actions.append(action)
            rewards.append(reward)
            if render and i_episode % render_interval == 0:
                env.render()
            if done:
                break

        # Compute (standardized) tail returns for the episode and update moving averages.
        episodic_rewards.append(episodic_reward)
        returns = compute_returns(rewards, discount_factor)
        if i_episode == 0:
            episodic_reward_ema = episodic_reward
            return_ema = returns.mean()
            return_emv = returns.var()
        else:
            episodic_reward_ema = 0.95 * episodic_reward_ema + (1 - 0.95) * episodic_reward
            return_ema = ema_factor * return_ema + (1 - ema_factor) * returns.mean()
            return_emv = ema_factor * (return_emv + (1 - ema_factor) * np.mean((returns - return_ema)**2))
        smooth_episodic_rewards.append(episodic_reward_ema)
        # Normalization
        standardized_returns = (returns - return_ema) / (np.sqrt(return_emv) + 1e-6)

        # Run a train step based on the episode's data.
        num_steps = len(states)

        # JAX prefers all arrays to be the same shape for jitting, so we pad to the batch size.
        opt_state, policy = train_step_for_episode(
            opt_state,
            policy,
            np.pad(states, ((0, max_steps - num_steps), (0, 0))),
            np.pad(actions, ((0, max_steps - num_steps), (0, 0))),
            np.pad(standardized_returns, ((0, max_steps - num_steps),)),
            num_steps,
        )

        # Periodically log results.
        if i_episode % 10 == 0:
            print(
                f"Episode {i_episode}\tLast reward: {episodic_reward:.2f}\tMoving average reward: {episodic_reward_ema:.2f}"
            )
        if episodic_reward_ema > 580:
            break

    plt.figure(1)
    plt.plot(episodic_rewards)
    plt.plot(smooth_episodic_rewards)
    plt.show()