"""policy gradient RIENFORCE with JAX"""
import numpy as np
import jax
import jax.numpy as jnp
import gym
from collections import OrderedDict
import optax
import equinox as eqx

import matplotlib.pyplot as plt

def to_numpy(x):
    return np.array(x)

def from_numpy(x):
    return jnp.array(x)

class ReplayBuffer():
    """replay buffer"""
    def __init__(self):
        self.rewards = []
        self.observations = []
        self.actions = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_obs(self, obs):
        self.observations.append(obs)

    def add_action(self, act):
        self.actions.append(act)

    @property
    def len(self):
        return len(self.rewards)

    def clear(self):
        self.rewards = []
        self.observations = []
        self.actions = []


class MultivariateNormalDiag(eqx.Module):
    # Gaussian Layer
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
    
    def __init__(self, state_dim, action_dim, key):
        keys = jax.random.split(key, 4)
        self.trunk_layers = [eqx.nn.Linear(state_dim, 16, key=keys[0]), 
                             eqx.nn.Linear(16, 16, key=keys[1]),]
        self.action_mean_head = eqx.nn.Linear(16, action_dim, key=keys[2])
        self.action_std_head = eqx.nn.Linear(16, action_dim, key=keys[3])

    @jax.jit
    def __call__(self, x):
        for layer in self.trunk_layers:
            x = jax.nn.relu(layer(x))
        action_mean = self.action_mean_head(x)
        action_std = jax.nn.softplus(self.action_std_head(x))
        return MultivariateNormalDiag(action_mean, action_std)

def compute_returns(rewards, discount_factor):
    tail_return = 0.
    returns = []
    for r in rewards[::-1]:
        tail_return = r + discount_factor * tail_return
        returns.append(tail_return)
    returns.reverse()
    return np.array(returns)

def train_loss_for_epsiode(policy: Policy, states, actions, returns, num_steps):
    action_distributions = jax.vmap(policy)(states)
    action_log_probs = action_distributions.log_prob(actions)
    advantages = returns
    mask = jnp.arange(max_steps) < num_steps
    actor_loss = jnp.sum(-action_log_probs * jax.lax.stop_gradient(advantages) * mask)
    return actor_loss

@jax.jit
def train_step_for_episode(opt_state, policy, states, actions, returns, num_steps):
    grads = jax.grad(train_loss_for_epsiode)(policy, states, actions, returns, num_steps)
    updates, opt_state = optimizer.update(grads, opt_state)
    policy = optax.apply_updates(policy, updates)
    return opt_state, policy


class exp_arg(object):
  def __init__(self):
    """
        experimental parameters
    """
    self.gamma = 0.98
    self.max_steps = 900
    self.seed = 0
    self.render = bool(0)
    self.log_interval = 5

if __name__ == '__main__':
    render = bool(1)
    render_interval = 50
    replaybuffer = ReplayBuffer()
    # RL environment setup.
    env = gym.make("InvertedPendulum-v4")
    env.reset(seed=0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_steps = 600

    # Training parameters.
    discount_factor = 0.99  # Discount factor for computing tail returns.
    ema_factor = 0.99       # Exponential moving average for standardizing returns.
    key = jax.random.PRNGKey(0)

    key, policy_key = jax.random.split(key)
    policy = Policy(4, 1, policy_key)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(policy)

    episodic_rewards = []
    smooth_episodic_rewards = []
    episodic_reward_ema = 0     # Exponential moving average of episodic rewards.
    return_ema = None           # Exponential moving average of returns (i.e., critic targets).
    return_emv = None           # Exponential moving variance of returns (i.e., critic targets).

    for i_episode in range(3001):
        # Reset environment at the start of each episode, and clear accumulators.
        state = env.reset()
        episodic_reward = 0

        replaybuffer.clear()
        # Sample a trajectory (episode) according to the stochastic policy with environment dynamics.
        for t in range(max_steps):
            replaybuffer.add_obs(state)

            action_distribution = policy(state)
            key, sample_key = jax.random.split(key)
            action = np.array(action_distribution.sample(sample_key))  # Leave JAX to interact with gym.
            state, reward, done, _ = env.step(action)
            episodic_reward += reward

            replaybuffer.add_action(action)

            replaybuffer.add_reward(reward)

            if render and i_episode % render_interval == 0:
                env.render()
            if done:
                break

        # Compute (standardized) tail returns for the episode and update moving averages.
        episodic_rewards.append(episodic_reward)
        returns = compute_returns(replaybuffer.rewards, discount_factor)
        
        return_ema = returns.mean()
        return_emv = returns.var()

        # for smoothing visualization
        episodic_reward_ema = 0.95 * episodic_reward_ema + (1 - 0.95) * episodic_reward
        smooth_episodic_rewards.append(episodic_reward_ema)
        # Normalization
        standardized_returns = (returns - return_ema) / (np.sqrt(return_emv) + 1e-6)

        # Run a train step based on the episode's data.
        num_steps = len(replaybuffer.rewards)

        # JAX prefers all arrays to be the same shape for jitting operation, so pad all the array.
        opt_state, policy = train_step_for_episode(
            opt_state,
            policy,
            np.pad( replaybuffer.observations, ((0, max_steps - num_steps), (0, 0)) ),
            np.pad( replaybuffer.actions, ((0, max_steps - num_steps), (0, 0)) ),
            np.pad( standardized_returns, ((0, max_steps - num_steps),) ),
            num_steps,
        )

        # Periodically log results.
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tLast reward: {episodic_reward:.2f}\tMoving average reward: {episodic_reward_ema:.2f}")

        if episodic_reward_ema > 580:
            break

    plt.figure(1)
    plt.plot(episodic_rewards)
    plt.plot(smooth_episodic_rewards)
    plt.show()