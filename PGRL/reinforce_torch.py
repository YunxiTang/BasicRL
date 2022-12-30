# my implementation of REINFORCE
"""reinforce"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym
import itertools

# utils 
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(*args, **kwargs):
    """put a variable to a device tensor"""
    return torch.from_numpy(*args, **kwargs).float().to('cpu')

class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def add_state(self, state):
        self.states.append(state)
        return None
    
    def add_action(self, action):
        self.actions.append(action)
        return None
    
    def add_reward(self, reward):
        self.rewards.append(reward)
        return None
    
    def clear(self):
        self.states = []
        self.rewards = []
        self.actions = []
        return None

def compute_returns(rewards, discount_factor):
    tail_return = 0.
    returns = []
    for r in rewards[::-1]:
        tail_return = r + discount_factor * tail_return
        returns.append(tail_return)
    returns.reverse()
    return np.array(returns)

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(MLPPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, act_dim)
        )
        self.std = nn.Parameter(torch.zeros((act_dim,), dtype=torch.float32))
        self.optimizer = optim.Adam(itertools.chain(self.mean_net.parameters(),
                                                    [self.std]), lr=1e-2)
        
    def forward(self, x: torch.tensor):
        mean = self.mean_net(x)

        return distributions.MultivariateNormal(loc=mean, scale_tril= torch.diag(self.std.exp()))
    
    def select_action(self, x: torch.tensor):
        dist = self.forward(x)
        action = dist.sample()
        return to_numpy(action)
    
    def update(self, states: np.array, actions: np.array, rewards: np.array, gamma: float):
        # update the policy
        states_tensor = from_numpy( states )
        actions_tensor = from_numpy(actions)
        
        actions_dist = self.forward(states_tensor)
        
        actions_logpdf = actions_dist.log_prob(actions_tensor)
        # compute returns
        returns = from_numpy( compute_returns(rewards, gamma) )
        # normalization
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        L = -(actions_logpdf * returns).sum()
        
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        
    
def train(env: gym.Env, policy: MLPPolicy, buffer: ReplayBuffer, render=False, max_episode=500, max_eplen=1000, Ns=1, gamma=0.98):
    """training"""
    Smooth_Ep_Rew = []
    for episode in range(max_episode):
        episode_reward = 0.
        smooth_episode_reward = 0
        # sampling trajs
        state = env.reset()
        for i in range(max_eplen):
            buffer.add_state(state)
            state_tensor = from_numpy(state)
            action = policy.select_action(state_tensor)
            
            buffer.add_action(action)
            state, reward, done, _  = env.step(action)
            if episode % 20 == 0:
                env.render()
            # reward += -(state[0] - 0.45)**2
            episode_reward += reward
            if i == 1:
                smooth_episode_reward = episode_reward
            else:
                smooth_episode_reward = 0.95 * smooth_episode_reward + 0.05 * episode_reward
            buffer.add_reward(reward)
            if done:
                break
        Smooth_Ep_Rew.append(smooth_episode_reward)
        policy.update(np.array(buffer.states), np.array(buffer.actions), buffer.rewards, gamma)
        
        if episode % 20 == 0:
            print('================================')
            print(f'Episode: {episode} || Last Reward: {episode_reward} || Episode Reward: {smooth_episode_reward}')
        
        buffer.clear()
    return Smooth_Ep_Rew

if __name__ == '__main__':
    policy = MLPPolicy(2, 1)
    env = gym.make('MountainCarContinuous-v0')

    buffer = ReplayBuffer()
    results = train(env, policy, buffer, max_episode=5000)