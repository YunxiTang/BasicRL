"""reinforce"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym
import itertools


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# utils 
def to_numpy(tensor):
  return tensor.to('cpu').detach().numpy()

def from_numpy(x):
  """put a variable to a device tensor"""
  return torch.from_numpy(x).float().to(device)

class rollout:
    """a complete trajectory"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        
    def add_state(self, state):
        self.states.append(state)
        return None
    
    def add_action(self, action):
        self.actions.append(action)
        return None
    
    def add_logprob(self, logprob):
        self.logprobs.append(logprob)
        return None
    
    def add_reward(self, reward):
        self.rewards.append(reward)
        
    @property
    def length(self):
        return len(self.rewards)

class ReplayBuffer:
    """replay buffer"""
    def __init__(self):
        self.rollouts = []
        
    def add_rollouts(self, rollout):
        self.rollouts.append(rollout)
        return None

    def sample_rollouts(self, agent, env, max_eplen = 200, Ns=1, render: bool = False):
        """sample Ns rollouts"""
        episode_reward = 0.
        for ns in range(Ns):
            path = rollout()
            state = env.reset()
            for i in range(max_eplen):
                # sample a single rollouts
                path.add_state(state)
                action = agent.select_action(state)
                path.add_action(action)
                state, reward, done, _, _  = env.step(action)
                if render and ns==(Ns-1):
                    env.render()
                
                episode_reward += reward
                path.add_reward(reward)
                if done:
                    self.add_rollouts(path)
                    break
        return episode_reward
    
    @property
    def length(self):
        return len(self.rollouts)
    
    def clear(self):
        self.rollouts = []
        return None

def compute_returns(rewards, discount_factor):
    tail_return = 0.
    returns = []
    for r in rewards[::-1]:
        tail_return = r + discount_factor * tail_return
        returns.append(tail_return)
    returns.reverse()
    return np.array(returns)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.mean_net = nn.Sequential(nn.Linear(obs_dim, 16), nn.ReLU(),
                                      nn.Linear(16, 16), nn.ReLU(),
                                      nn.Linear(16, act_dim))
        self.std = nn.Parameter(torch.zeros((act_dim, ), dtype=torch.float32))
        
    def forward(self, x):
        batched_mean = self.mean_net(x)
        batch_dim = batched_mean.shape[0]
        single_scale_tril = torch.diag(self.std.exp())
        batched_std = single_scale_tril.repeat(batch_dim, 1, 1)
        return distributions.MultivariateNormal(batched_mean, scale_tril=batched_std)

class Critic(nn.Module):
    def __init__(self, obs_dim, value_dim):
        super(Critic, self).__init__()
        assert value_dim == 1, 'output dim of critic net can only be 1'
        self.value_net = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(),
                                       nn.Linear(32, 32), nn.ReLU(),
                                       nn.Linear(32, value_dim))
    
    def forward(self, x):
        return self.value_net(x)
         

class Agent:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, 1).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimzer = optim.Adam(self.critic.parameters(), lr=5e-4)
        
        self.critic_lossfunc = nn.MSELoss()
    
    @torch.no_grad()
    def select_action(self, x: np.array):
        state_tensor = from_numpy(x)
        dist = self.actor(state_tensor)
        action = dist.sample()
        return to_numpy(action).reshape(self.act_dim,)
    
    def update(self, buffer: ReplayBuffer, gamma: float, use_advatage: bool = False):
        # update the policy
        assert buffer.length > 0, 'Buffer is empty! Sample some data firstly!'
        actor_losses = []
        critic_losses = []
        for path in buffer.rollouts:
            states = np.array(path.states)
            actions = np.array(path.actions)
            rewards = np.array(path.rewards)
            
            states_tensor = from_numpy(states)
            actions_tensor = from_numpy(actions)
        
            actions_dist = self.actor(states_tensor)
        
            actions_logpdf = actions_dist.log_prob(actions_tensor)
            # compute returns
            returns = from_numpy( compute_returns(rewards, gamma) )

            # normalization
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            if use_advatage:
                val_pred = self.critic(states_tensor)
                L_critic = self.critic_lossfunc(val_pred, returns.reshape(val_pred.shape))
                critic_losses.append(L_critic)

                # compute advantages, detach the val_pred from computational gragh
                values = val_pred.detach()
              
                advantages = returns - values.squeeze()
            else:
                advantages = returns

            L_actor = -(actions_logpdf * advantages).sum()
            actor_losses.append(L_actor)
            
        if use_advatage:
            self.critic_optimzer.zero_grad()
            critic_loss = sum(critic_losses).mean()
            critic_loss.backward()
            self.critic_optimzer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = sum(actor_losses).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        return None
        
        
    
def train(env: gym.Env, agent: Agent, buffer: ReplayBuffer, render=False, use_advatage=False, max_episode=500, max_eplen=1000, Ns=10, gamma=0.98):
    """training"""
    res = []
    for episode in range(max_episode):
        episode_reward = buffer.sample_rollouts(agent, env, max_eplen=max_eplen, Ns=Ns, render=render)
                
        if episode == 0:
            smooth_episode_reward = episode_reward
        else:
            smooth_episode_reward = 0.95 * smooth_episode_reward + 0.05 * episode_reward

        res.append(smooth_episode_reward/Ns)
        agent.update(buffer, gamma, use_advatage)
        
        if episode % 50 == 0:
            print('================================')
            print(f'Episode: {episode} || buffer_len: {buffer.length} || Last Reward: {episode_reward/Ns} || Episode Reward: {smooth_episode_reward/Ns}')
        
        buffer.clear()
    return res
    
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    agent = Agent(11, 1)
    env = gym.make('InvertedDoublePendulum-v4', new_step_api=True)

    buffer = ReplayBuffer()
    res = train(env, agent, buffer, Ns=10, render=False, max_episode=int(5e3), use_advatage=bool(0))
    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.plot(res, 'k-')
    plt.show()
    