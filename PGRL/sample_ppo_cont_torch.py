"""continuous ppo"""
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import random, os
from collections import deque
import pickle


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def set_seeds(env, seed = 1):
    ''' 
        set seeds
    '''
    if seed == 0:
        return
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# utils 
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(x):
    """put a variable to a device tensor"""
    return torch.from_numpy(x).float().to(device)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.mean_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, act_dim))
        self.std = nn.Parameter(torch.zeros((act_dim, ), dtype=torch.float32))
        
    def forward(self, x):
        batched_mean = self.mean_net(x)
        batch_dim = batched_mean.shape[0]
        single_scale_tril = torch.diag(self.std.exp())
        batched_std = single_scale_tril.repeat(batch_dim, 1, 1)
        return distributions.MultivariateNormal(batched_mean, scale_tril=batched_std)

class Critic(nn.Module):
    def __init__(self, obs_dim, value_dim, hidden_dim=256):
        super(Critic, self).__init__()
        assert value_dim == 1, 'output dim of critic net can only be 1'
        self.value_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, value_dim))
    
    def forward(self, x):
        return self.value_net(x)


class ReplayBufferQue:
    '''replay buffer for DQN'''
    def __init__(self, capacity: int = int(1e5)):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        '''
            add trainsitions: tuple
        '''
        self.buffer.append(transitions)

    def sample(self, 
               batch_size: int, 
               sequential: bool = False):

        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # sequential sampling
        if sequential: 
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)

        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class ReplayBufferPG(ReplayBufferQue):
    '''
        repaly buffer for PG inherited from ReplayBufferQue
    '''
    def __init__(self):
        self.buffer = deque()

    def sample(self):
        ''' 
            sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

class Agent:
    """rl agent"""
    def __init__(self, config):
        """
            config: experimental configuration
        """
        self.gamma = config.gamma
        self.device = torch.device(config.device) 
        self.act_dim = config.act_dim
        self.actor = Actor(config.state_dim, config.act_dim, hidden_dim = config.actor_hidden_dim).to(self.device)
        self.critic = Critic(config.state_dim, 1, hidden_dim=config.critic_hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.buffer = ReplayBufferPG()
        self.k_epochs = config.k_epochs         # update policy for K epochs
        self.eps_clip = config.eps_clip         # clip parameter for PPO
        self.entropy_coef = config.entropy_coef # entropy coefficient

        self.step_count = 0
        self.update_freq = config.update_freq

    def sample_action(self, state):
        self.step_count += 1
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        dist = self.actor(state_tensor)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().reshape(self.act_dim,)

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        dist = self.actor(state)
        action = dist.sample()
        return action.detach().cpu().numpy().reshape(self.act_dim,)

    def update(self):
        # update policy every n steps
        if self.step_count % self.update_freq != 0:
            return
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.buffer.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) 
            # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            dist = self.actor(old_states)

            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            # old_log_probs must be detached
            ratio = torch.exp(new_probs - old_log_probs) 
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.buffer.clear()

import copy
def train(cfg, env, agent):
    ''' 
        training
    '''
    print("=================")
    # rewards trace
    rewards = []  
    steps = []
    # max episode reaards
    best_ep_reward = 0 
    output_agent = None

    for i_ep in range(cfg.train_eps):
        # reward for one episode
        ep_reward = 0  
        ep_step = 0

        # reset
        state = env.reset()  
        for _ in range(cfg.max_steps):
            ep_step += 1
            # sample an action
            action = agent.sample_action(state)  
            # leave to interact with env，return a transition tupel 
            next_state, reward, done, _ = env.step(action)  
            
            # save the transition in replay buffer
            agent.buffer.push((state, action, agent.log_probs, reward, done)) 
            # update the env state
            state = next_state  
            # update the agent
            agent.update()  
            # accumulating rewards
            ep_reward += reward  
            if done:
                break

        if (i_ep+1) % cfg.eval_per_episode == 0:
            # evalutaion
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                state = env.reset()
                for _ in range(cfg.max_steps):
                    # sample an action
                    action = agent.predict_action(state)  
                    next_state, reward, done, _ = env.step(action)  
                    env.render()
                    state = next_state  
                    eval_ep_reward += reward 
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward/cfg.eval_eps
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"episode: {i_ep+1}/{cfg.train_eps}, \
                       ep_reward: {ep_reward:.2f}, \
                       mean_eval_reward: {mean_eval_reward:.2f}, \
                       best_eval_reward: {best_ep_reward:.2f}")
            else:
                print(f"episode: {i_ep+1}/{cfg.train_eps}, \
                       ep_reward: {ep_reward:.2f}, \
                       mean_eval_reward: {mean_eval_reward:.2f}, \
                       best_eval_reward: {best_ep_reward:.2f}")
        steps.append(ep_step)
        rewards.append(ep_reward)
    print("Done with training...")
    env.close()
    return output_agent, {'rewards':rewards}

def all_seed(env, seed = 1):
    ''' 
        set seeds
    '''
    if seed == 0:
        return
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    env = gym.make(cfg.env_name)
    all_seed(env,seed=cfg.seed)
    act_shape = env.action_space.shape
    obs_shape = env.observation_space.shape
    setattr(cfg, 'state_dim', obs_shape[0])
    setattr(cfg, 'act_dim', act_shape[0]) 
    agent = Agent(cfg)
    return env, agent


class Config:
    def __init__(self) -> None:
        self.env_name = "InvertedDoublePendulum-v4" # 环境名字
        self.new_step_api = False # 是否用gym的新api
        self.algo_name = "PPO" # 算法名字
        self.mode = "train" # train or test
        self.seed = 1 # 随机种子
        self.device = "cpu" # device to use
        self.train_eps = 5000 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        self.max_steps = 1000 # 每个回合的最大步数
        self.eval_eps = 5 # 评估的回合数
        self.eval_per_episode = 10 # 评估的频率

        self.gamma = 0.99 # 折扣因子
        self.k_epochs = 4 # 更新策略网络的次数
        self.actor_lr = 0.0003 # actor网络的学习率
        self.critic_lr = 0.0003 # critic网络的学习率
        self.eps_clip = 0.2 # epsilon-clip
        self.entropy_coef = 0.01 # entropy的系数
        self.update_freq = 100 # 更新频率
        self.actor_hidden_dim = 256 # actor网络的隐藏层维度
        self.critic_hidden_dim = 256 # critic网络的隐藏层维度

def smooth(data, weight=0.9):  
    '''smooth the plotted curve
    '''
    last = data[0] 

    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 
        plotter
    '''
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    import gym
    import os
    import numpy as np
    cfg = Config() 
    # train
    env, agent = env_agent_config(cfg)
    best_agent,res_dic = train(cfg, env, agent)
    plot_rewards(res_dic['rewards'], cfg, tag="train")  
    