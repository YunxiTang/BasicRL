"""SAC Algorithm"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import gym
import random, os, sys, copy
import argparse
import time
from logger import Logger
from collections import OrderedDict

def orthogonal_init(layer, gain=np.sqrt(2.)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def set_seeds(env:gym.Env, seed = 1):
    ''' 
        set seeds
    '''
    if seed == 0:
        return
    # env config
    # env.seed(seed) # deprecated
    np.random.seed(seed)
    random.seed(seed)
    # config for CPU
    torch.manual_seed(seed) 
    # config for GPU
    torch.cuda.manual_seed(seed) 
    # config for python scripts
    os.environ['PYTHONHASHSEED'] = str(seed) 
    # config for cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


# utils 
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(x):
    """put a variable to a device tensor"""
    return torch.from_numpy(x).float().to(device)


class ReplayBuffer(object):
    """simple replay buffer for SAC"""
    def __init__(self, args):
        obs_dim = args.obs_dim
        act_dim = args.act_dim

        self.max_size = int(1e4)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, obs_dim))
        self.a = np.zeros((self.max_size, act_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, obs_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        # count reaches max_size, it will be reset to 0.
        self.count = (self.count + 1) % self.max_size  
        # Record the number of  transitions
        self.size = min(self.size + 1, self.max_size)  

    def sample(self, batch_size):
        # Randomly sampling
        index = np.random.choice(self.size, size=batch_size)  
        batch_s = from_numpy(self.s[index])
        batch_a = from_numpy(self.a[index])
        batch_r = from_numpy(self.r[index])
        batch_s_ = from_numpy(self.s_[index])
        batch_dw = from_numpy(self.dw[index])
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def sample_rollouts(self, env:gym.Env, agent, max_episode_steps):
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, _ = agent.select_action(s)
            s_, r, done, _, _ = env.step(a)

            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            # Store the transition
            self.store(s, a, r, s_, dw)  
            s = s_


class Actor(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, 
                 max_action:torch.Tensor, hidden_dim=128):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.mean_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, act_dim), nn.Tanh())
        self.std = nn.Parameter(torch.zeros((act_dim, ), dtype=torch.float32))
        for layer in self.mean_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer)
        
    def forward(self, x):
        batched_mean = self.mean_net(x)
        batched_mean = batched_mean * self.max_action
        batch_dim = batched_mean.shape[0]
        std = torch.clamp(self.std, -20., 2.)
        single_scale_tril = torch.diag(std.exp())
        batched_std = single_scale_tril.repeat(batch_dim, 1, 1)
        return distributions.MultivariateNormal(batched_mean, scale_tril=batched_std)

    def get_mean_action(self, x):
        """for policy evaluation"""
        return self.mean_net(x) * self.max_action


class Critic(nn.Module):  
    # double Q-network: Q(s,a)
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # Q1
        self.Q1_net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))
        # Q2
        self.Q2_net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))
        
        for layer in self.Q1_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer)

        for layer in self.Q2_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer)
        

    def forward(self, s, a):
        if isinstance(s, np.ndarray):
            s = from_numpy(s)
        if isinstance(a, np.ndarray):
            a = from_numpy(a[:,None])
        
        s_a = torch.cat([s, a], 1)
        q1 = self.Q1_net(s_a)
        q2 = self.Q2_net(s_a)
        return q1, q2


class Agent:
    def __init__(self, args):
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.max_action = from_numpy(args.max_action)
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps

        self.GAMMA = 0.95
        self.TAU = 0.005

        # Target Entropy
        self.target_entropy = -self.act_dim
        # learn log_alpha to ensure: alpha=exp(log_alpha) > 0
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.actor = Actor(self.obs_dim, self.act_dim, self.max_action)
        self.critic = Critic(self.obs_dim, self.act_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
    
    @torch.no_grad()
    def select_action(self, x):
        if len(x.shape) == 1:
            x = x[None]
        if isinstance(x, np.ndarray):
            state_tensor = from_numpy(x)
        else:
            state_tensor = x
        dist = self.actor(state_tensor)
        action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        action_logpdf = dist.log_prob(action)
        return to_numpy(action).flatten(), to_numpy(action_logpdf).flatten()

    @torch.no_grad()
    def evaluate(self, x):
        if len(x.shape) == 1:
            x = x[None]
        if isinstance(x, np.ndarray):
            state_tensor = from_numpy(x)
        else:
            state_tensor = x
        action = self.actor.get_mean_action(state_tensor)
        return to_numpy(action).flatten()

    def update(self, buffer: ReplayBuffer):
        # Sample a mini-batch
        batch_s, batch_a, batch_r, batch_s_, batch_dw = buffer.sample(self.mini_batch_size)  

        #============================Q-net update=================================#
        # Compute current Q values
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
 
        with torch.no_grad():
            # a_ from the current policy
            batch_a_, log_pi_ = self.select_action(batch_s_)  
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - (self.alpha * log_pi_)[:,None])
        
        
        # Compute Q loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize Q-network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #============================Policy-net update=================================#
        # Freeze critic networks to save computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        
        actions_dist = self.actor(batch_s)
        a = actions_dist.rsample()
        log_pi = actions_dist.log_prob(a).sum()
        log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum()
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        return None

def evaluate_policy(env: gym.Env, agent: Agent, max_episode_step, render: bool = False):
    times = 3
    evaluate_reward = 0
    for i in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        for _ in range(max_episode_step):
            action = agent.evaluate(s)
            s_, r, done, _, _ = env.step(action)
            if i == times-1:
                env.render()
            episode_reward += r
            s = s_
            if done:
                break
        evaluate_reward += episode_reward
    return evaluate_reward / times

def train(env:gym.Env, agent:Agent, buffer:ReplayBuffer, args):
    """helper function of training"""
    for train_step in range(args.max_train_steps):
        buffer.sample_rollouts(env, agent, args.max_episode_steps)
        if train_step > 200:
            for i in range(25):
                log_info = agent.update(buffer)
        
        # loggings
        # exp_logger.log_scalar(episode_reward, "Train_AverageReturn", train_step)
        # exp_logger.log_scalar(log_info['critic loss'], 'Train_CriticLoss', train_step)
        
        if train_step % 50 == 0:
            evaluated_reward = evaluate_policy(env, agent, args.max_episode_steps)
            print(f'+----------------------------------Train Step: {train_step}------------------------------------+\n') 
            print(f'Evaluated Reward: {evaluated_reward}')
            print('+-------------------------------------------------------------------------------+\n') 
    return None

def parse_args(env: gym.Env):
    """parameter settings"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, default=env.observation_space.shape[0],
                                      help="obs/state dimension")
    parser.add_argument("--act_dim", type=int, default=env.action_space.shape[0],
                                      help="action dimension")
    parser.add_argument("--max_action", type=int, default=env.action_space.high,
                                      help="max action")
    parser.add_argument("--mini_batch_size", type=int, default=50,
                                      help="mini_batch_size")
    parser.add_argument("--buffer_capacity", type=int, default=2000,
                                      help="buffer_capacity")
    
    parser.add_argument("--max_episode_steps", type=int, default=1000,
                                    help="max episode length")
    
    parser.add_argument("--max_train_steps", type=int, default=int(2e6),
                                    help="max training epoch")

    parser.add_argument("--gae", type=bool, default=True,
                                    help="use gae or not")
    parser.add_argument("--lamda", type=float, default=0.0,
                                    help="gae factor") # lamda=0, gae turns to be normal case
    parser.add_argument("--gamma", type=float, default=0.99,
                                    help="discount factor")
    parser.add_argument("--K_epochs", type=int, default=5,
                                    help="PPO inner update steps")
    parser.add_argument("--lr_a", type=float, default=2e-4,
                                  help="actor learning rate")
    parser.add_argument("--lr_c", type=float, default=2e-4,
                                  help="crtic learning rate")
    
    parser.add_argument("--render", type=bool, default=False,
                                    help="render or not")
    parser.add_argument("--evaluate", type=bool, default=False,
                                  help="evaluate_policy or not")
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    
    env_platforms = ['InvertedDoublePendulum-v4', 
                     'HumanoidStandup-v4', 
                     'Walker2d-v4']
    idx = 0
    env = gym.make(env_platforms[idx], new_step_api=True)
    set_seeds(env, seed=10086)

    args = parse_args(env)
    agent = Agent(args)
    buffer = ReplayBuffer(args)
 
    # data_path = os.path.dirname(os.path.realpath(__file__)) + '\data'

    # if not (os.path.exists(data_path)):
    #     os.makedirs(data_path)
    # print(data_path)
    # logdir_prefix = 'tyx' 
    # logdir = logdir_prefix + '_' + env_platforms[idx] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    # logdir = os.path.join(data_path, logdir)
    # if not(os.path.exists(logdir)):
    #     os.makedirs(logdir)

    # exp_logger = Logger(logdir)
    train(env, agent, buffer, args)
