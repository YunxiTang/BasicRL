"""generate expert data/policy via ppo"""
import torch, gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rl_utils
import numpy as np


ENV_NAME = 'CartPole-v0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    ''' PPO-clip '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  
        self.eps = eps  
        self.device = device

    @torch.no_grad()
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state[None,:])
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            # clip
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage 
            # actor loss 
            actor_loss = torch.mean(-torch.min(surr1, surr2)) 
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def sample_expert_data(env, agent, n_episode):
    states = []
    actions = []
    traj_reward = 0.0
    for _ in range(n_episode):
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            traj_reward += reward
    return np.array(states), np.array(actions, dtype=np.int64), traj_reward

def generate_expert_data():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 250
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2

    env = gym.make(ENV_NAME)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, 
                    hidden_dim, 
                    action_dim, 
                    actor_lr, 
                    critic_lr, 
                    lmbda,
                    epochs, 
                    eps, 
                    gamma, 
                    device)

    return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

    # generate expert data
    env.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    n_episode = 1
    expert_s, expert_a, expert_reward = sample_expert_data(env, ppo_agent, n_episode)
    return expert_s, expert_a, expert_reward
