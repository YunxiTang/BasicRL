"""A2C"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import gym
import random, os
import argparse

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


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
        self.next_states = []
        self.dones = []
        
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
        return None

    def add_next_state(self, next_state):
        self.next_states.append(next_state)
        return None

    def add_done(self, done):
        self.dones.append(done)
        
    @property
    def length(self):
        return len(self.rewards)

class ReplayBuffer:
    """replay buffer"""
    def __init__(self, args):
        self.buffer_capacity = args.buffer_capacity
        self.rollouts = []
        self.buffer_size = 0
        
    def add_rollouts(self, path: rollout):
        assert len(path.actions) == len(path.logprobs) == len(path.rewards) == len(path.next_states) == len(path.states) == len(path.dones), \
              'Lengthes of <s, a, a_logprob, r, s_next, done> in path do not match'
        self.rollouts.append(path)
        self.buffer_size += path.length
        return None

    def sample_rollouts(self, agent, env, max_eplen = 200, render: bool = False):
        """sample a batch of rollouts"""
        episode_reward = 0.
        num_traj = 0
        while self.buffer_size < self.buffer_capacity:
            path = rollout()
            state = env.reset()
            steps = 0
            while True:
                # sample a single rollout
                action, action_logpdf = agent.select_action(state)
                path.add_state(state)
                path.add_action(action)
                path.add_logprob(action_logpdf)
                state, reward, done, _, _  = env.step(action)
                
                path.add_reward(reward)
                path.add_next_state(state)
                path.add_done(done)

                if render:
                    # only render the last sampled rollout
                    env.render()
                steps += 1
                episode_reward += reward
                rollout_done = (done or steps == max_eplen)
                if rollout_done:
                    break
            self.add_rollouts(path)
            num_traj += 1
        return episode_reward / num_traj

    def list_to_array(self):
        states_np = np.concatenate([np.array(path.states) for path in self.rollouts])
        actions_np = np.concatenate([np.array(path.actions) for path in self.rollouts])
        logprobs_np = np.concatenate([np.array(path.logprobs) for path in self.rollouts])
        rewards_np = np.concatenate([np.array(path.rewards) for path in self.rollouts])
        next_states_np = np.concatenate([np.array(path.next_states) for path in self.rollouts])
        dones_np = np.concatenate([np.array(path.dones) for path in self.rollouts])
        return states_np, actions_np, logprobs_np, rewards_np, next_states_np, dones_np
    
    @property
    def length(self):
        return len(self.rollouts)
    
    def clear(self):
        self.rollouts = []
        self.buffer_size = 0
        return None

def compute_qvals(rewards, discount_factor):
    """Compute Q values using Monte Carlo estimation
       rewards: rewards_list"""
    tail_return = 0.
    qvals = []
    for r in rewards[::-1]:
        tail_return = r + discount_factor * tail_return
        qvals.append(tail_return)
    qvals.reverse()
    return np.array(qvals)


def compute_rtg(rewards, dones, discount_factor):
    """Compute reward-to-go
       rewards: np.array"""
    rtgs = []
    q = 0
    for r, d in zip(reversed(rewards.flatten()), reversed(dones.flatten())):
        q = r + discount_factor * q * (1.0 - d)
        rtgs.append(q)
    rtgs.reverse()
    return np.array(rtgs)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()
        self.max_action = from_numpy(max_action)
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
        single_scale_tril = torch.diag(self.std.exp())
        batched_std = single_scale_tril.repeat(batch_dim, 1, 1)
        return distributions.MultivariateNormal(batched_mean, scale_tril=batched_std)

    def get_mean_action(self, x):
        """for policy evaluation"""
        return self.mean_net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, value_dim, hidden_dim=64):
        super(Critic, self).__init__()
        assert value_dim == 1, 'output dim of critic net can only be 1'
        self.value_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, value_dim))
        
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer)
    
    def forward(self, x):
        return self.value_net(x)
         

class Agent:
    def __init__(self, args):
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.max_action = from_numpy(args.max_action)
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps

        self.actor = Actor(self.obs_dim, self.act_dim, args.max_action).to(device)
        self.critic = Critic(self.obs_dim, 1).to(device)

        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimzer = optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.critic_lossfunc = nn.MSELoss()
    
    @torch.no_grad()
    def select_action(self, x: np.array):
        if len(x.shape) == 1:
            x = x[None]
        state_tensor = from_numpy(x)
        
        dist = self.actor(state_tensor)
        action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        action_logpdf = dist.log_prob(action)
        return to_numpy(action).reshape(self.act_dim,), to_numpy(action_logpdf)

    @torch.no_grad()
    def evaluate(self, x: np.array):
        state_tensor = from_numpy(x)
        action = self.actor.get_mean_action(state_tensor)
        return to_numpy(action).flatten()

    def update_critic(self,
                     qvals,
                     states_tensor):
        """update the critic net"""
        
        state_vals_pred = self.critic(states_tensor)
        L_critic = self.critic_lossfunc(state_vals_pred, qvals.reshape(state_vals_pred.shape))
        
        self.critic_optimzer.zero_grad()
        critic_loss = L_critic.mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimzer.step()
        return to_numpy(critic_loss)

    def update_actor(self, 
                    advantages,
                    logprobs_new_tensor,
                    logprobs_old_tensor,
                    ):
        """update the actor net"""
        L_actor = -(logprobs_new_tensor * advantages).mean()
        L_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        return None

    def update(self, buffer: ReplayBuffer, gamma: float, train_step:int):
        # update the agent
        assert buffer.length > 0, 'Buffer is empty! Sample some data firstly!'
        states_np, actions_np, logprobs_np, rewards_np, next_states_np, dones_np = buffer.list_to_array()
        states_tensor = from_numpy(states_np)
        actions_tensor = from_numpy(actions_np)
        logprobs_old_tensor = from_numpy( logprobs_np )
        next_states_tensor = from_numpy(next_states_np)
        rewards_tensor = from_numpy(rewards_np)
        dones_tensor = from_numpy( dones_np )

        rtgs = compute_rtg(rewards_np, dones_np, gamma)
        qvals = from_numpy( rtgs )
        # compute advantages
        with torch.no_grad():
            # advantages contain no gradient information
            # Querying critic net which estimate V(s_t) function
            values = self.critic(states_tensor)
            # tmp = qvals
            tmp = rewards_tensor + gamma * (1-dones_tensor) * self.critic(next_states_tensor).squeeze()
            advantages = tmp - values.reshape(tmp.shape)
        # advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        for index in BatchSampler(SubsetRandomSampler(range(buffer.buffer_size)), self.mini_batch_size, True):
            # this is for constructing the computational gragh
            actions_dist = self.actor(states_tensor[index])
            logprobs_new_tensor = actions_dist.log_prob(actions_tensor[index])
            self.update_actor(advantages[index], logprobs_new_tensor, logprobs_old_tensor[index])
            critic_loss = self.update_critic(qvals[index], states_tensor[index])

        self.lr_decay(train_step)
        return critic_loss
    
    def lr_decay(self, train_step):
        lr_a_now = self.lr_a * (1 - train_step / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - train_step / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimzer.param_groups:
            p['lr'] = lr_c_now

def evaluate_policy(env: gym.Env, agent: Agent):
    times = 3
    evaluate_reward = 0
    for i in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _, _ = env.step(action)
            if i == times-1:
                env.render()
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

    
def train(env: gym.Env, agent: Agent, buffer: ReplayBuffer, render=False, max_train_steps=500, max_eplen=1000, gamma=0.98):
    """training"""
    res = []
    for train_step in range(max_train_steps):
        episode_reward = buffer.sample_rollouts(agent, env, max_eplen=max_eplen, render=render)
                
        if train_step == 0:
            smooth_episode_reward = episode_reward
        else:
            smooth_episode_reward = 0.95 * smooth_episode_reward + 0.05 * episode_reward

        res.append(smooth_episode_reward)
        agent.update(buffer, gamma, train_step)
        
        if train_step % 5 == 0:
            print(f'+----------Train Step: {train_step} (Buffer Steps: {buffer.buffer_size}) ----------+')
            print(f'Last Ave. Batch Reward: {episode_reward} || Ave. Reward: {smooth_episode_reward}')
            if train_step % 20 == 0:
                evaluated_reward = evaluate_policy(env, agent)
                print(f'Evaluated Reward: {evaluated_reward}')
            print('+-------------------------------------------------------------------------------+')
        buffer.clear()
    return res


def parse_args(env: gym.Env):
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, default=env.observation_space.shape[0],
                                      help="obs/state dimension")
    parser.add_argument("--act_dim", type=int, default=env.action_space.shape[0],
                                      help="action dimension")
    parser.add_argument("--max_action", type=int, default=env.action_space.high,
                                      help="max action")
    parser.add_argument("--mini_batch_size", type=int, default=100,
                                      help="mini_batch_size")
    parser.add_argument("--buffer_capacity", type=int, default=1500,
                                      help="buffer_capacity")
    parser.add_argument("--render", type=bool, default=False,
                                    help="render or not")
    parser.add_argument("--max_train_steps", type=int, default=500,
                                    help="max training epoch")
                                    

    parser.add_argument("--lr_a", type=float, default=2e-4,
                                  help="actor learning rate")
    parser.add_argument("--lr_c", type=float, default=2e-4,
                                  help="crtic learning rate")
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    
    env = gym.make('InvertedDoublePendulum-v4', new_step_api=True)
    # env = gym.make('HumanoidStandup-v4', new_step_api=True)
    # env = gym.make('Walker2d-v4', new_step_api=True)
    
    args = parse_args(env)
    agent = Agent(args)
    buffer = ReplayBuffer(args)

    res = train(env, agent, buffer, render=args.render, max_train_steps=args.max_train_steps, max_eplen=200)

    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.plot(res, 'k-')
    plt.show()
    