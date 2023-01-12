"""A2C"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gym
import random, os


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

class rollout:
    """a complete trajectory"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        
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
        
    @property
    def length(self):
        return len(self.rewards)

class ReplayBuffer:
    """replay buffer"""
    def __init__(self, max_capacity: int = 1000):
        self.rollouts = []
        self.steps = 0
        
    def add_rollouts(self, path: rollout):
        assert len(path.actions) == len(path.logprobs) == len(path.rewards) == len(path.next_states) == len(path.states), \
              'Lengthes of <s, a, a_logprob, r, s_next> in path do not match'
        self.rollouts.append(path)
        return None

    def sample_rollouts(self, agent, env, max_eplen = 200, Ns=1, render: bool = False):
        """sample Ns rollouts"""
        episode_reward = 0.
        for ns in range(Ns):
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
                self.steps += 1
                steps += 1
                path.add_reward(reward)
                path.add_next_state(state)

                if render and ns==(Ns-1):
                    # only render the last sampled rollout
                    env.render()
                
                episode_reward += reward
                rollout_done = (done or steps == max_eplen)
                if rollout_done:
                    break

            self.add_rollouts(path)
        return episode_reward / Ns
    
    @property
    def length(self):
        return len(self.rollouts)
    
    def clear(self):
        self.rollouts = []
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

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.mean_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
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
        self.value_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, value_dim))
    
    def forward(self, x):
        return self.value_net(x)
         

class Agent:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, 1).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimzer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        self.critic_lossfunc = nn.MSELoss()
    
    @torch.no_grad()
    def select_action(self, x: np.array):
        if len(x.shape) == 1:
            x = x[None]
        state_tensor = from_numpy(x)
        
        dist = self.actor(state_tensor)
        action = dist.sample()
        
        action_logpdf = dist.log_prob(action)
        return to_numpy(action).reshape(self.act_dim,), to_numpy(action_logpdf)

    def update_critic(self, buffer: ReplayBuffer, gamma: float):
        """update the critic net"""
        critic_losses = []
        for path in buffer.rollouts:
            states = np.array(path.states)
            rewards = np.array(path.rewards)
            states_tensor = from_numpy(states)
            # compute Q(s, a) values
            qvals = from_numpy( compute_qvals(rewards, gamma) )
            # Querying critic net which estimate V(s_t) function
            state_vals_pred = self.critic(states_tensor)
            # Critic Loss
            # Monte-Carlo Estimation of V(s_t)
            val_target = qvals
            L_critic = self.critic_lossfunc(state_vals_pred, val_target.reshape(state_vals_pred.shape))
            critic_losses.append(L_critic)  
            
        self.critic_optimzer.zero_grad()
        critic_loss = sum(critic_losses).mean()
        critic_loss.backward()
        self.critic_optimzer.step()
        return None

    def update_actor(self, buffer: ReplayBuffer, gamma: float, use_advatage: bool = False):
        """update the actor net"""
        actor_losses = []
        for path in buffer.rollouts:
            states = np.array(path.states)
            actions = np.array(path.actions)
            action_logprobs = np.array(path.logprobs)
            rewards = np.array(path.rewards)
            next_states = np.array(path.next_states)
            
            states_tensor = from_numpy(states)
            actions_tensor = from_numpy(actions)
            action_logprobs_tensor = from_numpy(action_logprobs)
            next_states_tensor = from_numpy(next_states)
            rewards_tensor = from_numpy(rewards)

            # this is for construct the computational gragh
            actions_dist = self.actor(states_tensor)
            actions_logpdf = actions_dist.log_prob(actions_tensor)

            # compute Q(s, a) values
            qvals = from_numpy( compute_qvals(rewards, gamma) )

            if use_advatage:
                # Querying critic net which estimate V(s_t) function
                state_vals_pred = self.critic(states_tensor)
                # compute advantages, detach the val_pred from computational gragh
                # advantages contain no gradient information
                values = state_vals_pred.detach().clone()
                tmp = rewards_tensor + gamma * self.critic(next_states_tensor).detach().reshape(rewards_tensor.shape)
                advantages = tmp - values.reshape(rewards_tensor.shape)
    
            else:
                # just set the advantage to [Q]s
                advantages = qvals

            # Advantages Normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            L_actor = -(actions_logpdf * advantages).sum()
            actor_losses.append(L_actor)

        self.actor_optimizer.zero_grad()
        actor_loss = sum(actor_losses).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        return None

    def update(self, buffer: ReplayBuffer, gamma: float, use_advatage: bool = False):
        # update the policy
        assert buffer.length > 0, 'Buffer is empty! Sample some data firstly!'
        self.update_actor(buffer, gamma, use_advatage)
        for _ in range(5):
            self.update_critic(buffer, gamma)
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

        res.append(smooth_episode_reward)
        agent.update(buffer, gamma, use_advatage)
        
        if episode % 50 == 0:
            print(f'+---------- Episode: {episode} (Env Steps: {buffer.steps}) ----------+')
            print(f'Traj_num: {buffer.length} || Last Reward: {episode_reward} || Episode Reward: {smooth_episode_reward}')
            print('+----------------------------------------------------+')
        buffer.clear()
    return res
    
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    
    env = gym.make('InvertedDoublePendulum-v4', new_step_api=True)
    # env = gym.make('HumanoidStandup-v4', new_step_api=True)
    # env = gym.make('Walker2d-v4', new_step_api=True)
    
    act_shape = env.action_space.shape
    obs_shape = env.observation_space.shape

    agent = Agent(obs_shape[0], act_shape[0])

    buffer = ReplayBuffer()
    res = train(env, agent, buffer, Ns=1, render=bool(0), max_episode=int(5e3), max_eplen=200, use_advatage=bool(1))

    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.plot(res, 'k-')
    plt.show()
    