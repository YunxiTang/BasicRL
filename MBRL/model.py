"""parameterized model"""
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.distributions as distributions
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
from typing_extensions import Any

##############################################################
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class CLF_QP:
    """CLf-QP safety filter"""
    def __init__(self, goal_shape) -> None:
        self.goal_shape = goal_shape
        pass

    def Lyapunov_func(self, x):
        """Lyapunov function V(x)"""
        return torch.sum( (x - self.goal_shape) * (x - self.goal_shape), dim=1 )

    def Lya_grad(self, x):
        """Lyapunov function gradient w.r.t x"""
        return x - self.goal_shape

    def solve(self, state, uref):
        pass


class MPPI:
    """sampling-based MPC controller: model predictive path integral method"""
    def __init__(self):
        pass

    def optimize(self, state):
        pass

class JacobianModel(nn.Module):
    """Kinematic Jacobian Model"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super(JacobianModel, self).__init__()
        self.mlp_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, output_dim))
        self.reg_optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.reg_loss = nn.MSELoss()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, torch.float32)
        jac = self.mlp_net(x)
        return jac

    def loss_calc(self, feats, labels):
        y_hat = self(feats)
        loss = self.reg_loss(y_hat, labels.reshape(y_hat.shape))
        return loss

    def update(self, loss):
        self.reg_optimizer.zero_grad()
        loss.backward()
        self.reg_optimizer.step()
        return loss.data.numpy()

class DOM_Actor(nn.Module):
    """Actor for Deformable Object Manipulation"""
    def __init__(self, obs_dim, act_dim, goal_shape, hidden_dim=128):
        super(DOM_Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_shape = goal_shape

        self.kin_model = JacobianModel(obs_dim, obs_dim * act_dim, hidden_dim=hidden_dim)

        self.mpc = MPPI()
        self.clf_qp = CLF_QP(goal_shape=goal_shape)

        # Gaussian std
        self.std = nn.Parameter(torch.zeros((act_dim, ), dtype=torch.float32))

    def forward(self, x, uref=None):
        """make sure every operation supports mini-batching"""
        if uref is None:
            uref = torch.zeros((self.act_dim, 1))

        batch_dim = x.shape[0]
        batched_J = self.kin_model(x)
        batched_J = batched_J.reshape((batch_dim, self.obs_dim, self.act_dim))

        batched_uref_val = uref.repeat(batch_dim, 1, 1)
        batched_H_sqrtval = self.H_sqrtval.repeat(batch_dim, 1, 1)

        batched_V_val = self.Lyapunov_func(x).reshape((batch_dim, 1))
        batched_dVdx = self.L_grad(x).reshape((batch_dim, 1, self.obs_dim))

        # print('===============shape info:============')
        # print(batched_V_val.shape)  # {batch x 1}
        # print(batched_dVdx.shape)   # {batch x 1 x nx}
        # print(batched_fx.shape)     # {batch x nx x 1}
        # print(batched_gx.shape)     # {batch x nx x nu}
        # print('======================================')

        batched_LgV_val = torch.bmm(batched_dVdx, batched_J).reshape((batch_dim, 1, self.act_dim))

        batched_u, _, _ = self.diff_qp(batched_H_sqrtval, 
                                    batched_uref_val, 
                                    batched_V_val,
                                    batched_LgV_val)

        single_scale_tril = torch.diag(self.std.exp())
        batched_std = single_scale_tril.repeat(batch_dim, 1, 1)
        batched_dist = distributions.MultivariateNormal(batched_u, scale_tril=batched_std)
        return batched_dist


class Agent:
    def __init__(self, input_dim, output_dim) -> None:
        self.approx_model = JacobianModel(input_dim, output_dim)
        self.mpc = MPPI()
        self.clf_qp = CLF_QP()

    def select_action(self, state):
        uref = self.mpc.optimize(state)
        u = self.clf_qp.solve(state, uref)

    def update_JacModel(self, feats, labels):
        """update the Jacobian model via regression"""
        loss = self.approx_model.loss_calc(feats, labels)
        self.approx_model.update(loss)
    





if __name__ == '__main__':
    def data_generation(w: torch.tensor, b: torch.tensor, num_samples: int):
        x_dim = w.shape[0]
        x = torch.normal(0., 2.0, (num_samples, x_dim))
        y = 5. * torch.sin( torch.matmul(x, w) ) + b 
        y += torch.normal(0., 0.2, y.shape)
        return x, y

    def data_iter(features, labels, batch_size):
        num_examples = len(labels)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i: min(i + batch_size, num_examples)]
            yield features[batch_indices], labels[batch_indices]

    w_true = torch.tensor([1.5,])
    b_true = torch.tensor([1.2,])
    x, y = data_generation(w_true, b_true, 2000)

    batch_size = 64

    # build a model and initialization
    model = JacobianModel(1, 1, hidden_dim=50)

    for epoch in range(50):
        for feat, label in data_iter(x, y, batch_size):
            loss = model.loss_calc(feat, label)
            loss_data = model.update(loss)
        print('epoch {} || loss {}'.format(epoch+1, loss_data))
    
    plt.figure(1)
    plt.scatter(x, y)
    plt.scatter(x, model(x).detach().numpy())
    plt.show()
        
        
