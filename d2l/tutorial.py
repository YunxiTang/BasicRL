import matplotlib.pyplot as plt
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F

def plot(xs, ys, xlim=(-3, 3), ylim=(-3, 3)):
    """self-maintained plot function"""
    fig, ax = plt.subplots()
    ax.plot(xs, ys, linewidth=5)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.show()
    
def set_device(num: int):
    has_gpu = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()
    if has_gpu and num < num_gpu:
        device = torch.device(f'cuda:{num}')
    else:
        device = torch.device('cpu')
    return device
    
def split_line(info=None):
    if info is None:
        print('==============')
    else:
        print(f'====={info}=====')
    
def run_part1():
    """numpy review"""
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([[8, 0, 7], [3, 0, 1]])
    z = np.random.rand(3, 2, 3)
    print(f"x, shape={x.shape}:\n{x}\n")
    print(f"y, shape={y.shape}:\n{y}\n")
    print(f"z, shape={z.shape}:\n{z}\n")
    split_line()
    
    #  ----------- index into numpy array -----------
    x = np.arange(1, 13, 1).reshape(3, 4)
    print(x)
    print(x[0,:])
    y = x[0:3, 1:3]
    print(y, y.shape)
    split_line()
    
    # ----------- Numpy arithmetic and broadcasting ---
    # can add/subtract/multiple/divide numpy arrays, 
    # as long as their dimensions match
    x = np.ones((2, 3))
    y = np.arange(1, 7, 1).reshape(2, 3)
    print(f'{x}\n + \n{y} =\n {x + y}')
    print(f'{x}\n - \n{y} =\n {x - y}')
    split_line()
    
    # broadcasting in numpy
    x = np.ones((1, 1, 2, 3))
    y = np.array([1, 2, 3])
    print(f'x.shape {x.shape} || y.shape {y.shape}')
    print(f'{x}\n +\n {y}\n =\n {x + y}')
    split_line()
    
    # axes in numpy
    x = np.array([[8, 7, 4], 
                  [5, 2, 2], 
                  [1, 6, 3]])
    print(np.sum(x, axis=0))
    print(np.sum(x, axis=1))
    
    print(np.max(x, axis=0))
    print(np.max(x, axis=1))
    split_line()
    
    # shapes and reshaping
    # (10, ) and (10, 1) are very different!!
    x = np.random.randint(5, 12, size=(10,))
    y = np.random.randint(5, 12, size=(10, 1))
    print(f'x shape: {x.shape} || y shape: {y.shape}')
    print(f'x: {x}\ny: {y}')
    print(f'x * y = \n{x * y}')
    
    # reshape to do element-wise operation
    x_reshaped = x.reshape(-1, 1)
    print(f'x_reshaped * y = \n{x_reshaped * y}')
    
    # Removing axis/axes from an array with squeeze
    z = y.squeeze()
    print(f'y shape: {y.shape}', f'z shape: {z.shape}')
    x = np.random.randint(0, 10, size=(3, 1, 1, 2, 2))
    print(f'x:\n{x}')
    y = x.reshape(3, 2, 2)
    # squeeze #1 and #2 axis
    z = x.squeeze((1, 2)) 
    print(f'x reshaped:\n{y}')
    print(f'x squeezed:\n{z}')
    
    # Adding axis with 'np.newaxis' or None
    print(x.shape)
    print(x[None, :].shape, x.shape)
    # '...' means the rest of dimensions
    print(x[np.newaxis, np.newaxis, ...].shape, x.shape)
    
    split_line()
    
    # array multipication
    # elemen-wise (*) and matrix operation (@)
    matrix = np.random.randint(10, size=(5, 5))
    # note that this is (1-dim) array! cannot be viewed as matrix (2-dim) directly
    row_vec = np.random.randint(10, size=(5,)) 
    col_vec = np.random.randint(10, size=(5, 1))
    
    print(f'matrix @ col_vec=\n{matrix @ col_vec}')
    print(f'matrix * row_vec=\n{matrix * row_vec}')
    
    # element-wise (broadcasting)
    print(f'row_vec * col_vec =\n{row_vec * col_vec}')
    
    # matrix product (get a (2, 2) matrix)
    print(f'row_vec @ col_vec =\n{row_vec[None,:] @ col_vec}')
    # get a scalar (0-dim)
    print(f'row_vec @ col_vec =\n{row_vec @ col_vec.squeeze()}')
    
    split_line()
    # transpose
    z = np.random.randint(10, size=(2,1,3))
    z_t = np.transpose(z)
    z_T = z.T
    print(f'z:\n{z}\nz_t:{z_t}\nz_T{z_T}')
    return None
    
def run_part2():
    """pytorch basics"""
    x = torch.zeros(2, 3)
    y = torch.ones(2, 3)
    print(x + y)
    split_line()
    
    # reduction operation works with 'dim'
    print(torch.sum(y, dim=0))
    print(torch.sum(y, dim=1))
    split_line()
    
    # broadcast also in torch
    x = torch.ones(3, 1)
    y = torch.ones(1, 3)
    z = x + y
    print(f'{x}+{y}={z}')
    split_line()
    
    # move between numpy and torch
    device = set_device(0)
    x_np = np.array([1.,2.])
    
    # torch.from_numpy() convert np array to torch tensor, 
    # The resulting tensor shares the same memory as the numpy array!!!
    x_torch = torch.from_numpy(x_np).to(device=device)
    print(f'x_np: {x_np}\nx_torch: {x_torch}')
    x_np[0] = 12.
    print(f'x_np: {x_np}\nx_torch: {x_torch}')
    
    # By default, numpy arrays are float64. 
    # By default, most tensors in pytorch are float32.
    x_torch2 = torch.from_numpy(x_np).to(torch.float32).to(device)
    print(f'x_np: {x_np}\nx_torch: {x_torch}\nx_torch2:{x_torch2}')
    
    # .numpy(): torch tensor -> numpy array 
    # still share the same memory !!!
    z = torch.tensor([[1., 2.],
                      [4.,6.]])
    w = torch.tensor([[2., 4.]])
    print(w * z)
    z_np = z.numpy()
    print(f'{z_np}')
    z[0] = 5.
    print(f'{z_np}, z device: {z.device}')
    return None

def run_part3():
    device = set_device(0)
    """NN-related functions"""
    xs = torch.linspace(-3., 3., 100).to(device)
    ys = torch.relu(xs)
    plot(xs.numpy(), ys.numpy())
    ys = torch.sigmoid(xs)
    plot(xs.numpy(), ys.numpy())
    ys = torch.tanh(xs)
    plot(xs.numpy(), ys.numpy())
    x = torch.tensor([[1., 2., 3.],
                      [2., 5., 7.]], device=device)
    print(torch.softmax(x, dim=0))
    
def run_part4():
    """auto-diff"""
    device = set_device(0)
    # By default, tensor with requires_grad = False, 
    # Specify it as true if you need the gradient
    
    # PyTorch makes auto-diff easy 
    # by having tensors keep track of their data and gradients
    x = torch.tensor([1., 2., 3.], requires_grad=True, device=device)
    y = torch.ones((3,), requires_grad=True, device=device)
    print(y, y.data, y.grad)
    L = ((2 * x + y) ** 2).sum()
    
    L.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    # gradients can accumulated, 
    # we can call backwards many times, 
    # but the last computational gragh is freed, 
    # so must re-construct the computational gragh
    loss = ((2 * x + y) ** 2).sum() # reconstruct a computational gragh
    loss.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    
    # multi-loss
    loss2 = (x ** 2).sum()
    loss2.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    
    # !!! stoping gradients and starting grads !!!
    # If you don't specify required_grad=True, 
    # the gradient will always be None
    x = torch.tensor([1., 2., 3.], requires_grad=True, device=device)
    y = torch.ones((3,), device=device)
    L = ((2 * x + y) ** 2).sum()
    L.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    
    # You can turn required_grad back on after initializing a tensor.
    y.requires_grad_()
    L = ((2 * x + y) ** 2).sum()
    L.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    
    # You can cut a gradient by calling y.detach(), 
    # which will return a new tensor with required_grad=False, but share the same data memory. 
    # Note that detach is not an in-place operation! 
    # You can do this during evaluation
    
    x = torch.tensor([1., 2., 3.], requires_grad=True, device=device)
    y = torch.ones((3,), requires_grad=True, device=device)
    # not in-place operation!
    y_detached = y.detach() # id(y_detached) != id(y)
    
    loss = ((2 * x + y_detached)**2).sum()
    loss.backward()
    print(f'x grad: {x.grad} || y grad: {y.grad}')
    
    """
    NOTES:
        * Cannot do any in-place operation on a tensor with 'requires_grad = true',
          e.g. x[0] = 2. is not allowed;
        * Cannot convert a tensor with 'requires_grad = true' to numpy array. Need
          to detach it firstly, i.e. y.detach().numpy();
        * Even though y.detach() returns a new tensor, that tensor occupies the same memory as y. 
          Unfortunately, PyTorch lets you make changes to y.detach() or y.detach.numpy() which will affect y as well! 
          If you want to safely mutate the detached version, you should use y.detach().clone() instead,
          which will create a tensor in new memory.
    """
    
def run_part5():
    """regression task"""
    x = torch.linspace(-5, 5, 100).view(100, 1)
    print(x.shape)
    y_target = torch.sin(x)
    loss_fn = nn.MSELoss()
    plot(x, y_target)   
    
    """modules:
       * nn.Module represents the building blocks of a computation graph.
       * All the classes inside of torch.nn are instances nn.Modules
    """
    class Net(nn.Module):
        """
            *In the __init__ function, any variable that is assigned to self 
            will be automatically added as a 'sub-module' if the variable is also a module.
            *The parameters of a module (and all sub-modules) can be accessed 
            through the 'parameters()' or 'named_parameters()' functions.
            These parameters will automatically have their gradients stored 
            (i.e. requires_grad=True);
            *WARNING: If you want to have a list of modules use 
                      'nn.ModuleList([network1, network2, ...])' so the modules can be tracked.
        """
        def __init__(self, input_size, output_size):
            super(Net, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 32), nn.ReLU(),
                nn.Linear(32, output_size)
            )
            
        def forward(self, x):
            for layer in self.net:
                x = layer(x)
            return x
        
    model = Net(1, 1)
    # don't call model.forward(x)
    # y is in the computational gragh
    y = model(x) 
    print(model.net[0].bias.grad)
    plot(x.detach().numpy(), y.detach().numpy())
    
    # accessing the parameters
    for name, p in model.named_parameters():
        print(name, p.shape)
        
    for p in model.parameters():
        print(p)
    
    # loss function
    loss_func = nn.MSELoss()
    
    # simple training
    for _ in range(1000):
        # compute the batch loss
        y_pred = model(x)
        loss = loss_func(y_pred, y_target)
        loss.backward()

        # We can manually update the parameters by adding the gradient (times a negative learning rate) 
        # and manually zero'ing out the gradients to prevent gradient accumulation.
        for p in model.parameters():
            p.data.add_(-0.01*p.grad)
            p.grad.data.zero_()
    y = model(x) 
    print(model.net[0].bias.grad)
    plot(x.detach().numpy(), y.detach().numpy())
     
    # using built-in optimizers
    
    model = Net(1, 1)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # simple training
    for _ in range(1000):
        # compute the batch loss
        y_pred = model(x)
        loss = loss_func(y_pred, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(model.net[0].bias.grad.data)
    
    # Switches the network to evaluation mode, 
    # disabling things like dropout and batch norm
    model.eval()
    # Temporarily disables all `requires_grad` within the enclosed block
    with torch.no_grad():
        y = model(x) 
    plot(x.numpy(), y.numpy())
    
    # saving model and loading model weights
    # Save a dictionary of parameter names and their values
    # saves everything including parameters and other state values (i.e. batch norm mean)
    PATH = "checkpoint.pt"
    torch.save(model.state_dict(), PATH) 
    
    # initialize a new model and load the trained state dict
    new_model = Net(1, 1)
    new_model.load_state_dict(torch.load(PATH))
    for (name1, val1), (name2, val2) in zip(model.state_dict().items(), new_model.state_dict().items()):
        assert name1 == name2 and torch.equal(val1, val2), f"{name1} and {name2} states differ!"
    new_model.eval()
    with torch.no_grad():
        y = model(x) 
    plot(x.numpy(), y.numpy())
    

def run_part6():
    """distributions in torch"""
    # create distributions by passing the parameters of the distribution
    mean = torch.zeros(1, requires_grad=True)
    std = torch.ones(1, requires_grad=True)
    gaussian = distributions.Normal(mean, std)
    # The two most useful operations you can do with Distribution objects 
    # are 'sample' and 'log_prob'
    sample = gaussian.sample((1,))
    print(sample)
    logprob = gaussian.log_prob(sample)
    print(f'logprob: {logprob}')
    """
     log probability depends on the the parameters of the distribution 
     Calling backward on a loss that depends on log_prob will back-propagate 
     gradients into the parameters of the distribution
     NOTE: this won't back-propagate through the samples (the "reparameterization trick''), 
     unless you use `rsample`, which is only implemented for some distributions, 
     samples from dist.sample() is gradient stopped!
    """ 
    loss = -logprob.sum()
    loss.backward()
    print(mean.grad)
    
    split_line()
    # batch-wise distribution
    # The distributions also support batch-operations. In this case, 
    # all the operations (sample, log_prob, etc.) are batch-wise
    mean_batched = torch.zeros(10)
    std_batched = torch.ones(10)
    gaussian_batched = distributions.Normal(mean_batched, std_batched)
    print(gaussian_batched, gaussian_batched.batch_shape)
    sample_batched = gaussian_batched.sample((5,))
    print(sample_batched.shape)
    logprob_batched = gaussian_batched.log_prob(sample_batched)
    print(logprob_batched)
    
    split_line('multivariate Normal')
    mean = torch.zeros(2)
    cov = torch.tensor(
        [[1., 0.8],
         [0.8, 1.]]
    )
    gaussian = distributions.MultivariateNormal(mean, cov)
    print(gaussian.sample((1,)))
    samples = gaussian.sample((500,))
    plt.gca().set_aspect("equal")
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy())
    plt.show()
    
    # clarify the shape in distribution
    """
        * batch shape: The number of different (possibly non-iid) values you will sample at once
                       Specified when initializing the distribution;
        * event shape: The shape of a single sample from one of the distributions in the batch (dim of random variable)
                       Specified when initializing the distribution;
        * sample shape: The number of iid samples we take from the overall distribution
                        Specified when calling sample()
        In general, when you call dist.sample(sample_shape), 
        the result will have shape (sample_shape, batch_shape, event_shape)
    """
    mean = torch.zeros((3, 2))
    cov = torch.tensor(
        [[[1, 0.8],
          [0.8, 2]],
         [[1, -0.2],
          [-0.2, 1]],
         [[4, 0.6],
          [0.6, 0.5]]])
    gaussian = distributions.MultivariateNormal(mean, cov)
    sample = gaussian.sample((5,))

    print(f'batch shape: {gaussian.batch_shape}') # 3
    print(f'event shape: {gaussian.event_shape}') # 2
    print(f'sample shape: {sample.shape}')        # 5
    
    split_line('categorical distribution')
    probs = torch.tensor([0.5, 0.5])
    cat_dist = distributions.Categorical(probs)
    samples = cat_dist.sample((50,))
    print(samples)
    print(cat_dist.log_prob(samples))
    
    split_line('distribution in neural nets')
    # network will output the parameters of a distribution 
    # (e.g. the mean and covariance of a Gaussian)
    class Net(nn.Module):
        def __init__(self, input_size, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    mean_net = Net(1, 1)
    x = torch.rand(10, 1)
    mean = mean_net(x)
    dist = distributions.Normal(mean, scale=1)
    print(dist)
    
    # Or put a distribution inside a nn module
    class Net(nn.Module):
        def __init__(self, input_size, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return distributions.Normal(x, scale=1)
        
    distribution_network = Net(1, 1)
    x = torch.randn(100, 1)
    distribution = distribution_network(x)
    print(distribution)
    
def run_part7():
    """train a conditional Gaussian model"""
    class GaussianPolicy(nn.Module):
        def __init__(self, input_size, output_size):
            super(GaussianPolicy, self).__init__()
            self.mean_fc1 = nn.Linear(input_size, 32)
            self.mean_fc2 = nn.Linear(32, 32)
            self.mean_fc3 = nn.Linear(32, output_size)
            self.log_std = nn.Parameter(torch.randn(output_size)) 
        
        def forward(self, x):
            mean = F.relu(self.mean_fc1(x))
            mean = F.relu(self.mean_fc2(mean))
            mean = self.mean_fc3(mean)
            return distributions.MultivariateNormal(mean, torch.diag(self.log_std.exp()))
        
    # 1000 samples of 2-dimensional states
    states = torch.rand(1000, 2) - 0.5 
    true_means = states**3 + 4.5*states
    true_cov = torch.diag(torch.tensor([0.1, 0.05]))
    expert_actions = torch.distributions.MultivariateNormal(true_means, true_cov).sample()
    plt.scatter(expert_actions[:,0], expert_actions[:,1])
    plt.show()
    
    policy = GaussianPolicy(2, 2)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    dataset = TensorDataset(states, expert_actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    losses = []
    for epoch in range(200):
        epoch_loss = 0.
        for curr_states, curr_actions in loader:
            dist = policy(curr_states)
            loss = -dist.log_prob(curr_actions).sum()
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().squeeze()
            optimizer.step()
        losses.append(epoch_loss / len(loader))
        print(f'epoch {epoch} || loss: {epoch_loss / len(loader)}')
    
    plt.plot(losses)
    plt.show()
    
    
    policy.eval()
    with torch.no_grad():
        dist = policy(states)
        pred_means = dist.mean.cpu().numpy()
        pred_actions = dist.sample().cpu().numpy()

    plt.figure()
    plt.title("Sampled actions")
    plt.scatter(pred_actions[:,0], pred_actions[:,1], color='r', label='learned policy')
    plt.scatter(expert_actions[:,0], expert_actions[:,1], color='b', label='expert')
    plt.legend()

    plt.figure()
    plt.title("Action means")
    plt.scatter(pred_means[:,0], pred_means[:,1], color='r', label='learned policy')
    plt.scatter(true_means[:,0], true_means[:,1], color='b', label='expert')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    # run_part1()
    # run_part2()
    # run_part3()
    # run_part4()
    # run_part5()
    # run_part6()
    run_part7()