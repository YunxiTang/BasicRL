## linear regression using Jax/Flax
from typing import Any
import numpy as np 
import matplotlib.pyplot as plt 
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
import torch.utils.data as data
from flax.training import train_state, checkpoints
import tqdm


class LinearRegDataset(data.Dataset):
    
    def __init__(self, num_point: int = 1000) -> None:
        super().__init__()
        self._N = num_point

        self._xs = np.random.normal(loc=0.0, size=(self._N, 1))
        noise = np.random.normal(loc=0.0, scale=0.5, size=(self._N, 1))
        self._ys = self._xs * 3 - 1 + noise
        
    def __len__(self):
        return self._N
        
    def __getitem__(self, index):
        return self._xs[index], self._ys[index]


class RegModel(nn.Module):
    
    def setup(self):
        self.net1 = nn.Dense(64)
        self.net2 = nn.Dense(1)
        
    def __call__(self, x):
        x = self.net1(x)
        x = nn.relu(x)
        x = self.net2(x)
        return x
    
    
def numpy_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def loss_func(param, state: train_state.TrainState, batch):
    data_input, labels = batch
    predicts = state.apply_fn(param, data_input)
    loss = jnp.sum(optax.l2_loss(predicts, labels))
    return loss

@jax.jit
def train_step(state: train_state.TrainState, batch):
    loss_grad_fn = jax.value_and_grad(loss_func)
    loss, grads = loss_grad_fn(state.params, state, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(state, data_loader):
    # Training loop
    for epoch in range(1):
        Loss = []
        for batch in data_loader:
            state, loss = train_step(state, batch)
            Loss.append( loss )
            # logging here
    return state, np.mean(Loss)

@jax.jit
def eval_step(state: train_state.TrainState, batch):
    loss = loss_func(state.params, state, batch)
    return loss

def eval_model(state: train_state.TrainState, data_loader):
    Loss = []
    for batch in data_loader:
        Loss.append(eval_step(state, batch))
    return np.mean(Loss)


def trainer(state: train_state.TrainState, train_data_loader, val_data_loader):
    eval_loss = 1e8
    for i in tqdm.tqdm(range(500)):
        state, train_loss = train_model(state, train_data_loader)
        eval_loss_new = eval_model(state, val_data_loader)
        # print(f'epoch {i} || train_loss {train_loss} || eval_loss: {eval_loss_new}')
        if eval_loss_new > eval_loss:
            break
        eval_loss = eval_loss_new
    return state

if __name__ == '__main__':
    
    # dataset
    train_dataset = LinearRegDataset()
    
    # data loader
    data_loader = data.DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=numpy_collate_fn)
    
    # optimizer
    optimizer = optax.adam(learning_rate=0.001)
    
    # model
    model = RegModel()
    rng_key = jax.random.PRNGKey(12)
    x_dummy = jax.random.normal(rng_key, (10, 1))
    param = model.init(rng_key, x_dummy)
    
    # train state
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=param,
        tx=optimizer
    )
    val_dataset = LinearRegDataset(200)
    val_data_loader = data.DataLoader(val_dataset, batch_size=32, collate_fn=numpy_collate_fn)
    
    # train a model
    optimized_state = trainer(model_state, data_loader, val_data_loader)
    
    # visualization 
    binded_model = model.bind(optimized_state.params)
    
    xs = jnp.linspace(-6., 6., 200)[:, None]
    ys = binded_model(xs)
    
    import seaborn as sns
    sns.set()
    sns.scatterplot(x=train_dataset._xs[:,0], y=train_dataset._ys[:,0])
    sns.lineplot(x=xs[:,0], y=ys[:,0], color='r')
    plt.show()
    
    
    
    