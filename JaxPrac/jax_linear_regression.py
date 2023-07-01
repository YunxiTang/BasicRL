## MLP regression using Jax/Flax
from typing import Any, Sequence
import numpy as np 
import matplotlib.pyplot as plt 
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
import torch.utils.data as data
from flax.training import checkpoints
import tqdm
from trainer import TrainState
import functools

class LinearRegDataset(data.Dataset):
    
    def __init__(self, num_point: int = 1000) -> None:
        super().__init__()
        self._N = num_point

        self._xs = np.random.normal(loc=0.0, scale=2.0, size=(self._N, 1))
        noise = np.random.normal(loc=0.0, scale=0.2, size=(self._N, 1))
        self._ys = np.sin( self._xs ) * 5 - 1 + noise
        
    def __len__(self):
        return self._N
        
    def __getitem__(self, index):
        return self._xs[index], self._ys[index]


class RegModel(nn.Module):
    hidden_dims : Sequence[int]
    output_dim : int = 1
    drop_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool):
        for dims in self.hidden_dims:
            x = nn.Dense(dims)(x)
            x = nn.silu(x)
            x = nn.Dropout(self.drop_rate)(x, deterministic=not train)
            x = nn.BatchNorm()(x, use_running_average=not train)
            
        x = nn.Dense(self.output_dim)(x)
        return x
    
    
def numpy_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def loss_func(param, state: TrainState, batch, rng, train: bool):
    data_input, labels = batch
    output = state.apply_fn(
        {'params': param, 'batch_stats': state.batch_stats}, 
        data_input,
        train=train,
        rngs={'dropout': rng} if train else None,
        mutable=['batch_stats'] if train else False
        )
    
    if train:
        predicts, new_model_state = output   
    else:
        predicts, new_model_state = output, None
        
    loss = jnp.mean(optax.l2_loss(predicts, labels))
    
    return loss, new_model_state

@jax.jit
def train_step(state: TrainState, batch):
    rng = jax.random.fold_in(state.rng, data=state.step)
    loss_grad_fn = jax.value_and_grad(loss_func, has_aux=True)
    ret, grads = loss_grad_fn(state.params, state, batch, rng, train=True)
    loss, new_model_state = ret[0], ret[1]
    state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'], rng=rng)
    return state, loss

def train_model(state, data_loader):
    # Training loop
    Loss = []
    for batch in data_loader:
        state, loss = train_step(state, batch)
        # print( state.rng )
        Loss.append( loss )
        # logging here
    return state, np.mean(Loss)

@jax.jit
def eval_step(state: TrainState, batch):
    loss, _ = loss_func(state.params, state, batch, state.rng, train=False)
    return loss

def eval_model(state: TrainState, data_loader):
    Loss = []
    for batch in data_loader:
        loss = eval_step(state, batch)
        Loss.append(loss)
    return np.mean(Loss)


def trainer(state: TrainState, train_data_loader, val_data_loader):
    eval_loss = 1e8
    count = 0
    for _ in tqdm.tqdm(range(500)):
        state, _ = train_model(state, train_data_loader)
        
        eval_loss_new = eval_model(state, val_data_loader)

        if eval_loss_new > eval_loss:
            count += 1
            if count > 20:
                break
        else:
            eval_loss = eval_loss_new
        
    return state


if __name__ == '__main__':
    
    # dataset
    train_dataset = LinearRegDataset()
    
    # data loader
    data_loader = data.DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=numpy_collate_fn)
    
    # optimizer
    optimizer = optax.adam(learning_rate=0.001)
    
    # create a model
    model = RegModel([32, 32, 32])
    root_key = jax.random.PRNGKey(12)
    root_key, param_key, dropout_key, dummy_key = jax.random.split(root_key, 4)
    x_dummy = jax.random.normal(dummy_key, (10, 1))
    variables = model.init(
        {'params': param_key}, 
        x_dummy,
        train=False
        )
    
    print(model.tabulate({'params': param_key, 'dropout': dropout_key}, x_dummy, train=True))

    # train state
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables['batch_stats'],
        rng=dropout_key
    )
    val_dataset = LinearRegDataset(700)
    val_data_loader = data.DataLoader(val_dataset, batch_size=64, collate_fn=numpy_collate_fn)
    
    test_dataset = LinearRegDataset(500)
    test_data_loader = data.DataLoader(test_dataset, batch_size=64, collate_fn=numpy_collate_fn)
    
    # train a model
    optimized_state = trainer(model_state, data_loader, val_data_loader)
    
    # # visualization 
    binded_model = model.bind({'params': optimized_state.params, 'batch_stats': optimized_state.batch_stats})
    binded_model = functools.partial(binded_model, train=False)
    xs = jnp.linspace(-5., 5., 600)[:, None]
    ys = binded_model(xs)
    
    import seaborn as sns
    sns.set()
    sns.scatterplot(x=test_dataset._xs[:,0], y=test_dataset._ys[:,0])
    sns.lineplot(x=xs[:,0], y=ys[:,0], color='r')
    plt.show()
    
    
    
    