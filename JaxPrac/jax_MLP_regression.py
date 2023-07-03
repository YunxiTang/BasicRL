## MLP regression using Jax/Flax
from typing import Any, Callable, Dict, Sequence, Tuple
import numpy as np 
import matplotlib.pyplot as plt 
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
import torch.utils.data as data
import seaborn as sns
from trainer import TrainState, TrainerModule, create_data_loaders
import functools


def target_function(x):
    return np.sin( x * 5 )  - 1


class LinearRegDataset(data.Dataset):
    
    def __init__(self, num_point: int = 1000) -> None:
        super().__init__()
        self._N = num_point

        self._xs = np.random.normal(loc=0.0, scale=2.0, size=(self._N, 1))
        noise = np.random.normal(loc=0.0, scale=0.1, size=(self._N, 1))
        self._ys = target_function(self._xs) + noise
        
    def __len__(self):
        return self._N
        
    def __getitem__(self, index):
        return self._xs[index], self._ys[index]


class RegModel(nn.Module):
    hidden_dims : Sequence[int]
    output_dim : int = 1
    drop_rate: float = 0.2
    
    @nn.compact
    def __call__(self, x, train: bool):
        x = jnp.concatenate(
                [jnp.sin(x), jnp.sin(2*x), jnp.cos(x), jnp.cos(2*x), x], axis=1
            )
        for dims in self.hidden_dims:
            x = nn.Dense(dims, 
                         kernel_init=nn.initializers.kaiming_normal(), 
                         bias_init=nn.initializers.zeros_init())(x)
            z = x
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = nn.silu(x)
            x = nn.Dropout(self.drop_rate)(x, deterministic=not train)
            x = nn.silu(x + z)
        x = nn.Dense(self.output_dim)(x)
        return x
    
class MLPRegTrainer(TrainerModule):
    def __init__(self, hidden_dims: Sequence, **kwargs):
        super().__init__(
            model_class=RegModel,
            model_hparams={'hidden_dims': hidden_dims},
            **kwargs
        )
        
    def create_functions(self):
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

        def train_step(state: TrainState, batch):
            rng = jax.random.fold_in(state.rng, data=state.step)
            loss_grad_fn = jax.value_and_grad(loss_func, has_aux=True)
            ret, grads = loss_grad_fn(state.params, state, batch, rng, train=True)
            loss, new_model_state = ret[0], ret[1]
            state = state.apply_gradients(
                grads=grads, 
                batch_stats=new_model_state['batch_stats'], 
                rng=rng)
            metrics = {'loss': loss}
            return state, metrics
        
        def eval_step(state: TrainState, batch):
            loss, _ = loss_func(state.params, state, batch, state.rng, train=False)
            return {'loss': loss}
        
        return train_step, eval_step


if __name__ == '__main__':
    
    DATASET_PATH = '../data/'
    CHECKPOINT_PATH = '../saved_models/'
    
    train_set = LinearRegDataset()
    val_set = LinearRegDataset(500)
    test_set = LinearRegDataset(300)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_set, val_set, test_set,
        train=[True, False, False],
        batch_size=256
        )
    
    trainer = MLPRegTrainer(
        hidden_dims=[32, 64, 32],
        optimizer_hparams={'lr': 1e-1},
        logger_params={'base_log_dir': CHECKPOINT_PATH},
        exmp_input=next(iter(train_loader))[0:1],
        check_val_every_n_epoch=5,
        debug=False,
        seed=15
    )
    metrics, optimized_state = trainer.train_model(
        train_loader,
        val_loader,
        test_loader=test_loader,
        num_epochs=50
        )
    # # visualization 
    binded_model = trainer.bind_model()
    binded_model = functools.partial(binded_model, train=False)
    xs = np.linspace(-5., 5., 1200)[:, None]
    ys = binded_model(xs)
    
    
    sns.set()
    sns.lineplot(x=xs[:,0], y=ys[:,0], color='r')
    sns.scatterplot(x=train_set._xs[:,0], y=train_set._ys[:,0])
    plt.show()
    
    
    
    