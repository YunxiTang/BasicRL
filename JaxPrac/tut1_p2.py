from typing import Any
import jax
import optax
from flax.training import train_state, checkpoints
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import torch.utils.data as data
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import tqdm

# ===== prepare some dataset ======
class XORDataset(data.Dataset):
    def __init__(self, size, seed, std=0.1) -> None:
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed)
        self.std = std
        self._generate_continuous_xor()
        
    def _generate_continuous_xor(self):
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int32)
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)
        self.data = data
        self.label = label
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
    
def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    sns.scatterplot(x=data_0[:,0], y=data_0[:,1], edgecolor="#333", label="Class 0")
    sns.scatterplot(x=data_1[:,0], y=data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    plt.show()
    
# ======== sampler ============
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)

# ======== nn Model ===========
class MyModuleTemplate(nn.Module):
    # Some dataclass attributes, like hidden dimension, 
    # number of layers, etc. of the form
    # varname : vartype
    
    def setup(self):
        # Flax uses "lazy" initialization. This function is called once before you
        # call the model, or try to access attributes. In here, define your submodules etc.
        pass

    def __call__(self, x):
        # Function for performing the calculation of the module.
        pass
    
class SimpleClassifier(nn.Module):
    hidden_dim: int
    output_dim: int
    
    def setup(self) -> None:
        self.linear1 = nn.Dense(self.hidden_dim)
        self.linear2 = nn.Dense(self.output_dim)
        
    def __call__(self, x):
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x
    

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc

@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc

@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

def train_model(state, data_loader, num_epochs=100):
    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state

def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")


def visualize_classification(model, data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4,4), dpi=500)
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Let's make use of a lot of operations we have learned above
    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))
    x1 = jnp.arange(-0.5, 1.5, step=0.01)
    x2 = jnp.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')  # Meshgrid function as in numpy
    model_inputs = np.stack([xx1, xx2], axis=-1)
    logits = model(model_inputs)
    preds = nn.sigmoid(logits)
    output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]  # Specifying "None" in a dimension creates a new one
    output_image = jax.device_get(output_image)  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig

    
if __name__ == '__main__':
    # dataset
    dataset = XORDataset(size=200, seed=42)
    print("Size of dataset:", len(dataset))
    print("Data point 0:", dataset[0])
    # visualize_samples(dataset.data, dataset.label)
    
    # sampler
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)
    data_inputs, data_labels = next(iter(data_loader))
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)
    
    # optimizer
    optimizer = optax.adam(learning_rate=0.1)
    
    # model
    model = SimpleClassifier(32, 1)
    rng = jax.random.PRNGKey(21)
    x_init = jax.random.normal(rng, (1, 2))
    param = model.init(rng, x_init)
    
    # model state
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=param,
                                                tx=optimizer)
    
    

    train_dataset = XORDataset(size=2500, seed=42)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)
    
    # train model
    trained_model_state = train_model(model_state, train_data_loader, num_epochs=500)
    
    # save model
    checkpoints.save_checkpoint(
        ckpt_dir='my_checkpoints/',
        target=trained_model_state,
        step=100,
        prefix='mymodel',
        overwrite=True
    )
    
    # load model
    loaded_model_state = checkpoints.restore_checkpoint(
        ckpt_dir='my_checkpoints/',
        target=model_state,
        prefix='mymodel'
    )
    
    # ======== eval model ============
    test_dataset = XORDataset(size=500, seed=123)
    # drop_last -> Don't drop the last batch although it is smaller than 128
    test_data_loader = data.DataLoader(test_dataset,
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   collate_fn=numpy_collate)

    eval_model(loaded_model_state, test_data_loader)
    
    # bind a model
    binded_model = model.bind(param)
    
    _ = visualize_classification(binded_model, dataset.data, dataset.label)
    plt.show()