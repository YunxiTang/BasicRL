import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

def data_generation(w: torch.tensor, b: torch.tensor, num_samples: int):
    x_dim = w.shape[0]
    x = torch.normal(0., 2.0, (num_samples, x_dim))
    y = torch.matmul(x, w) + b 
    y += torch.normal(0., 1.0, y.shape)
    return x, y

def data_iter(features, labels, batch_size):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]

class BaseModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(n_inputs, 3), nn.ReLU(), nn.Linear(3, n_outputs))
    
    def forward(self, x):
        return self.model(x)

class EnsembleModel(nn.Module):
    """"""
    def __init__(self, n_inputs, n_actions, n_models):
        super().__init__()
        self.models = nn.ModuleList([BaseModel(n_inputs, n_actions) for i in range(n_models)])
        
    def forward(self, x):
        predictions = []
        for model in self.models:
            prediction = model(x)
            predictions.append(prediction)
        predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(predictions, dim=0)
        return mean_prediction
    
    def print_model_info(self):
        for model in self.models:
            for name, param in model.named_parameters():
                print(name, param)
            print('==================')
    
    
def app():
    model = EnsembleModel(1, 1, 10)
    # set loss function
    loss = nn.MSELoss()

    # set optimizer
    optimizer = optim.Adam(model.parameters(), 5e-3)
    w_true = torch.tensor([1.5,])
    b_true = torch.tensor([1.2,])
    x, y = data_generation(w_true, b_true, 2000)

    batch_size = 200

    for epoch in range(500):
        for feat, label in data_iter(x, y, batch_size):
            y_hat = model(feat)
            l = loss(y_hat, label.reshape(y_hat.shape))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        with torch.no_grad():
            tmp = model(x)
            tmp_l = loss(tmp, y.reshape(tmp.shape))
            print('epoch {} || loss {}'.format(epoch+1, tmp_l))
    model.print_model_info()
    plt.figure(1)
    plt.scatter(x.numpy(), y.numpy())
    plt.scatter(x.numpy(), model(x).detach().numpy())
    plt.show()
    
if __name__ == '__main__':
    app()