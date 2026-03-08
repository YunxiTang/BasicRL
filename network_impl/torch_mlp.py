"""pytorch-based mlp"""
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils import data
import random
import matplotlib.pyplot as plt
import torch_utils as ptu

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def data_generation(w: torch.tensor, b: torch.tensor, num_samples: int):
    x_dim = w.shape[0]
    x = torch.normal(0., 2.0, (num_samples, x_dim))
    y = torch.sin( torch.matmul(x, w) + b )
    y += torch.normal(0., 0.02, y.shape)
    return x, y

def data_iter(features, labels, batch_size):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]

class MLPScratch(nn.Module):
    def __init__(self, feature_dim, label_dim, num_hiddens, lr, sigma=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.lr = lr

        self.W1 = nn.Parameter(torch.randn(feature_dim, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))

        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_hiddens))

        self.W3 = nn.Parameter(torch.randn(num_hiddens, label_dim) * sigma)
        self.b3 = nn.Parameter(torch.zeros(label_dim))

    def __call__(self, x):
        x = x.reshape((-1, self.feature_dim))
        o = relu( torch.matmul(x, self.W1) + self.b1 )
        o = relu( torch.matmul(o, self.W2) + self.b2 )
        return torch.matmul(o, self.W3) + self.b3

class myMLP(nn.Module):
    def __init__(self, feature_dim, label_dim, num_hiddens, lr, sigma=0.1):
        super(myMLP, self).__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.lr = lr

        self.layers = nn.Sequential(nn.Linear(feature_dim, num_hiddens), 
                                    nn.ReLU(), 
                                    nn.Linear(num_hiddens, num_hiddens), 
                                    nn.ReLU(),
                                    nn.Linear(num_hiddens, label_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def low_api_run():
    # model = MLPScratch(feature_dim=1, label_dim=1, num_hiddens=20, lr=0.01)
    model = myMLP(feature_dim=1, label_dim=1, num_hiddens=20, lr=0.01)

    def loss(y_hat, y):
        return torch.mean( (y_hat - y.reshape(y_hat.shape)) ** 2 / 2. )

    w_true = torch.tensor([1.5,])
    b_true = torch.tensor([1.2,])
    x, y = data_generation(w_true, b_true, 2000)

    batch_size = 20
    lr = 0.03
    optimizer = torch.optim.SGD(model.parameters(), lr)

    for epoch in range(500):
        for feat, label in data_iter(x, y, batch_size):
            y_hat = model(feat)
            optimizer.zero_grad()
            l = loss(y_hat, label)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            print('epoch {} || loss {}'.format(epoch+1, ptu.to_numpy(loss(model(x), y))))

    plt.figure(1)
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(y))
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(model(x)))
    plt.show()
    return ptu.to_numpy(l)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    low_api_run()


