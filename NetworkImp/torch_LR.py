"""Linear Regression with pytorch"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import torch_utils as ptu
import random

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

def low_api_run():

    def prediction(w, b, x):
        return torch.matmul(x, w) + b
    
    def loss(y_hat, y):
        return torch.mean( (y_hat - y.reshape(y_hat.shape)) ** 2 / 2. )
    cfg = OmegaConf.load(r'Config\torch_LR.yaml')
    w_true = torch.tensor([1.5,])
    b_true = torch.tensor([1.2,])
    x, y = data_generation(w_true, b_true, 2000)

    w = torch.normal(0., 0.1, (1, 1), requires_grad=True)
    # w = torch.zeros((1,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    batch_size = cfg.batch_size
    lr = 0.03

    for epoch in range(50):
        for feat, label in data_iter(x, y, batch_size):
            y_hat = prediction(w, b, feat)
            l = loss(y_hat, label)
            l.backward()
            with torch.no_grad():
                w -= lr * w.grad # inplace operation!
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
        with torch.no_grad():
            print('epoch {} || loss {}'.format(epoch+1, ptu.to_numpy(loss(prediction(w, b, x), y))))

    plt.figure(1)
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(y))
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(prediction(w, b, x)))
    plt.show()
    return ptu.to_numpy(l)
    
@hydra.main(config_path='Config', config_name='torch_LR', version_base=None)
def high_api_run(config):
    # data generation
    w_true = torch.tensor([1.5,])
    b_true = torch.tensor([1.2,])
    x, y = data_generation(w_true, b_true, 2000)

    batch_size = config.batch_size
    lr = config.lr

    # build a model and initialization
    prediction = nn.Sequential( nn.Linear(1, 1) )
    prediction[0].weight.data.normal_(0., 0.1)
    prediction[0].bias.data.fill_(0.)
    
    # set loss function
    loss = nn.MSELoss()

    # set optimizer
    optimizer = torch.optim.SGD(prediction.parameters(), lr)

    for epoch in range(50):
        for feat, label in data_iter(x, y, batch_size):
            y_hat = prediction(feat)
            l = loss(y_hat, label.reshape(y_hat.shape))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        with torch.no_grad():
            tmp = prediction(x)
            tmp_l = loss(tmp, y.reshape(tmp.shape))
            print('epoch {} || loss {}'.format(epoch+1, tmp_l))
        
    plt.figure(1)
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(y))
    plt.scatter(ptu.to_numpy(x), ptu.to_numpy(prediction(x)))
    plt.show()
    return ptu.to_numpy(l)


if __name__ == '__main__':
    final_loss1 = low_api_run()
    final_loss2 = high_api_run()
    # print(final_loss1, final_loss2 / 2)