import torch
from torch import nn 
import torchvision
from torch.utils import data
from torchvision import transforms
import utils
import tqdm

def evaluate_accuracy(model, data_iter):
    if isinstance(net, torch.nn.Module):
        model.eval() # 将模型设置为评估模式
    metric = utils.Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            test_loss = torch.mean( (model(X) - y) ** 2 )
            metric.add(test_loss, y.numel())
    return metric[0] / metric[1] 

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = utils.synthetic_data(true_w, true_b, 1000)
features_test, labels_test = utils.synthetic_data(true_w, true_b, 500)

net = nn.Sequential(nn.Linear(2, 10),
                    nn.ReLU(),
                    nn.Linear(10,10),
                    nn.ReLU(),
                    nn.Linear(10,10),
                    nn.ReLU(),
                    nn.Linear(10,10),
                    nn.ReLU(),
                    nn.Linear(10, 1))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)

net.apply(init_weights)

def train_epoch(model, train_iter, loss, updater):
    metric = utils.Accumulator(1)
    if isinstance(model, torch.nn.Module):
        # set the model in training mode
        model.train()
    for X, y in train_iter:
        y_hat = model(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # use built-in PyTorch optimizers and loss function
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # use self-defined loss and optimizers
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()))
    return metric[0]

def train(model, train_iter, test_iter, loss, updater, num_epoch):
    for i in range(num_epoch):
        l_iter = train_epoch(model, train_iter, loss, updater)
        test_loss = evaluate_accuracy(net, test_iter)
        print('epoch: {} || training loss: {} || test loss: {}'.format(i, l_iter, test_loss))

batch_size, lr, num_epochs = 50, 0.01, 100
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter = utils.load_array((features, labels), batch_size)
test_iter = utils.load_array((features_test, labels_test), batch_size=10)
train(net, train_iter, test_iter, loss, trainer, num_epochs)

