import torch
from torch import nn 
from xi_rl.infrastructures.logger import Logger
from torch.utils import data
from torchvision import transforms
import utils
import tqdm

Log = Logger('./data')

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

class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 10),
                            nn.ReLU(),
                            nn.Linear(10,10),
                            nn.ReLU(),
                            nn.Linear(10,10),
                            nn.ReLU(),
                            nn.Linear(10,10),
                            nn.ReLU(),
                            nn.Linear(10, 1))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def train(self, train_iter, num_epochs):
        for i in range(num_epochs):
            l_sum = 0.
            for X, y in train_iter:
                y_hat = model(X)
                l = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                l.mean().backward()
                self.optimizer.step()
            l_sum += float(l.sum())
            print('epoch: {} || training loss: {}'.format(i, l))
    

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)

model = net()
# model.apply(init_weights)

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
        test_loss = evaluate_accuracy(model, test_iter)
        # Log.log_scalar(l_iter, 'traininhg', i)
        # Log.log_scalar(test_loss, 'test', i)
        print('epoch: {} || training loss: {} || test loss: {}'.format(i, l_iter, test_loss))

batch_size, lr, num_epochs = 100, 0.01, 100
loss = nn.MSELoss(reduction='none')
trainer = model.optimizer

train_iter = utils.load_array((features, labels), batch_size)
test_iter = utils.load_array((features_test, labels_test), batch_size=10)
# train(model, train_iter, test_iter, loss, trainer, num_epochs)
model.train(train_iter, 100)
