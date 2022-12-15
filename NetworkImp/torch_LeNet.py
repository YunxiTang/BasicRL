import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import torch_utils as ptu
import torchvision
from torchvision import transforms
from torch.utils import data
from torch_utils import Logger, setup_seed



def init_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def print_model_shape_info(net):
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t', X.shape)

# load the dataset
def data_loader_fashion_mnist(batch_size, resize=None):
    """return the training and testing iterators"""
    trans_list = [transforms.ToTensor(),]
    if resize:
        trans_list.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans_list)
    mnist_train = torchvision.datasets.FashionMNIST(root='..\data', train=True, 
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='..\data', train=False, 
                                                   transform=trans, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter

def accuracy(y_hat, y): 
    """get accuracy"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def high_api_run(net, train_iter, test_iter, batch_size, lr):
    """softmax regression with high-level api of pytorch"""
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(50):
        acc = 0.0
        i = 0
        net.train()
        for feats, labels in train_iter:
            predicts = net(feats)
            l = loss(predicts, labels)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                i += 1
                acc += accuracy(predicts, labels)
        test_acc = 0.
        j = 0  
        net.eval()     
        for test_feat, test_label in test_iter:
            with torch.no_grad():
                test_predicts = net(test_feat)
                test_acc += accuracy(test_predicts, test_label)
                j += 1
        print('Epoch: {} || Training Accuracy {} || Testing Accuracy {}'.format(epoch+1, 
                                                                                acc/(i*len(labels)), 
                                                                                test_acc/(j*len(test_label))) )
    return acc

if __name__ == '__main__':

    setup_seed(20)

    net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10), nn.Softmax())
    lenet = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                          nn.Linear(120, 60), nn.ReLU(),
                          nn.Linear(60, 10), 
                          nn.Softmax()
                          )
    lenet.apply(init_normal)
    net.apply(init_normal)
    # data loading
    batch_size = 1000
    train_iter, test_iter = data_loader_fashion_mnist(batch_size)

    # ===========
    lr = 0.003
    
    high_api_run(lenet, train_iter, test_iter, batch_size, lr)
    
