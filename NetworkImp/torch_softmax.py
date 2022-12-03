"""pytorch-based simple classification"""
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import torch_utils as ptu
import torchvision
from torchvision import transforms
from torch.utils import data
from torch_utils import Logger
    

def get_fashion_mnist_labels(labels):
    """get Fashion-MNIST text-labels"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    """plot the images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # if tensors
            ax.imshow(img.numpy())
        else:
            # if PIL
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


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
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def low_api_run(train_iter, test_iter,  batch_size=256, lr=2e-3):
    """softmax regression with low api of torch"""
    
    def softmax(x):
        tmp = torch.sum( torch.exp(x), dim=1, keepdim=True )
        return torch.exp(x) / tmp

    def net(w, b, X):
        outputs = torch.matmul(X.reshape((-1, w.shape[0])), w) + b
        probs = softmax(outputs)
        return probs

    def cross_entropy_loss(y_hat, y):
        return torch.mean( -torch.log(y_hat[range(len(y_hat)), y]) )

    w = torch.normal(0.0, 0.01, (28*28, 10), requires_grad=True)
    b = torch.zeros((10,), requires_grad=True)

    for epoch in range(50):
        acc = 0.0
        i = 0
        for feats, labels in train_iter:
            predicts = net(w, b, feats)
            loss = cross_entropy_loss(predicts, labels)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
                i += 1
                acc += accuracy(predicts, labels)
        test_acc = 0.
        j = 0       
        for test_feat, test_label in test_iter:
            with torch.no_grad():
                test_predicts = net(w, b, test_feat)
                test_acc += accuracy(test_predicts, test_label)
                j += 1
        print('Epoch: {} || Training Accuracy {} || Testing Accuracy {}'.format(epoch+1, 
                                                                                acc/(i*len(labels)), 
                                                                                test_acc/(j*len(test_label))) )

    return acc


def high_api_run(train_iter, test_iter, batch_size, lr):
    """softmax regression with high-level api of pytorch"""
    net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10), nn.Softmax())
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    for epoch in range(50):
        acc = 0.0
        i = 0
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

    # data loading
    batch_size = 100
    train_iter, test_iter = data_loader_fashion_mnist(batch_size)

    # ===========
    lr = 0.1
    high_api_run(train_iter, test_iter, batch_size, lr)
    
