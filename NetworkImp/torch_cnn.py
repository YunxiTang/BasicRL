import torch
import torch.nn as nn
import numpy as np

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def corr2d_multi_channel(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_channel(X, k) for k in K], 0)

def test_corr2d():
    X = torch.tensor(np.arange(9).reshape(3,3), dtype=torch.float32)
    K = torch.tensor(np.arange(4).reshape(2,2), dtype=torch.float32)
    res = corr2d(X, K)
    print('{}\n{}\n{}'.format(X, K, res))

    X = torch.ones((6, 8))
    X[:, 2:6] = 0.0
    K = torch.tensor([[1.0, -1.0]])
    res = corr2d(X.t(), K)
    print('{}\n{}'.format(X,res))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super.__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1,))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

def test_simple_kernel():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0.0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print('{}\n{}'.format(X, Y))

    model = nn.Conv2d(1, 1, (1, 2), bias=False)
    X = X.reshape((1,1,6,8))
    Y = Y.reshape((1,1,6,7))
    
    for i in range(10):
        prediction = model(X)
        loss = torch.sum( (Y - prediction) ** 2 )
        model.zero_grad()
        loss.backward()
        model.weight.data[:] -= 0.03 * model.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {loss:.3f}')
    print(model.weight.data.squeeze())

def test_conv2d():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0.0
    print('{}\n'.format(X))

if __name__ == '__main__':
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 2), stride=(2,2))
    X = torch.rand(size=(1, 1, 8, 8))
    pre = conv2d(X)
    print(pre.shape)

    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    res = corr2d_multi_channel(X, K)
    print(res)

    K = torch.stack((K, K + 1, K + 2), 0)
    print(K.shape)