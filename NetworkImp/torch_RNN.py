"""RNN with Pytorch"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
import matplotlib.pyplot as plt
import hydra
import pdb

device = torch.device('cpu')

def grad_clipping(net, theta):
    """
        gradient clipping
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class CustomDataSet(data.Dataset):
    def __init__(self, xs, ys):
        super(CustomDataSet, self).__init__()
        self._xs = xs[:,None]
        self._ys = ys[:,None]

    def __getitem__(self, index):
        return (self._ys[index:index+5], self._ys[index+5])
    
    def __len__(self):
        return self._xs.shape[0]-10


class MyRNNCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, bias:bool = True, nonlinearity:str = 'tanh'):
        """
            single step of a single layer rnn
        """
        super(MyRNNCell, self).__init__()
        self.input_trans = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_trans = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = torch.relu
            
    def forward(self, x, h):
        """
            forward pass of single RNN time step
            x: input of RNNCell: (batch_size, input_size)
            h: hidden variable:  (batch_size, hidden_size)
        """
        igate = self.input_trans(x)
        hgate = self.hidden_trans(h)
        h_next = self.activation(igate + hgate)
        return h_next
    
    
class MyRNN(nn.Module):
    def __init__(self, 
                input_size:int, 
                hidden_size:int, 
                output_size:int,
                batch_first:bool = False, 
                num_layers:int = 1,
                nonlinearity: str = 'tanh',
                bias: bool = True, 
                dropout: float = 0):
        """
        My RNN module (inefficient)

        Args:
            input_size (int): input size 
            hidden_size (int): hidden size
            batch_first (bool, optional): batch first. Defaults to False.
            num_layers (int, optional): number of rnn layers. Defaults to 1.
            nonlinearity (str, optional): nonlinearity. Defaults to 'tanh'.
            bias (bool, optional): use bias. Defaults to True.
            dropout (float, optional): dropout rate. Defaults to 0.
        """
        super(MyRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # support multi-layer
        self.cells = nn.ModuleList([MyRNNCell(input_size, hidden_size, bias, nonlinearity)] + [MyRNNCell(hidden_size, hidden_size, bias, nonlinearity) for _ in range(num_layers-1)])
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = dropout
        if dropout:
            # Dropout layer
            self.dropout_layer = nn.Dropout(dropout)
            
    def init_hidden_state(self, batch_size, device):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device, dtype=torch.float32)
        return h
            
    def forward(self, input:torch.Tensor):
        """
            forward pass of rnn module

        Args:
            input (torch.Tensor): shape: [n_steps, batch_size, input_size] if batch_first is False
        """
        if self.batch_first:
            batch_dim = 0
        else:
            batch_dim = 1
        
        if input.ndim != 3:
            # convert into batched form
            input = input.unsqueeze(dim=batch_dim)
            
        if self.batch_first:
            batch_size, n_steps, _ = input.shape
            # put batch into the middle dimension (standard way)
            input = input.permute(1, 0, 2)
        else:
            n_steps, batch_size, _ = input.shape
        
        h = self.init_hidden_state(batch_size, input.device)
        output = []
        
        for t in range(n_steps):
            hs = []
            inp = input[t]
            for layer in range(self.num_layers):
                hidden_s = self.cells[layer](inp, h[layer])
                hs.append(hidden_s)
                inp = hidden_s
                if self.dropout and layer != self.num_layers-1:
                    inp = self.dropout_layer(inp)
            h = torch.stack(hs)
            output.append(h[-1])
        
        output = torch.stack(output) # [seq_len, batch_size, input_size]
        
        if self.batch_first:
            output = output.permute(1, 0, 2) # [batch_size, seq_len, input_size]
            
        pred = self.fc( output[:, -1, :] )
        
        return pred, h


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, h = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, h


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    
    xs = np.linspace(0, 10, 10000, dtype=np.float32)
    ys = np.sin(2 * xs) + 0.2 * xs + np.random.normal(0., 0.05, (10000,)).astype(np.float32)
    dataset = CustomDataSet(xs, ys)
    data_iter = data.DataLoader(dataset, 50, False, drop_last=True)
    loss_func = nn.MSELoss()
    model = RNN(1, 32, 3, 1)
    # model = MyRNN(input_size=1, hidden_size=32, output_size=1, num_layers=3, nonlinearity='relu', batch_first=True, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    
    for iter in range(10):
        Loss = 0.0
        for feat, label in data_iter:
            pred, _ = model(feat)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(model, 1.0)
            optimizer.step()
            Loss += loss.detach().numpy()
            
        if iter % 20 == 0: 
            print(f'Iter: {iter} || Loss: {Loss}')  
    # breakpoint() # for debugging
    # =================== model evaluation ====================
    model.eval()
    x_help = np.linspace(8, 18, 10000, dtype=np.float32)
    y_help = np.sin(2 * x_help) + 0.2 * x_help
    y_true = []
    y_eval = [y_help[0:5]]

    for i in range(9990):
        with torch.no_grad():
            feat = torch.tensor( y_eval[i:i+5] )
            y_predict, _ = model(feat[None,:,None])
            y_true.append(y_help[i+5])
            y_eval.append(y_predict.squeeze().numpy())
        
    plt.figure()
    plt.plot(y_true, 'r-.')
    plt.plot(y_eval)
    plt.show()