"""RNN with Pytorch"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
import matplotlib.pyplot as plt

current_path = os.path.dirname(__file__)

class CustomDataSet(data.Dataset):
    def __init__(self, xs, ys):
        super(CustomDataSet, self).__init__()
        self._xs = xs[:,None]
        self._ys = ys[:,None]

    def __getitem__(self, index):
        return (self._xs[index:index+5], self._ys[index+5])
    
    def __len__(self):
        return self._xs.shape[0]-10
    
def test_single_layer_rnn():
    # network parameters
    # dim of x_{t}
    input_size = 10
    # dim of h_{t-1}
    hidden_size = 10
    # number of rnn layers
    num_layers = 1

    # data parameters
    # sequence length
    seq_len = 5
    # bactch size
    batch_size = 3
    # 
    data_dim = input_size

    # a data sample
    data_sample = torch.randn(seq_len, batch_size, data_dim)

    # official RNN
    o_rnn = nn.RNN(data_dim, hidden_size, num_layers)

    # init hidden state
    h0 = torch.randn(num_layers, batch_size, hidden_size)

    class CustomRNN:
        """
            Single Layer RNN
        """
        def __init__(self, input_size, seq_len, num_layer):
            self.input_size = input_size
            self.num_layer = num_layer
            self.seq_len = seq_len
            self.W_ih = torch.nn.Parameter(o_rnn.weight_ih_l0.T)
            self.b_ih = torch.nn.Parameter(o_rnn.bias_ih_l0)
            self.W_hh = torch.nn.Parameter(o_rnn.weight_hh_l0.T)
            self.b_hh = torch.nn.Parameter(o_rnn.bias_hh_l0)
            self.ht = torch.nn.Parameter(h0)
            self.myoutput = []

        def forward(self, x):
            for i in range(self.seq_len):
                igates = torch.matmul(x[i], self.W_ih) + self.b_ih
                hgates = torch.matmul(self.ht, self.W_hh) + self.b_hh
                self.ht = torch.tanh(igates + hgates)
                self.myoutput.append(self.ht)
            return self.ht, self.myoutput
        
    m_rnn = CustomRNN(input_size, seq_len, num_layers)
    myht, myoutput = m_rnn.forward(data_sample)
    official_output, official_hn = o_rnn(data_sample, h0)
    

def test_two_layer_rnn():
    #network parameters
    input_size = 10
    hidden_size = 20
    num_layers = 2

    #data parameters
    seq_len = 5
    batch_size = 3
    data_dim = input_size

    data_sample = torch.randn(seq_len, batch_size, data_dim)

    #original official rnn in pytorch
    o_rnn = nn.RNN(input_size, hidden_size, num_layers)
    h0 = torch.randn(num_layers, batch_size, hidden_size)

    class CustomRNN2:
        def __init__(self):
            self.W_ih_l0 = torch.nn.Parameter(o_rnn.weight_ih_l0.T)
            self.b_ih_l0 = torch.nn.Parameter(o_rnn.bias_ih_l0)
            self.W_hh_l0 = torch.nn.Parameter(o_rnn.weight_hh_l0.T)
            self.b_hh_l0 = torch.nn.Parameter(o_rnn.bias_hh_l0)
            
            self.W_ih_l1 = torch.nn.Parameter(o_rnn.weight_ih_l1.T)
            self.b_ih_l1 = torch.nn.Parameter(o_rnn.bias_ih_l1)
            self.W_hh_l1 = torch.nn.Parameter(o_rnn.weight_hh_l1.T)
            self.b_hh_l1 = torch.nn.Parameter(o_rnn.bias_hh_l1)
            
            self.ht0 = torch.nn.Parameter(h0[0])
            self.ht1 = torch.nn.Parameter(h0[1])
            
            self.myoutput = []
            
        def forward(self, x):
            """forward

            Args:
                x (torch.tensor): input with shape of (seq_len, batch_size, data_dim)
            """
            for i in range(x.shape[0]):
                # First layer
                igates_l0 = torch.mm(x[i], self.W_ih_l0) + self.b_ih_l0
                hgates_l0 = torch.mm(self.ht0, self.W_hh_l0) + self.b_hh_l0
                self.ht0 = torch.tanh(igates_l0 + hgates_l0)
                
                # second layer
                igates_l1 = torch.mm(self.ht0, self.W_ih_l1) + self.b_ih_l1
                hgates_l1 = torch.mm(self.ht1, self.W_hh_l1) + self.b_hh_l1
                self.ht1 = torch.tanh(igates_l1 + hgates_l1)
                ht_final_layer = [self.ht0, self.ht1]
                self.myoutput.append(self.ht1)
            return self.myoutput, ht_final_layer
        
    myrnn = CustomRNN2()
    myoutput, myht = myrnn.forward(data_sample)
    official_output, official_hn = o_rnn(data_sample, h0)

    print ('myht:')
    print (myht[0])
    print ('official_hn:')
    print (official_hn[0])

    print ("--" * 40)
    print ('myoutput:')
    print (myoutput[2])
    print ('official_output')
    print(official_output[2])


"""
    practical usage of RNN
"""
class MyRNNCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, bias:bool = True, nonlinearity:str = 'tanh'):
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
            h: hidden variable: (batch_size, hidden_size)
        """
        igate = self.input_trans(x)
        hgate = self.hidden_trans(h)
        h_next = self.activation(igate + hgate)
        return h_next
    
    
class MyRNN(nn.Module):
    def __init__(self, 
                input_size:int, 
                hidden_size:int, 
                batch_first:bool = False, 
                num_layers:int = 1,
                nonlinearity: str = 'tanh',
                bias: bool = True, 
                dropout: float = 0):
        """My RNN

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
        self.cells = nn.ModuleList([MyRNNCell(input_size, hidden_size, bias, nonlinearity) for _ in range(num_layers)])
        self.dropout = dropout
        if dropout:
            # Dropout layer
            self.dropout_layer = nn.Dropout(dropout)
            
    def forward(self, input:torch.Tensor, h_0:torch.Tensor):
        """forward pass of rnn module

        Args:
            input (torch.Tensor): shape: [n_steps, batch_size, input_size] if batch_first=False
            h_0 (torch.Tensor): shape:   [num_layers, batch_size, hidden_size]
        """
        is_batched = (input.ndim == 3)
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            # convert into batched data
            input = input.unsqueeze(batch_dim)
            if h_0 is not None:
                h_0 = h_0.unsqueeze(1)
            
        if self.batch_first:
            batch_size, n_steps, _ = input.shape
            # put batch into the middle dimension (standard way)
            input = input.permute(1, 0, 2)
        else:
            n_steps, batch_size, _ = input.shape
        
        if h_0 is None:
            h = [torch.zeros((batch_size, self.hidden_size), device=input.device, dtype=torch.float32) for _ in range(self.num_layers)]
        else:
            h = h_0
            h = list(torch.unbind(h)) 
       
        output = []
        
        for t in range(n_steps):
            inp = input[t]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](inp, h[layer])
                inp = h[layer]
                if self.dropout and layer != self.num_layers-1:
                    inp = self.dropout_layer(inp)
            
            output.append(10.*h[-1])
        
        output = torch.stack(output)
        if self.batch_first:
            output = output.permute(1, 0, 2)
        h_n = torch.stack(h)
        return output, h_n

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    xs = np.linspace(0, 10, 10000, dtype=np.float32)
    ys = np.sin(2 * xs) + 0.2 * xs + np.random.normal(0., 0.05, (10000,)).astype(np.float32)
    dataset = CustomDataSet(xs, ys)
    data_iter = data.DataLoader(dataset, 100, False, drop_last=True)
    loss_func = nn.MSELoss()
    model = MyRNN(input_size=1, hidden_size=1, num_layers=3, batch_first=True)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    hn = None
    for iter in range(40):
        Loss = 0.0
        for feat, label in data_iter:
            output, hn = model(feat, hn)
            hn = hn.detach()
            # print(output.shape) (batch_size, seq_len, input_dim)
            pred = output[:,-1,:]
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.detach().numpy()
            
        print(f'Iter: {iter} || Loss: {Loss}')  
    
    
    