"""Transformer with Pytorch"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
import matplotlib.pyplot as plt


class MyTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_seq):
        super(MyTransformer, self).__init__(nn.Module)
        self.input_size = input_size
        self.output_size = output_size
        self.num_seq = num_seq
        
        self.encoder_layer = nn.TransformerEncoderLayer()
        
    def encoder(self, x):
        """encoder

        Args:
            x (_type_): _description_
        """
        pass
    
    def decoder(self, o):
        """decoder

        Args:
            o (_type_): _description_
        """
        pass
    
    def forward(self, x):
        pass