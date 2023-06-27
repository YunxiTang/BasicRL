"""AutoEncoder w/ Pytorch"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0')

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)

class Encoder(nn.Module):
    def __init__(self):
        """Encoder Block"""
        super(Encoder, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 36), nn.ReLU(),
            nn.Linear(36, 18), nn.ReLU(),
            nn.Linear(18, 9), nn.Softmax()
        )

    def forward(self, x):
        return self.encoder_net(x)
    
class Decoder(nn.Module):
    def __init__(self):
        """Decoder Block"""
        super(Decoder, self).__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(9, 18), nn.ReLU(),
            nn.Linear(18, 36), nn.ReLU(),
            nn.Linear(36, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder_net(x)
    
class AutoEncoder(nn.Module):
    def __init__(self):
        """AutoEncoder"""
        super().__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

if __name__ == '__main__':
    
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)

    model = AutoEncoder()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-3,
        weight_decay = 1e-8
    )

    epochs = 100
    outputs = []
    losses = []
    
    for epoch in range(epochs):
        iter_loss = 0.
        for (images, _) in loader:
            # Reshaping the image to (-1, 784)
            images = images.reshape(-1, 28*28).to(device)
            reconstructed = model(images)
            loss = loss_func(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_loss += loss.to('cpu').detach().numpy()
        losses.append(iter_loss)
        print(f'iter: {epoch} || loss: {iter_loss}')
        outputs.append((epochs, images, reconstructed))
    
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-500:])
    plt.show()