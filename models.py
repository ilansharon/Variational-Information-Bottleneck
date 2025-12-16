import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input=784, hidden1=256, hidden2=128, z=8, classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fcz = nn.Linear(hidden2, z)
        self.fcout = nn.Linear(z, classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        z = self.fcz(x)
        logits = self.fcout(z)

        return logits, z
    
    
    
class MLPVIB(nn.Module):
    def __init__(self, input=784, hidden1=256, hidden2=128, z=8, classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fcmu = nn.Linear(hidden2, z)
        self.fclogvar = nn.Linear(hidden2, z)
        self.fcout = nn.Linear(z, classes)

        self.activation = nn.ReLU()

    def forward(self, x, force_sample=False):
        x = self.flatten(x)
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))

        #now instead of a determinsitic bottleneck layer, 
        #we parametrize z into a gaussian, and introduce some noise into the model
        mu = self.fcmu(h)
        logvar = self.fclogvar(h)
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)

        z = mu.clone()
        if self.training or force_sample:
            z += std * eps
        logits = self.fcout(z)

        return logits, mu, logvar, z
    



