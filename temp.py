import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST("", train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False,
                          download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # Linear: Fully Connected
        self.fc1 = nn.Linear(28*28, 64)     # input, output
        self.fc2 = nn.Linear(64, 64)     # input, output
        self.fc3 = nn.Linear(64, 64)     # input, output
        self.fc4 = nn.Linear(64, 10)     # input, output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()

X = torch.rand((28, 28))
output = net(X.view(1,28*28))

print(output)