import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 outputs

    def forward(self, x):
        # Max pooling with a 2x2 window
        x = nnf.max_pool2d(nnf.relu(self.conv1(x)), (2, 2))

        # This is the same activation function as above
        # If the size is a square, you can specify a single number as a shortcut
        x = nnf.max_pool2d(nnf.relu(self.conv2(x)), 2)
        x = x.view(-1, num_flat_features(x))

        # Apply fully connected with relu
        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# Predict with random inputs
input = torch.randn(1, 1, 32, 32)
print(f'{input=}')
output = net(input)
print(f'{output=}')

# Backpropagate with random output
net.zero_grad()
rands = torch.randn(1, 10)
output.backward(rands)

# Compute loss for random target
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f'{loss=}')

# Backpropagate loss
net.zero_grad()
print(f'before backprop, {net.conv1.bias.grad=}')
loss.backward()
print(f'after backprop, {net.conv1.bias.grad=}')

# Simple learning step by updating the weights according to the gradient
lr = 0.01
for f in net.parameters():
    f.data = f.data.sub(f.grad.data * lr)

# Learning step with optimizer with torch.optim
optimizer = optim.SGD(net.parameters(), lr=lr)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
