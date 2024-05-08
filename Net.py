import numpy as np
import torch
import torch.nn as nn
import sympy as sp


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function='Square'):
        super().__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.layer1(x)
        if self.activation_function == 'Square':
            x = x ** 2
        elif self.activation_function == 'Poly2':
            h = x.shape[1] // 2
            x1, x2 = x[:, :h], x[:, h:]
            x = torch.cat((x1, x2), dim=1)

        x = self.layer2(x)
        return x

    def get_polynomial(self):
        x = sp.symbols([f'x{i + 1}' for i in range(self.input_size)])
        w1 = self.layer1.weight.detach().numpy()
        b1 = self.layer1.bias.detach().numpy()
        x = np.dot(w1, x) + b1
        if self.activation_function == 'Square':
            x = x ** 2
        elif self.activation_function == 'Poly2':
            h = x.shape[0] // 2
            x1, x2 = x[:h], x[h:]
            x = np.concatenate((x1, x2), axis=0)
        w2 = self.layer2.weight.detach().numpy()
        b2 = self.layer2.bias.detach().numpy()
        x = np.dot(w2, x) + b2
        x = sp.expand(x[0])
        print('Controller:', x)
        return x


class Learner:
    def __init__(self, net: Net, lr, loops, eps=1e-3):
        self.net = net
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loops = loops
        self.eps = eps

    def train(self, x, y):
        x, y = torch.Tensor(x), torch.Tensor(y)
        for _ in range(self.loops):
            self.optimizer.zero_grad()

            y_pred = self.net(x)
            loss = self.criterion(y_pred, y)
            print(f'iter:{_}, Loss: {loss.item()}')
            if loss.item() < self.eps or _ == self.loops - 1:
                self.net.get_polynomial()
                break
            loss.backward()
            self.optimizer.step()
