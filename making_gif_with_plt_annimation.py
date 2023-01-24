import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
'''
problem setting 

"AND" gate          "OR" gate           "NAND" gate         "XOR" gate
| x1 | x2 | y |     | x1 | x2 | y |     | x1 | x2 | y |     | x1 | x2 | y | 
| 0  | 0  | 0 |     | 0  | 0  | 0 |     | 0  | 0  | 1 |     | 0  | 0  | 0 |
| 1  | 0  | 0 |     | 1  | 0  | 1 |     | 1  | 0  | 1 |     | 1  | 0  | 1 |
| 0  | 1  | 0 |     | 0  | 1  | 1 |     | 0  | 1  | 1 |     | 0  | 1  | 1 |
| 1  | 1  | 1 |     | 1  | 1  | 1 |     | 1  | 1  | 0 |     | 1  | 1  | 0 |

** graph ** 

^                   ^                   ^                   ^
|                   |                   |                   |  
0          1        1          1        1          0        1          0
|                   |                   |                   |
|                   |                   |                   |
|                   |                   |                   |
0 -- -- -- 0 -->    0 -- -- -- 1 -->    1 -- -- -- 1 -->    0 -- -- -- 1 -->
'''


def gate(x1, x2, y):

    ani = Animator(x1, x2, y)

    x1 = torch.Tensor(x1)
    x2 = torch.Tensor(x2)

    X = torch.stack([x1, x2], dim=1)
    Y = torch.Tensor(y).unsqueeze(1)
    model = nn.Sequential(nn.Linear(2, 8),
                          nn.ReLU(),
                          nn.Linear(8, 1),

                          # nn.Linear(1, 1),
                          nn.Sigmoid(),
                          )

    criterion = nn.BCELoss()
    optimizer = SGD(params=model.parameters(), lr=1)

    for iter in range(70):

        X_, Y_ = np.meshgrid(np.linspace(-.5, 1.5, 100),
                             np.linspace(-.5, 1.5, 100),)
        x_tensor = torch.from_numpy(X_).view(-1).type(torch.float32)
        y_tensor = torch.from_numpy(Y_).view(-1).type(torch.float32)
        x1x2_tensor = torch.stack([x_tensor, y_tensor], dim=1).type(torch.float32)
        z_tensor = model(x1x2_tensor).squeeze()

        x_tensor = x_tensor.view(100, 100).detach().numpy()
        y_tensor = y_tensor.view(100, 100).detach().numpy()
        z_tensor = z_tensor.view(100, 100).detach() # .numpy()
        # z_tensor = (z_tensor == 0.5).numpy()
        # z_tensor = ((z_tensor > 0.5 - 1e-3) & (z_tensor < 0.5 + 1e-3)).numpy()
        z_tensor = (z_tensor > 0.5).numpy()
        ani.get_elements(x_tensor, y_tensor, z_tensor)

        Y_pred = model(X)                       # [4, 1]
        loss = criterion(Y_pred, Y)             # [4, 1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            predicted = (Y_pred > 0.5).float()
            acc = (predicted == Y).float().mean()

            print('Iter : {}\t'
                  'loss : {}\t'
                  'Acc   : {}\t'
                  .format(iter,
                          loss,
                          acc))

    ani.animate()


class Animator(object):
    def __init__(self, x1, x2, y, number_of_dim=3, ):
        if number_of_dim == 3:
            self.X = []
            self.Y = []
            self.Z = []
        self.fig, self.ax = plt.subplots()

        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.y = np.array(y)

        self.gt_xs = self.x1[self.y == 1]
        self.gt_ys = self.x2[self.y == 1]
        self.no_gt_xs = self.x1[self.y == 0]
        self.no_gt_ys = self.x2[self.y == 0]

    def get_elements(self, x, y, z):
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

    def animate_contuor(self, i):

        self.ax.clear()
        plt.contour(self.X[i], self.Y[i], self.Z[i], color='g', level=[0])
        self.ax.scatter(self.no_gt_xs, self.no_gt_ys, c='b')
        self.ax.scatter(self.gt_xs, self.gt_ys, c='r')

    def animate(self):

        anim = FuncAnimation(self.fig, self.animate_contuor, frames=200, interval=100, repeat=False)
        #plt.show()

        f = r"./gif/xor_prob_8.gif"
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writer=writergif)


if __name__ == '__main__':

    # and
    x1 = [0, 1, 0, 1]
    x2 = [0, 0, 1, 1]
    y = [0, 0, 0, 1]

    # or
    x1 = [0, 1, 0, 1]
    x2 = [0, 0, 1, 1]
    y = [0, 1, 1, 1]

    # nand
    x1 = [0, 1, 0, 1]
    x2 = [0, 0, 1, 1]
    y = [1, 1, 1, 0]

    # xor
    x1 = [0, 1, 0, 1]
    x2 = [0, 0, 1, 1]
    y = [0, 1, 1, 0]

    gate(x1, x2, y)