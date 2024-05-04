import numpy as np
import torch
import timeit

from MPC_controler import MPC
from Net import Net, Learner
from Env import get_Env

if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    lr = 1e-3

    begin = timeit.default_timer()
    X, Y = None, None
    example = get_Env(13)
    mpc = MPC(example)
    for i in range(30):
        x, y = mpc.solve()
        if X is None:
            X, Y = x, y
        else:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
    print(X.shape, Y.shape)
    net = Net(example.n_obs, 20, example.u_dim, 'Poly2')
    learner = Learner(net, lr, 300)
    learner.train(X, Y)
    end = timeit.default_timer()
    print(f'Total time:{end - begin}s')
